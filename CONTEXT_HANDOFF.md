# Pedantix RL — Context Handoff

## The Game

**Pedantix** is a French Wikipedia word-guessing game. Given a hidden article, the player guesses words one at a time. Each guess scores based on cosine similarity to the article's content words. Words above a semantic threshold get "revealed" in the article text. **Goal: guess all title words** (e.g. "Napoléon Bonaparte" = 2 title words to reveal). Solving gives +1000 reward. Invalid/irrelevant guesses give −40.

## The Approach: Vocab-Action Policy

Instead of free-text generation, we built a **fixed-vocabulary classifier**:
- **Backbone**: Qwen3 LLM (frozen), encodes the game state (prompt = visible article text + guess history)
- **Action head**: Linear layer (hidden_size → N_actions), selects the next word to guess
- **Action space**: 8000 base French words + per-episode title words + TinyModel neighbors = ~9478 total
- **Training**: GRPO (group relative policy optimization, 16 generations per episode) + BC from a 1-step greedy oracle (DAgger)

### Key mechanism: Episode-local dynamic expansion (ep_mask)
During training, for each episode the action space is expanded with:
- The episode's title words (guaranteed correct answers)
- Their top-20 TinyPedantixModel neighbors

This ensures the model CAN pick the title word. Without it, proper nouns never appear in the base 8000 words and the model never gets +1000 reward (cold-start failure).

**TinyPedantixModel**: co-occurrence similarity model, `neighbors` dict maps word → {neighbor: cosine_sim}. Used for expansion and semantic similarity scoring.

### Training data: `data/oracle_v7_1word.jsonl` (72,725 rows)
Each row = a game state (title, intro, history of 5–14 past guesses) where the oracle's best next action is a title word. **88.8% of states have exactly 1 title word remaining** — they are genuine near-solve states.

### Key code files
- `pedantix_project/vocab_action_policy.py` — main training + eval logic
- `pedantix_project/cli.py` — CLI entry points (`llm-vocab-grpo`, `llm-vocab-eval`)
- `pedantix_project/llm_policy.py` — prompt formatting, game simulation, oracle
- `pedantix_project/model.py` — TinyPedantixModel

---

## Experiment History

| Exp | Model | Key config | avg_solve (best window) | Peak | Verdict |
|-----|-------|-----------|------------------------|------|---------|
| exp7 | Qwen3-0.6B frozen | rollout=2, expand=20, dagger_every=10, bc_steps=50 | 0.042 (steps 901-1000) | 0.133 | ✅ Baseline that works |
| exp8 | Qwen3-0.6B frozen | rollout=3, expand=40 | abandoned ~0.020 | — | ❌ Too slow (38s/step), underperformed |
| exp9 | Qwen3-0.6B frozen | rollout=2, expand=20, dagger_every=20, bc_steps=25 | 0.028 (steps 701-800, still rising) | 0.160 | ✅ Reduced DAgger = less oscillation, trend upward; crashed at step 844 |
| exp10 | Qwen3-0.6B + LoRA r=16 warm-start from exp7 | Same as exp9 + LoRA unfreezes at step 101 | 0.047 flat | 0.154 | ❌ No improvement vs exp7 |
| exp11 | **Qwen3-4B frozen** | rollout=2, expand=20, dagger_every=20, bc_steps=25, bc_warmup=100 | **0.060 (steps 801-900)** | **0.156** | ✅ 2× better than exp7 at every window |

### exp10 LoRA postmortem
LoRA **did** train (lora_B norms: 0→0.13-0.53, grad_norm +20% when unfrozen). No gradient checkpointing bug. But GRPO with ~4.5% solve rate is too sparse to guide 196 LoRA matrices usefully — gradients wash out to noise.

### exp11 results (Qwen3-4B frozen, 1000 steps complete)
```
steps   1-100: avg_solve=0.0021  peak=0.031  avg_reward= -51.6
steps 101-200: avg_solve=0.0109  peak=0.068  avg_reward= -22.1
steps 201-300: avg_solve=0.0257  peak=0.129  avg_reward= +13.6
steps 301-400: avg_solve=0.0359  peak=0.147  avg_reward= +39.0
steps 401-500: avg_solve=0.0444  peak=0.135  avg_reward= +61.2
steps 501-600: avg_solve=0.0473  peak=0.121  avg_reward= +67.9
steps 601-700: avg_solve=0.0504  peak=0.127  avg_reward= +75.8
steps 701-800: avg_solve=0.0488  peak=0.154  avg_reward= +71.5
steps 801-900: avg_solve=0.0602  peak=0.156  avg_reward=+100.7
steps 901-1000:avg_solve=0.0555  peak=0.145  avg_reward= +89.0
```
Saved at `models/exp11_qwen4b_frozen`.

---

## The Architectural Ceiling (Critical Finding)

**Setup**: 88.8% of training states have 1 title word remaining. Title word is always in episode-expanded action space (~1478 words). Rollout=2, 16 generations.

**Theoretical maximum solve_rate** with a perfect policy = **0.50** (all 256 trajectories solve at step 1 → 256/512).

**Exp11 best = 0.060 = 12% of theoretical maximum.**

**Implied**: the model assigns only ~3.5% probability to the *correct* title word per step (50× better than random 0.07%, but nowhere near the 50%+ needed for reliable solves).

**Why**: The frozen backbone's hidden states + linear action head cannot reliably identify which of ~1478 episode-expanded words is the correct title word. The linear head must generalize across 72,000 diverse French Wikipedia game states — it's essentially a 1-of-9478 classification problem where every article has a different answer. BC+GRPO over 1000 steps pushes toward the answer but can't reach it reliably.

### Test-time eval (truly unseen pages, from empty history)
- **Without dynamic expansion**: model spams title-only words (`hoverboard`, `pomlt`...) — 0% solve, reward −40 every step. The ep_mask during training created a "vocabulary availability = article identity" shortcut that vanishes at test time.
- **With test-time dynamic expansion** (base vocab only, expand from revealed words): picks sensible French words (`depasser`, `egalement`, `lait`) but still 0% solve — model was never trained on empty-history states, and rare proper-noun titles are unreachable through neighbor expansion chains.

---

## Config notes for future experiments

Standard env vars needed:
```bash
HF_HOME=models/hf_cache
HF_HUB_CACHE=models/hf_cache/hub
TRANSFORMERS_CACHE=models/hf_cache/transformers
```

Available cached models: `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3.5-4B`

GPU: NVIDIA RTX A5000, 25.3 GB VRAM. Qwen3-4B frozen uses ~8GB, leaving ~17GB headroom. Step time for frozen-4B eval: ~27s/step.

---

## What to Try Next

The vocab-action classifier has a ceiling at ~6-10% training solve_rate (12% of theoretical max). To break through, the fundamental architecture needs to change.

**Proposed direction: free-text generation with reward signal**

Instead of classifying over a fixed vocab, let the LM *generate* the title word as text:
- Input: game state prompt (same as before)
- Output: the model generates a word token (or a few tokens for multi-token words)
- Score: run the generated word through the Pedantix game simulator
- Train with GRPO on the generation log-probs

This uses the LM's pre-trained knowledge of which French words are contextually appropriate, instead of a learned linear mapping over 9478 classes. The model already knows "Napoléon" and "Bonaparte" — it just needs to be steered toward them from game context.

Key challenges:
- Generation is slower than classification (autoregressive vs single forward pass)
- Need to handle multi-token words (e.g. "New York")
- GRPO on generation is standard (this is the original GRPO paper setup)

There are already earlier experiments in this codebase (`llm-vocab-grpo` vs the older `llm-grpo` / `llm_grpo_visible` experiments in `models/`) that tried free-text generation but presumably had other issues. Worth revisiting with the lessons learned here (dynamic expansion, near-solve training states, DAgger BC).
