
# Palimot

A French Wikipedia word-guessing game where players uncover the title of a hidden article one word at a time — with solo, multiplayer, and speedrun modes, plus an AI agent as a competitive baseline.

## Game Modes

**Solo (Daily)** — One article per day, same for everyone. Guess until you find the title; your time and guess count are saved.

**Duel / Co-op** — Real-time multiplayer via WebSocket rooms. In duel mode, two players race on the same article. In co-op mode, they share guesses to solve it together.

**Theme Speedrun** — Pick a theme (history, geography, sport, art, science, cinema, music, literature) and solve 3 articles from that theme as fast as possible. Total time across all 3 pages is your score.

The AI agent plays the same game and posts a reference time — the speedrun baseline humans try to beat.

## Architecture

### Web Application
```
web/frontend/         — React + Vite + Tailwind UI
web/src/              — Cloudflare Workers backend (TypeScript)
  index.ts            — API router: /api/daily, /api/themes, /api/rooms
  room.ts             — Durable Object: real-time room state + WebSocket fan-out
  game.ts             — Core game logic (shared server/client)
```

**Stack:** React frontend, Cloudflare Workers + Durable Objects for real-time multiplayer, D1 (SQLite) for persistence.

### Game Engine (Python)
```
pedantix_project/simulator.py     — masked game, guess scoring, reveal logic
pedantix_project/model.py         — TinyModel: IDF + co-occurrence similarity
pedantix_project/dataset.py       — WikiPage, Wikipedia API + bulk dump ingestion
pedantix_project/llm_policy.py    — reward function, training loops, oracle
pedantix_project/claude_agent.py  — Claude API agent for baseline + data generation
```

### Data
1,007,585 French Wikipedia articles filtered from the Wikimedia dump (redirects, disambiguation pages, short articles removed). A new daily article is selected each day at noon Paris time.

## AI Agent (Speedrun Baseline)

The agent is trained to play the game and post a reference solve time per theme. Players compete to beat it.

### Iteration history (all POC runs on Qwen3-4B before scaling)

Each step below produced a measurable reward improvement but failed on cold starts (turn 0, blank page). The core problem throughout: training on mid-game oracle snapshots while evaluating from turn 0 — a distribution mismatch no amount of reward tuning could fix.

| POC iteration | What changed | Reward trend | Why it still failed |
|---|---|---|---|
| Tabular RL (Q-table, 5k vocab) | Baseline — no LLM, fixed word list | Flat | No page context; vocabulary too small to cover most titles |
| SFT warm-up on oracle trajectories | Taught format: model outputs `MOT: word` | Format correct, no strategy | Learns to imitate structure, not when or why to guess what |
| SFT → bandit GRPO (frozen snapshots) | Reward signal introduced | Slowly rising | Trained on turn-18 oracle states; collapses at turn-0 eval |
| + DAgger (oracle injection on bad turns) | Oracle rescues stuck games mid-episode | Small further gain | GRPO gradient dominates oracle signal; model ignores injections |
| + Reward reshaping (IDF, near-solve shaping) | Denser gradient; penalises generic-word farming | Visibly improving | Cold-start distribution mismatch still unresolved; 0% eval solve |

The persistent failure mode: the model settled on ~10 generic French words ("egalement", "premier", "europe") that score slightly positive on almost any page. Each reward tweak nudged the training curve up but the underlying mismatch — training on revealed mid-game states, evaluating from a blank board — meant nothing transferred to eval.

### Final model — Trajectory GRPO + DAgger on Qwen3.5-27B (H100, 20 hours)

The fix: replace frozen oracle snapshots with a **live game buffer**. The model plays full games from turn 0, generating its own training states. GRPO updates on those live rollouts. The model experiences the full causal chain: a bad guess at turn 1 leads to a harder state at turn 5.

**Architecture:**
- `OnlineGameBuffer`: pool of 32 active games running in parallel. At each step, sample a batch, generate 4 candidate words per game (constrained to 96k valid French words via trie), score each with the reward function, GRPO update, advance each game.
- **DAgger rescue**: if a game's last 3 turns all score below threshold, inject an oracle word. Prevents degenerate spirals where the model gets stuck repeating high-shaping words.
- **Variable seeding**: pre-play 0–15 oracle turns on some games so the model sees diverse mid-game states during early training, not just blank boards.
- **Constrained decoding**: trie over `fr_FR.dic` (96k words) forces all outputs to valid French — zero garbage tokens, zero wasted turns.

**Scale-up:** moved from Qwen3-4B (local A5000, 24 GB) to **Qwen3.5-27B on H100 80 GB** (Omniva cluster). ~80M trainable LoRA parameters (0.30% of 27B). Training ran for ~20 hours across multiple jobs.

**Key engineering fix:** gradient checkpointing + PEFT on PyTorch 2.8 produced NaN gradients in 500/896 LoRA tensors on every step. Root cause: calling `gradient_checkpointing_enable()` after `get_peft_model()` means PEFT's internal hook never fires. Fix: enable on base model first, then wrap with PEFT (documented in `BUG.md`).

**Results:** reward EMA improved from **−57 → −12.6** over the first 918 steps — 44 points of improvement. Qualitatively, the model identifies article domains within 3–5 turns and builds a semantic map rather than spamming function words. On a Taylor Swift article: `musique (52) → chanteuse (65) → album (71) → pop (73)` — clear domain-narrowing behavior absent in all previous runs. Solve rate went from 0% on cold starts to 18%. 

## Reward Function

```
reward = step_penalty (−10)
       + exact_match_info × 0.8          (IDF-weighted)
       + semantic_info × 0.01            (capped, anti-spam)
       + title_semantic_info × 3.0       (proximity to title)
       + title_hit_info × 25.0           (IDF-weighted title hit)
       + title_words_revealed × 200
       + near_solve_bonus                (max cosine sim to unrevealed title word × 30)
       + [solved: +1000 − 2 × steps]
```

## Quick Start

```bash
pip install 'datasets>=2.19' 'trl>=0.18' 'peft>=0.11' 'accelerate>=0.30' anthropic

# Build similarity model
python3 -m pedantix_project.cli train \
  --pages data/clean_pages.jsonl --output models/tiny_model.json

# Run Claude baseline (needs API key)
ANTHROPIC_API_KEY=sk-ant-... bash scripts/run_claude_eval.sh

# Generate Claude SFT data (1k pages)
N_PAGES=1000 ANTHROPIC_API_KEY=sk-ant-... bash scripts/run_claude_sft_gen.sh

# Fine-tune on Claude data
bash scripts/run_v3_sft.sh

# GRPO training
bash scripts/run_v2_grpo.sh
```

## Compute

POC iterations ran on a local NVIDIA A5000 (24 GB VRAM) — Qwen3-4B only, 1500-step runs at ~4 hours each. The final model used an **NVIDIA H100 80 GB** on the Omniva cluster (SLURM, PyXIS container `nvcr.io#nvidia/pytorch:25.06-py3`), enabling Qwen3.5-27B with LoRA rank 16. Total H100 training time across all jobs: ~20 hours. Compute credits provided by the Omniva cluster (Inria / École Polytechnique) and DigitalOcean (CS 153 partnership).

## References

- French Word2Vec: https://fauconnier.github.io/#data
- TRL (GRPO): https://github.com/huggingface/trl
- Qwen3-4B: https://huggingface.co/Qwen/Qwen3-4B
- Cloudflare Durable Objects: https://developers.cloudflare.com/durable-objects/
