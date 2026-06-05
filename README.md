
# WikiBlind

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

### What we tried

Each iteration below was a proof-of-concept on Qwen3-4B before scaling to larger models. Every step produced a small reward improvement but the model still failed on cold starts (turn 0, blank page).

| POC iteration | What changed | Reward trend | Cold-start failure |
|---------------|-------------|--------------|-------------------|
| Tabular RL (Q-table, 5k vocab) | Baseline: no LLM, fixed vocabulary | Flat ~2% solve | No page context at all |
| SFT warm-up on oracle trajectories | Format learning: model outputs `MOT: word` correctly | Better format, no strategy | Learns format, not when/why to guess what |
| SFT → GRPO (bandit, frozen snapshots) | Reward signal introduced; model starts adapting guesses | Reward slowly rising | Trained on turn-18 states, collapses at turn 0 |
| + DAgger (oracle injection on bad turns) | Oracle rescues stuck games, provides positive examples | Small further gain | GRPO gradient dominates; oracle signal washed out |
| + Reward reshaping (IDF weighting, near-solve shaping) | Denser gradient; less farming of generic words | Reward visibly improving | Still 0% solve on cold-start eval — distribution mismatch unresolved |

The persistent failure mode: the model learned a fixed set of ~10 generic French words ("egalement", "premier", "europe") that score slightly positive on almost any page, and never adapted to page-specific content. Each reward tweak pushed the training curve up a few points but the cold-start distribution mismatch — training on mid-game oracle states while evaluating from turn 0 — meant improvements never transferred to eval.

### Next: Claude strategy distillation

1. **Claude API baseline** — run Claude Haiku on 100 pages, measure solve rate (~15–40% expected). This sets the target and validates the task.
2. **Generate SFT data** — Claude plays 1,000+ games; each step becomes a training example `(game_state → word)`. Unlike previous SFT, these examples show *state-conditional* choices: Claude uses semantic score feedback to narrow the topic domain.
3. **Fine-tune Qwen3-4B** on Claude trajectories — model learns strategy, not just format.
4. **GRPO refinement** — run GRPO on the Claude-SFT checkpoint, now with near-solve shaping active. Starting from a model that already plays strategically rather than from random guessing.

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

Training runs on a local NVIDIA A5000 (24 GB VRAM). At this VRAM budget, only Qwen3-4B fits at LoRA rank 32; a 1500-step GRPO run takes ~4 hours. Scaling to larger models (Qwen3-8B, Llama-3-8B) or running more parallel rollouts would require a 40–80 GB GPU (A100/H100).

## References

- French Word2Vec: https://fauconnier.github.io/#data
- TRL (GRPO): https://github.com/huggingface/trl
- Qwen3-4B: https://huggingface.co/Qwen/Qwen3-4B
- Cloudflare Durable Objects: https://developers.cloudflare.com/durable-objects/
