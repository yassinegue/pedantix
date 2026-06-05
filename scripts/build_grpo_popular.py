#!/usr/bin/env python3
"""Generate GRPO-format training data from popular Wikipedia pages.

Uses filtered_pages.jsonl (pageview-ranked) as source — no DO API needed.
The stochastic oracle (TinyModel soft_oracle_guess) generates warm-start
histories, producing diverse game states across many early/mid game turns.

Output columns match what train_llm_grpo's reward_func expects:
  prompt, completion, title, intro, history (JSON string)

Usage:
    python scripts/build_grpo_popular.py \
        --pages data/filtered_pages.jsonl \
        --tiny-model models/tiny_model.json \
        --output data/grpo_popular.jsonl \
        --n-pages 2000 \
        --states-per-page 8 \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pedantix_project.dataset import WikiPage
from pedantix_project.llm_policy import (
    _feedback_step,
    compact_visible_text,
    format_prompt_for_model,
    make_prompt,
    replay_game,
    soft_oracle_guess,
)
from pedantix_project.model import TinyPedantixModel

MAX_STEPS = 30
# Warm-start checkpoints: record a GRPO prompt at these turn counts
WARM_STOPS = [0, 3, 6, 10, 15, 20]


def build_grpo_rows(
    page: WikiPage,
    sim_model: TinyPedantixModel,
    rng: random.Random,
    states_per_page: int,
    temperature: float,
    top_k: int,
) -> list[dict]:
    """Run the oracle on one page, recording game states at WARM_STOPS turns."""
    rows: list[dict] = []
    history: list[dict] = []
    guessed: set[str] = set()

    stops = sorted(random.Random(rng.random()).sample(
        WARM_STOPS, min(states_per_page, len(WARM_STOPS))
    ))

    for turn in range(MAX_STEPS):
        if not history or turn in stops:
            # Record this game state as a GRPO training example
            game = replay_game(page, sim_model, history)
            visible = compact_visible_text(game)
            prompt = make_prompt(history, max_steps=MAX_STEPS, visible_text=visible)
            formatted = format_prompt_for_model(prompt, chat_format="qwen")
            # Peek at oracle's next word as the stored completion (ignored by GRPO)
            next_word = soft_oracle_guess(
                page, sim_model, history,
                temperature=temperature, top_k=top_k,
            )
            completion = f" {next_word}" if next_word else " inconnu"
            rows.append({
                "prompt": formatted,
                "completion": completion,
                "title": page.title,
                "intro": page.intro,
                "history": json.dumps(history, ensure_ascii=False),
            })
            if len(rows) >= states_per_page:
                break

        # Advance game with oracle guess
        word = soft_oracle_guess(
            page, sim_model, history,
            temperature=temperature, top_k=top_k,
        )
        if not word or word in guessed:
            break
        guessed.add(word)
        step = _feedback_step(page, sim_model, history, word)
        history = history + [step]
        if step.get("solved"):
            break

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="data/filtered_pages.jsonl")
    ap.add_argument("--tiny-model", default="models/tiny_model.json")
    ap.add_argument("--output", default="data/grpo_popular.jsonl")
    ap.add_argument("--n-pages", type=int, default=2000)
    ap.add_argument("--states-per-page", type=int, default=8,
                    help="GRPO prompts per page (spread across warm-stop turns)")
    ap.add_argument("--top-k-pages", type=int, default=20000,
                    help="Sample only from the top-K most popular pages")
    ap.add_argument("--temperature", type=float, default=4.0,
                    help="Oracle softmax temperature — higher = more diverse histories")
    ap.add_argument("--top-k", type=int, default=12,
                    help="Oracle top-K sampling")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print(f"Loading TinyModel from {args.tiny_model} ...", flush=True)
    sim_model = TinyPedantixModel.load(args.tiny_model)

    print(f"Loading pages from {args.pages} ...", flush=True)
    with open(args.pages, encoding="utf-8") as f:
        all_pages = [json.loads(l) for l in f]

    # filtered_pages is already sorted by pageview_rank asc (most popular first)
    # Take top-K, then sample n-pages from those
    pool = all_pages[: args.top_k_pages]
    # Filter: need at least 300 chars of intro to be playable
    pool = [p for p in pool if len(p.get("intro", "")) >= 300]
    print(f"Pool after intro filter: {len(pool)} pages", flush=True)

    selected = rng.sample(pool, min(args.n_pages, len(pool)))
    print(f"Selected {len(selected)} pages, generating GRPO prompts ...", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for i, p in enumerate(selected):
            page = WikiPage(title=p["title"], intro=p["intro"])
            try:
                rows = build_grpo_rows(
                    page, sim_model, rng,
                    states_per_page=args.states_per_page,
                    temperature=args.temperature,
                    top_k=args.top_k,
                )
            except Exception as exc:
                print(f"  [skip] {page.title}: {exc}", flush=True)
                continue
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += len(rows)
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(selected)}] {total} examples written", flush=True)

    print(f"\nDone: {total} GRPO examples from {len(selected)} pages → {out_path}", flush=True)


if __name__ == "__main__":
    main()
