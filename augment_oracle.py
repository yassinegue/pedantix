"""
Augment an oracle dataset with:
  1. Cold-start rows (0-4 words revealed) by truncating existing step-5 rows
  2. Extended rows (15+ words) by continuing from step-14 rows using a greedy oracle

Usage:
    python augment_oracle.py \
        --input data/oracle_20k_10state_v7.jsonl \
        --tiny-model models/tiny_model.json \
        --output data/oracle_augmented.jsonl \
        --n-extra 10 \
        --workers 8
"""

import argparse
import json
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pedantix_project.dataset import WikiPage
from pedantix_project.llm_policy import (
    _apply_llm_exact_reveals,
    _best_rewarding_guess,
    _feedback_step,
    _page_probe_words,
    compact_visible_text,
    format_prompt_for_model,
    make_prompt,
    replay_game,
)
from pedantix_project.model import TinyPedantixModel
from pedantix_project.simulator import PedantixGame


# Per-process model, loaded once in the worker initializer
_sim_model: TinyPedantixModel | None = None
_tiny_model_path: str = ""


def _init_worker(tiny_model_path: str) -> None:
    global _sim_model, _tiny_model_path
    _tiny_model_path = tiny_model_path
    _sim_model = TinyPedantixModel.load(tiny_model_path)


def _build_row(history: list[dict], completion: str, intro: str, title: str) -> dict:
    page = WikiPage(title=title, intro=intro)
    game = PedantixGame(page)
    for step in history:
        _apply_llm_exact_reveals(game, step["guess"])
    visible = compact_visible_text(game)
    prompt_raw = make_prompt(history, max_steps=200, visible_text=visible)
    prompt = format_prompt_for_model(prompt_raw, chat_format="qwen")
    return {
        "prompt": prompt,
        "completion": f" {completion}",
        "text": prompt + f" {completion}",
        "title": title,
        "intro": intro,
        "history": json.dumps(history),
    }


def _process_page_group(args: tuple) -> list[dict]:
    rows_for_page, n_extra, broad_vocab = args
    rows_by_len = {}
    for r in rows_for_page:
        h = json.loads(r["history"]) if isinstance(r["history"], str) else r["history"]
        rows_by_len[len(h)] = (r, h)

    title = rows_for_page[0]["title"]
    intro = rows_for_page[0]["intro"]
    result: list[dict] = []

    # --- Cold starts: steps 0-4 from the step-5 row ---
    if 5 in rows_by_len:
        _, history5 = rows_by_len[5]
        for target_len in range(0, 5):
            trunc = history5[:target_len]
            completion_word = history5[target_len]["guess"]
            try:
                result.append(_build_row(trunc, completion_word, intro, title))
            except Exception:
                pass

    # --- Extended trajectories: steps 15+ from the step-14 row ---
    if n_extra > 0 and 14 in rows_by_len and _sim_model is not None:
        _, history14 = rows_by_len[14]
        page = WikiPage(title=title, intro=intro)

        # Candidate vocabulary: page-specific probes + broad vocab
        try:
            probe = _page_probe_words(page, _sim_model, limit=120)
        except Exception:
            probe = []
        candidates = probe + [w for w in broad_vocab if w not in probe]

        history = list(history14)
        for _ in range(n_extra):
            blocked = {step["guess"] for step in history}
            next_word = _best_rewarding_guess(
                page, _sim_model, history, candidates, blocked=blocked
            )
            if not next_word:
                break
            try:
                result.append(_build_row(history, next_word, intro, title))
            except Exception:
                pass
            feedback = _feedback_step(page, _sim_model, history, next_word)
            history.append(feedback)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--tiny-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-extra", type=int, default=10,
                        help="extra oracle steps to simulate beyond step 14")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()

    print(f"Loading {args.input} ...", flush=True)
    rows_by_page: dict[str, list[dict]] = {}
    with open(args.input) as f:
        for line in f:
            row = json.loads(line)
            key = row["title"]
            rows_by_page.setdefault(key, []).append(row)

    n_pages = len(rows_by_page)
    print(f"  {sum(len(v) for v in rows_by_page.values())} rows across {n_pages} pages", flush=True)

    # Build a broad vocabulary from oracle completions for greedy oracle
    print("Building broad vocab from completions ...", flush=True)
    from collections import Counter
    import re
    counter: Counter = Counter()
    for rows in rows_by_page.values():
        for r in rows:
            w = r.get("completion", "").strip().lower()
            w = re.sub(r"[^a-zA-ZÀ-ÿ\-]", "", w)
            if w:
                counter[w] += 1
    broad_vocab = [w for w, _ in counter.most_common(5000)]
    print(f"  broad vocab: {len(broad_vocab)} words", flush=True)

    page_groups = [
        (rows, args.n_extra, broad_vocab) for rows in rows_by_page.values()
    ]

    print(f"Augmenting with {args.workers} workers ...", flush=True)
    written = 0
    with open(args.output, "w") as out_f:
        with multiprocessing.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(args.tiny_model,),
        ) as pool:
            try:
                from tqdm.auto import tqdm
                it = tqdm(
                    pool.imap_unordered(_process_page_group, page_groups, chunksize=4),
                    total=n_pages,
                    desc="pages",
                )
            except ImportError:
                it = pool.imap_unordered(_process_page_group, page_groups, chunksize=4)
            for new_rows in it:
                for row in new_rows:
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
                if written % 5000 == 0 and written > 0:
                    print(f"  written {written} rows so far", flush=True)

    print(f"Done. {written} new rows written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
