#!/usr/bin/env python3
"""Convert oracle_combined_v7.jsonl to LLM SFT format using current make_prompt().

Filters out stopwords and invalid completions.
Outputs: data/sft_oracle_v7.jsonl
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pedantix_project.llm_policy import make_prompt
from pedantix_project.model import TinyPedantixModel

INPUT  = Path("data/oracle_combined_v7.jsonl")
OUTPUT = Path("data/sft_oracle_v7.jsonl")
MAX_STEPS = 30
EOS = "<|im_end|>"
MIN_IDF = 1.5   # skip stopwords
MAX_ROWS = 150_000  # cap to keep SFT fast

print(f"Loading TinyModel for IDF filtering...", flush=True)
sim = TinyPedantixModel.load("models/tiny_model.json")
def get_idf(word: str) -> float:
    return sim.idf.get(word, 0.0)

STOPWORDS = {
    "de", "la", "le", "les", "du", "des", "un", "une", "en", "et", "ou",
    "est", "au", "aux", "se", "sa", "son", "ses", "ce", "qui", "que", "par",
    "sur", "avec", "dans", "pour", "il", "elle", "ils", "elles", "on",
    "a", "an",
}

n_written = n_skipped_stop = n_skipped_empty = n_skipped_idf = 0
seen: Counter = Counter()

with INPUT.open() as fin, OUTPUT.open("w") as fout:
    for line_no, line in enumerate(fin):
        if n_written >= MAX_ROWS:
            break
        row = json.loads(line)

        word = row["completion"].strip().lstrip()
        if not word:
            n_skipped_empty += 1
            continue
        if word in STOPWORDS:
            n_skipped_stop += 1
            continue
        if get_idf(word) < MIN_IDF:
            n_skipped_idf += 1
            continue

        # Convert history: oracle uses "semantic" key, make_prompt uses "score"
        history_raw = row.get("history", "[]")
        if isinstance(history_raw, str):
            history_raw = json.loads(history_raw)
        history = [
            {
                "guess": s["guess"],
                "exact": s.get("exact", 0),
                "score": s.get("semantic", s.get("score", 0)),
            }
            for s in history_raw
        ]

        prompt_body = make_prompt(history, max_steps=MAX_STEPS)
        prompt = (
            f"<|im_start|>user\n{prompt_body}\n"
            f"/no_think<|im_end|>\n<|im_start|>assistant\nMOT:"
        )
        completion = f" {word}{EOS}"

        out = {
            "prompt": prompt,
            "completion": completion,
            "page_title": row.get("title", ""),
        }
        fout.write(json.dumps(out, ensure_ascii=False) + "\n")
        n_written += 1
        seen[word] += 1

        if n_written % 20_000 == 0:
            print(f"  {n_written:,} written (line {line_no:,})...", flush=True)

print(f"\nDone.")
print(f"  Written:          {n_written:,}")
print(f"  Skipped stopword: {n_skipped_stop:,}")
print(f"  Skipped low-IDF:  {n_skipped_idf:,}")
print(f"  Skipped empty:    {n_skipped_empty:,}")
print(f"  Unique words:     {len(seen):,}")
print(f"  Top 20 words: {seen.most_common(20)}")
print(f"  Output: {OUTPUT}")
