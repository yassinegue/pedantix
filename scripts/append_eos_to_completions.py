"""Append the Qwen3 turn-end token to every SFT completion.

The original v2_sft_mix.jsonl trained the model on completions like ' philosophie'
with no stop signal, so at eval time the model never terminated and emitted
subword concatenations (annee + aux + esques = anneauxesques). Appending
<|im_end|> (Qwen3's actual eos_token, id 151645) teaches the model to stop after
one word; generation then halts naturally on the existing eos_token_id default.

Reads the source JSONL, appends '<|im_end|>' to each row's `completion` and
mirrors the change into the `text` field if present, then writes to --output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


STOP = "<|im_end|>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/v2_sft_mix.jsonl")
    parser.add_argument("--output", default="data/v2_sft_mix_eos.jsonl")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    n_unchanged = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            row = json.loads(line)
            comp = row.get("completion")
            if comp is None:
                raise ValueError(f"row missing 'completion' field: {row.keys()}")
            if comp.endswith(STOP):
                n_unchanged += 1
                new_comp = comp
            else:
                new_comp = comp + STOP
            row["completion"] = new_comp
            if "text" in row and isinstance(row["text"], str):
                prompt = row.get("prompt", "")
                if prompt and row["text"].endswith(comp):
                    row["text"] = row["text"][: -len(comp)] + new_comp
                elif prompt:
                    row["text"] = prompt + new_comp
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    print(json.dumps({
        "input": str(src),
        "output": str(dst),
        "rows": n,
        "rows_already_had_stop": n_unchanged,
        "stop_token": STOP,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
