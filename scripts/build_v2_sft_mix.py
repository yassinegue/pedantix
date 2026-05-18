"""Build the two-tier SFT data mix described in plan v2.

80% near-solve states from oracle_v7_1word.jsonl, 20% 10-state trajectory
states from oracle_20k_10state_v7.jsonl, shuffled with a fixed seed.
Output goes to data/v2_sft_mix.jsonl.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return handle.readlines()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v2 SFT data mix")
    parser.add_argument("--near-solve", default="data/oracle_v7_1word.jsonl")
    parser.add_argument("--multi-state", default="data/oracle_20k_10state_v7.jsonl")
    parser.add_argument("--output", default="data/v2_sft_mix.jsonl")
    parser.add_argument("--multi-state-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--max-near-solve",
        type=int,
        default=0,
        help="cap on near-solve rows used (0 = use all)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    near = read_lines(Path(args.near_solve))
    multi = read_lines(Path(args.multi_state))

    if args.max_near_solve and args.max_near_solve < len(near):
        rng.shuffle(near)
        near = near[: args.max_near_solve]

    fraction = args.multi_state_fraction
    near_count = len(near)
    if fraction <= 0.0:
        multi_count = 0
    elif fraction >= 1.0:
        multi_count = len(multi)
    else:
        target_total = near_count / (1.0 - fraction)
        multi_count = min(len(multi), int(round(target_total * fraction)))

    rng.shuffle(multi)
    multi = multi[:multi_count]

    merged = near + multi
    rng.shuffle(merged)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        handle.writelines(merged)

    summary = {
        "near_solve_rows": near_count,
        "multi_state_rows": multi_count,
        "total_rows": len(merged),
        "near_solve_share": round(near_count / max(1, len(merged)), 4),
        "multi_state_share": round(multi_count / max(1, len(merged)), 4),
        "output": str(out),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
