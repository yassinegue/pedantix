#!/usr/bin/env python3
"""
Filter clean_pages.jsonl to a culturally interesting subset using a complete
fr.wikipedia pageview ranking (from build_pageview_ranking.py).

Unlike the old version that used the Wikimedia REST API (capped at top-1000/month
→ only ~9K unique articles), this version consumes the full ranking TSV produced by
build_pageview_ranking.py, which covers every fr.wikipedia article sorted by
cumulative annual views — no cap.

Strategy:
  - Keep articles with rank ≤ rank_max (default 120,000).
    Photosynthèse, ADN, Napoléon Bonaparte all land comfortably in this range;
    obscure village stubs fall below.
  - Apply regex exclusions for structurally bad article types (communes, lists,
    disambiguations, year-stub sports pages, elections, seasons, etc.).
  - Apply minimum intro-length floor (400 chars) to ensure enough text to play.

Usage:
    # Step 1 — build the ranking (one-time, ~50 GB streaming download for 2 years):
    python scripts/build_pageview_ranking.py --years 2023,2024 \\
        --output data/pageview_ranking.tsv

    # Step 2 — filter:
    python scripts/build_pageview_filter.py \\
        --pages data/clean_pages.jsonl \\
        --ranking data/pageview_ranking.tsv \\
        --output data/filtered_pages.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# ── Regex patterns for structurally bad titles ──────────────────────────────

_BAD_TITLE_PATTERNS = [
    # Geographic stubs
    r"^(Commune|Village|Canton|Arrondissement|Municipalité|Circonscription)\s+de\s+",
    # Lists
    r"^Liste\s+(des?|du|de\s+la|d'|de\s+)",
    # Portals and meta-pages
    r"^(Portail|Aide|Wikipédia|Discussion|Utilisateur)\s*:",
    # Disambiguation
    r"\(homonymie\)",
    # Niche sports: championship/cup + year suffix
    r"^Championnat\s+.{0,50}\d{4}$",
    r"^(Coupe|Saison|Édition)\s+.{0,50}\d{4}$",
    # Season ranges like "2003-04" or "2003-2004"
    r"\b(19|20)\d{2}[-–]\d{2,4}\b",
    # Elections
    r"^Élections?\s+",
    # Episode/season stubs
    r"(saison\s+\d+|épisode\s+\d+)",
    # Reality TV marker
    r"\bémission\b",
]

_BAD_RE = re.compile("|".join(_BAD_TITLE_PATTERNS), re.IGNORECASE)


def is_bad_title(title: str) -> bool:
    return bool(_BAD_RE.search(title))


# ── Load ranking ─────────────────────────────────────────────────────────────

def load_ranking(ranking_path: Path) -> dict[str, int]:
    """Load rank TSV → {title: rank}. TSV has header: rank<TAB>title<TAB>views."""
    rank_of: dict[str, int] = {}
    with open(ranking_path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rank = int(parts[0])
            title = parts[1]
            rank_of[title] = rank
    return rank_of


# ── Main filtering pipeline ──────────────────────────────────────────────────

def build_filtered_pages(
    pages_path: Path,
    output_path: Path,
    ranking_path: Path,
    *,
    rank_max: int = 120_000,
    min_intro_chars: int = 400,
    verbose: bool = True,
) -> int:
    if verbose:
        print(f"Loading ranking from {ranking_path} ...", flush=True)
    rank_of = load_ranking(ranking_path)
    if verbose:
        print(f"  {len(rank_of):,} articles in ranking", flush=True)

    if verbose:
        print(f"\nFiltering {pages_path} ...", flush=True)

    n_total = n_bad_title = n_too_short = n_no_rank = n_rank_skip = n_kept = 0

    with open(pages_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            n_total += 1
            p = json.loads(line)
            title = p["title"]

            if is_bad_title(title):
                n_bad_title += 1
                continue

            if len(p["intro"]) < min_intro_chars:
                n_too_short += 1
                continue

            rank = rank_of.get(title)
            if rank is None:
                n_no_rank += 1
                continue
            if rank > rank_max:
                n_rank_skip += 1
                continue

            fout.write(json.dumps({
                "title": title,
                "intro": p["intro"],
                "url": p.get("url", ""),
                "pageview_rank": rank,
            }, ensure_ascii=False) + "\n")
            n_kept += 1

    if verbose:
        print(f"\nResults:")
        print(f"  Total input:       {n_total:,}")
        print(f"  Bad title pattern: {n_bad_title:,}")
        print(f"  Intro too short:   {n_too_short:,}")
        print(f"  Not in ranking:    {n_no_rank:,}")
        print(f"  Rank > {rank_max:,}: {n_rank_skip:,}")
        print(f"  Kept:              {n_kept:,}")

    return n_kept


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", default="data/clean_pages.jsonl")
    parser.add_argument("--ranking", default="data/pageview_ranking.tsv",
                        help="TSV from build_pageview_ranking.py (rank/title/views)")
    parser.add_argument("--output", default="data/filtered_pages.jsonl")
    parser.add_argument("--rank-max", type=int, default=120_000,
                        help="Keep articles with rank ≤ this (default: 120000)")
    parser.add_argument("--min-intro-chars", type=int, default=400)
    args = parser.parse_args()

    n = build_filtered_pages(
        pages_path=Path(args.pages),
        output_path=Path(args.output),
        ranking_path=Path(args.ranking),
        rank_max=args.rank_max,
        min_intro_chars=args.min_intro_chars,
    )
    print(f"\nDone: {n:,} pages written to {args.output}")


if __name__ == "__main__":
    main()
