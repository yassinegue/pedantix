#!/usr/bin/env python3
"""
Filter clean_pages.jsonl using QRank (Wikidata-based popularity ranking).

Pipeline (top-down — fast):
  1. Download QRank CSV (~105 MB) if not cached.
  2. Read the first rank_max lines (already sorted desc) → top QIDs.
  3. Batch-query Wikidata API (50 QIDs/request, parallel) → {QID: fr_title}.
  4. Look up each fr_title in clean_pages.jsonl index.
  5. Apply title-regex + intro-length filters, write filtered_pages.jsonl.

Only rank_max Wikidata API batches needed (e.g. 120K → 2,400 batches ≈ 5 min).

Usage:
    python scripts/build_qrank_filter.py \
        --pages data/clean_pages.jsonl \
        --output data/filtered_pages.jsonl \
        --rank-max 120000
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

_USER_AGENT = "pedantix-qrank-filter/1.0 (research; contact: yguenn@stanford.edu)"
_QRANK_URL = "https://qrank.toolforge.org/download/qrank.csv.gz"
_WIKIDATA_API = "https://www.wikidata.org/w/api.php"

# ── Bad title patterns ────────────────────────────────────────────────────────

_BAD_TITLE_PATTERNS = [
    r"^(Commune|Village|Canton|Arrondissement|Municipalité|Circonscription)\s+de\s+",
    r"^Liste\s+(des?|du|de\s+la|d'|de\s+)",
    r"^(Portail|Aide|Wikipédia|Discussion|Utilisateur)\s*:",
    r"\(homonymie\)",
    r"^Championnat\s+.{0,50}\d{4}$",
    r"^(Coupe|Saison|Édition)\s+.{0,50}\d{4}$",
    r"\b(19|20)\d{2}[-–]\d{2,4}\b",
    r"^Élections?\s+",
    r"(saison\s+\d+|épisode\s+\d+)",
    r"\bémission\b",
]
_BAD_RE = re.compile("|".join(_BAD_TITLE_PATTERNS), re.IGNORECASE)

def is_bad_title(title: str) -> bool:
    return bool(_BAD_RE.search(title))


# ── Step 1: Download QRank, read top N QIDs ───────────────────────────────────

def load_top_qids(cache_path: Path, rank_max: int) -> list[tuple[int, str]]:
    """Download QRank CSV if needed. Return [(rank, QID)] for ranks 1..rank_max."""
    if not cache_path.exists():
        print(f"Downloading QRank from {_QRANK_URL} ...", flush=True)
        req = urllib.request.Request(_QRANK_URL, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as resp, \
             open(cache_path, "wb") as f:
            downloaded = 0
            with tqdm(unit="B", unit_scale=True, unit_divisor=1024,
                      desc="qrank.csv.gz") as pbar:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
                    pbar.update(len(chunk))
                    downloaded += len(chunk)
        print(f"  Saved {downloaded/1e6:.0f} MB to {cache_path}", flush=True)

    print(f"Reading top {rank_max:,} QIDs from QRank ...", flush=True)
    top: list[tuple[int, str]] = []
    with gzip.open(cache_path, "rt", encoding="utf-8") as f:
        next(f)  # skip header — file is already sorted desc by score
        for rank, line in enumerate(tqdm(f, total=rank_max, desc="QRank", unit=" QIDs"), 1):
            if rank > rank_max:
                break
            qid = line.split(",", 1)[0]
            top.append((rank, qid))
    print(f"  {len(top):,} QIDs loaded (ranks 1–{rank_max:,})", flush=True)
    return top


# ── Step 2: Index clean_pages.jsonl by title ──────────────────────────────────

def index_pages(pages_path: Path, min_intro: int) -> dict[str, dict]:
    """Return {fr_title: page_dict} for pages passing intro-length filter."""
    index: dict[str, dict] = {}
    with open(pages_path, encoding="utf-8") as f:
        for line in tqdm(f, desc="Indexing pages", unit=" pages"):
            p = json.loads(line)
            if len(p.get("intro", "")) >= min_intro:
                index[p["title"]] = p
    print(f"  {len(index):,} pages indexed", flush=True)
    return index


# ── Step 3: Batch Wikidata API (QID → fr_title) ───────────────────────────────

def _query_wikidata_batch(qids: list[str], retries: int = 4) -> dict[str, str]:
    """Query Wikidata for fr.wikipedia titles of a batch of QIDs.
    Returns {QID: fr_title}.
    """
    params = urllib.parse.urlencode({
        "action": "wbgetentities",
        "ids": "|".join(qids),
        "props": "sitelinks",
        "sitefilter": "frwiki",
        "format": "json",
    })
    url = f"{_WIKIDATA_API}?{params}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            result: dict[str, str] = {}
            for qid, entity in data.get("entities", {}).items():
                if "missing" in entity:
                    continue
                fr_title = (entity.get("sitelinks") or {}).get("frwiki", {}).get("title")
                if fr_title:
                    result[qid] = fr_title
            return result
        except urllib.error.HTTPError as e:
            wait = int(e.headers.get("Retry-After", 10)) if e.code == 429 else min(2 ** attempt, 30)
            time.sleep(wait)
        except Exception:
            time.sleep(min(2 ** attempt, 30))
    return {}


def get_titles_for_qids(
    qids: list[str], workers: int = 20, batch_size: int = 50
) -> dict[str, str]:
    """Return {QID: fr_title} for all QIDs that have a fr.wikipedia article."""
    batches = [qids[i:i + batch_size] for i in range(0, len(qids), batch_size)]
    qid_to_title: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_query_wikidata_batch, batch): batch for batch in batches}
        with tqdm(total=len(batches), desc="Wikidata API", unit="batch") as pbar:
            for fut in as_completed(futures):
                qid_to_title.update(fut.result())
                pbar.set_postfix({"found": len(qid_to_title)})
                pbar.update(1)

    return qid_to_title


# ── Main ──────────────────────────────────────────────────────────────────────

def build_filtered_pages(
    pages_path: Path,
    output_path: Path,
    qrank_cache: Path,
    *,
    rank_max: int = 120_000,
    min_intro_chars: int = 400,
    wikidata_workers: int = 20,
    verbose: bool = True,
) -> int:
    # 1. Top N QIDs from QRank
    top_qids = load_top_qids(qrank_cache, rank_max)
    rank_map = {qid: rank for rank, qid in top_qids}  # {QID: rank}

    # 2. Index clean_pages by title
    if verbose:
        print(f"\nIndexing {pages_path} ...", flush=True)
    pages_index = index_pages(pages_path, min_intro_chars)

    # 3. Wikidata: QID → fr_title
    qids = [qid for _, qid in top_qids]
    if verbose:
        print(f"\nQuerying Wikidata for {len(qids):,} QIDs ({wikidata_workers} workers) ...", flush=True)
    qid_to_title = get_titles_for_qids(qids, workers=wikidata_workers)
    if verbose:
        print(f"  {len(qid_to_title):,} QIDs have a fr.wikipedia article", flush=True)

    # 4. Intersect with pages index, apply title filter
    kept: list[tuple[int, str, str]] = []  # (rank, title, qid)
    for qid, fr_title in qid_to_title.items():
        page = pages_index.get(fr_title)
        if page is None:
            continue
        if is_bad_title(fr_title):
            continue
        kept.append((rank_map[qid], fr_title, qid))

    kept.sort()
    if verbose:
        print(f"  {len(kept):,} articles kept after all filters", flush=True)

    # 5. Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        for rank, title, qid in kept:
            page = pages_index[title]
            fout.write(json.dumps({
                "title": title,
                "intro": page["intro"],
                "url": page.get("url", ""),
                "pageview_rank": rank,
                "qid": qid,
            }, ensure_ascii=False) + "\n")

    if verbose:
        print("\nRank boundaries (spot-check):")
        for r in [1, 100, 1_000, 5_000, 10_000, 20_000, 50_000, 100_000]:
            matches = [(rk, t) for rk, t, _ in kept if rk <= r]
            if matches:
                rk, t = max(matches, key=lambda x: x[0])
                print(f"  rank ≤ {r:6d}: last entry is {t!r} (rank {rk})")

    return len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", default="data/clean_pages.jsonl")
    parser.add_argument("--output", default="data/filtered_pages.jsonl")
    parser.add_argument("--qrank-cache", default="data/qrank.csv.gz")
    parser.add_argument("--rank-max", type=int, default=120_000)
    parser.add_argument("--min-intro-chars", type=int, default=400)
    parser.add_argument("--wikidata-workers", type=int, default=20)
    args = parser.parse_args()

    n = build_filtered_pages(
        pages_path=Path(args.pages),
        output_path=Path(args.output),
        qrank_cache=Path(args.qrank_cache),
        rank_max=args.rank_max,
        min_intro_chars=args.min_intro_chars,
        wikidata_workers=args.wikidata_workers,
    )
    print(f"\nDone: {n:,} pages written to {args.output}")


if __name__ == "__main__":
    main()
