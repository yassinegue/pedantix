#!/usr/bin/env python3
"""
Build a complete fr.wikipedia pageview ranking from Wikimedia's monthly dump files.

Two-phase pipeline:
  Phase 1 — Download: 2 parallel workers pull bz2 files to data/pageview_tmp/.
             Wikimedia rate-limits to ~2 concurrent connections; 429s are retried.
  Phase 2 — Decompress: 12 parallel workers read the local bz2 files, filter to
             fr.wikipedia rows, accumulate view counts, then delete the bz2 files.

Expected wall-clock time for 2023+2024 (24 files × ~3 GB each):
  Download  : ~10–20 min  (network bound, 2 workers)
  Decompress: ~20–30 min  (CPU bound,     12 workers)
  Total     : ~30–50 min

Output: data/pageview_ranking.tsv  (rank<TAB>title<TAB>views)

Usage:
  python scripts/build_pageview_ranking.py --years 2023,2024
  python scripts/build_pageview_ranking.py --dry-run   # inspect format
"""
from __future__ import annotations

import argparse
import bz2
import pickle
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

_BASE_URL = "https://dumps.wikimedia.org/other/pageview_complete/monthly"
_USER_AGENT = "pedantix-pageview-ranking/1.0 (research; contact: yguenn@stanford.edu)"
_PREFIX = b"fr.wikipedia "


def _month_url(year: int, month: int) -> str:
    return f"{_BASE_URL}/{year}/{year}-{month:02d}/pageviews-{year}{month:02d}-user.bz2"


# ── Phase 1: Download ─────────────────────────────────────────────────────────

def _download_month(
    year: int, month: int, tmp_dir: Path, max_retries: int = 8
) -> tuple[str, Path]:
    """Download one monthly bz2 to tmp_dir. Returns (label, local_path)."""
    label = f"{year}-{month:02d}"
    url = _month_url(year, month)
    out_path = tmp_dir / f"pageviews-{year}{month:02d}-user.bz2"

    if out_path.exists() and out_path.stat().st_size > 1_000_000:
        size_gb = out_path.stat().st_size / 1e9
        tqdm.write(f"  [{label}] already on disk ({size_gb:.1f} GB), skipping download", file=sys.stderr)
        return label, out_path

    for attempt in range(max_retries):
        try:
            t0 = time.time()
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=300) as resp, \
                 open(out_path, "wb") as f:
                while chunk := resp.read(1 << 20):  # 1 MB chunks
                    f.write(chunk)
            elapsed = time.time() - t0
            size_gb = out_path.stat().st_size / 1e9
            speed = size_gb / elapsed * 1000
            tqdm.write(
                f"  [{label}] downloaded {size_gb:.1f} GB in {elapsed:.0f}s ({speed:.0f} MB/s)",
                file=sys.stderr,
            )
            return label, out_path
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = int(e.headers.get("Retry-After", 60))
                tqdm.write(f"  [{label}] 429 — waiting {wait}s ...", file=sys.stderr)
                time.sleep(wait)
            else:
                wait = min(2 ** attempt, 60)
                tqdm.write(f"  [{label}] HTTP {e.code} — retry in {wait}s", file=sys.stderr)
                time.sleep(wait)
        except Exception as e:
            out_path.unlink(missing_ok=True)
            wait = min(2 ** attempt, 60)
            tqdm.write(f"  [{label}] error ({e}) — retry in {wait}s", file=sys.stderr)
            time.sleep(wait)

    raise RuntimeError(f"[{label}] download failed after {max_retries} attempts")


# ── Phase 2: Decompress ───────────────────────────────────────────────────────

def parse_line(raw: bytes) -> tuple[str, int] | None:
    try:
        parts = raw.decode("utf-8").rstrip("\n").split(" ")
        title = parts[1].replace("_", " ")
        views = int(parts[3])  # parts[2] is page_id, parts[3] is monthly views
        return title, views
    except Exception:
        return None


def _decompress_month(
    label: str, bz2_path: Path, cache_path: Path, *, keep_files: bool = False
) -> tuple[str, dict[str, int]]:
    """Decompress + filter one bz2 file. Saves result to cache_path (pickle).
    On next run, loads from cache and skips decompression entirely.
    Deletes bz2 after unless keep_files.
    """
    # Resume: load from cache if already done
    if cache_path.exists():
        t0 = time.time()
        with open(cache_path, "rb") as f:
            month_views = pickle.load(f)
        tqdm.write(
            f"  [{label}] loaded from cache ({len(month_views):,} articles, {time.time()-t0:.1f}s)",
            file=sys.stderr,
        )
        return label, month_views

    month_views: dict[str, int] = {}
    t0 = time.time()
    with open(bz2_path, "rb") as raw_f:
        for line in bz2.BZ2File(raw_f):
            if not line.startswith(_PREFIX):
                continue
            parsed = parse_line(line)
            if parsed is None:
                continue
            title, v = parsed
            if v <= 0:
                continue
            if v > month_views.get(title, 0):
                month_views[title] = v

    # Save checkpoint before deleting bz2
    with open(cache_path, "wb") as f:
        pickle.dump(month_views, f, protocol=pickle.HIGHEST_PROTOCOL)

    if not keep_files:
        bz2_path.unlink(missing_ok=True)
    elapsed = time.time() - t0
    tqdm.write(
        f"  [{label}] {len(month_views):,} articles in {elapsed:.0f}s",
        file=sys.stderr,
    )
    return label, month_views


# ── Dry-run ───────────────────────────────────────────────────────────────────

def _dry_run(years: list[int]) -> None:
    """Stream-fetch the first file and print 40 fr.wikipedia lines."""
    for year in years:
        for month in range(1, 13):
            url = _month_url(year, month)
            print(f"Dry-run — first 40 fr.wikipedia lines from:\n  {url}\n")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    found = 0
                    for raw in bz2.BZ2File(resp):
                        if raw.startswith(_PREFIX):
                            parsed = parse_line(raw)
                            if parsed:
                                title, views = parsed
                                print(f"  {views:>10,}  {title}")
                            else:
                                print(f"  [parse error] {raw[:100]}")
                            found += 1
                            if found >= 40:
                                break
                return
            except Exception as e:
                print(f"  Warning: {e}, trying next month...")
    print("Could not fetch any file.")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_ranking(
    years: list[int],
    output_path: Path,
    *,
    download_workers: int = 2,
    decompress_workers: int = 12,
    keep_files: bool = False,
    verbose: bool = True,
) -> int:
    tmp_dir = output_path.parent / "pageview_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    all_months = [(y, m) for y in years for m in range(1, 13)]

    # ── Phase 1: Download ─────────────────────────────────────────────────────
    if verbose:
        print(f"\nPhase 1 — downloading {len(all_months)} files "
              f"({download_workers} workers) ...", flush=True)

    downloaded: dict[tuple[int, int], Path] = {}
    with ThreadPoolExecutor(max_workers=download_workers) as pool:
        futures: dict = {}
        for i, (y, m) in enumerate(all_months):
            if i > 0:
                time.sleep(3)  # stagger: never open 2 connections in the same second
            futures[pool.submit(_download_month, y, m, tmp_dir)] = (y, m)
        with tqdm(total=len(all_months), desc="download", unit="file") as pbar:
            for fut in as_completed(futures):
                y, m = futures[fut]
                try:
                    label, path = fut.result()
                    downloaded[(y, m)] = path
                except Exception as e:
                    tqdm.write(f"  DOWNLOAD FAILED {y}-{m:02d}: {e}", file=sys.stderr)
                pbar.set_postfix({"done": len(downloaded)})
                pbar.update(1)

    if verbose:
        print(f"  {len(downloaded)}/{len(all_months)} files downloaded to {tmp_dir}",
              flush=True)

    # ── Phase 2: Decompress ───────────────────────────────────────────────────
    if verbose:
        print(f"\nPhase 2 — decompressing {len(downloaded)} files "
              f"({decompress_workers} workers) ...", flush=True)

    views: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=decompress_workers) as pool:
        futures = {
            pool.submit(
                _decompress_month,
                f"{y}-{m:02d}",
                path,
                tmp_dir / f"pageviews-{y}{m:02d}.cache.pkl",
                keep_files=keep_files,
            ): (y, m)
            for (y, m), path in downloaded.items()
        }
        with tqdm(total=len(futures), desc="decompress", unit="file") as pbar:
            for fut in as_completed(futures):
                y, m = futures[fut]
                try:
                    label, month_views = fut.result()
                    for title, v in month_views.items():
                        views[title] = views.get(title, 0) + v
                except Exception as e:
                    tqdm.write(f"  DECOMPRESS FAILED {y}-{m:02d}: {e}", file=sys.stderr)
                pbar.set_postfix({"articles": f"{len(views):,}"})
                pbar.update(1)

    # Clean up empty tmp dir
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    # ── Sort and write ────────────────────────────────────────────────────────
    if verbose:
        print(f"\nSorting {len(views):,} articles ...", flush=True)
    ranked = sorted(views.items(), key=lambda x: -x[1])

    if verbose:
        print("Rank boundaries (spot-check):")
        for r in [1, 100, 1_000, 5_000, 10_000, 20_000, 50_000, 100_000, 150_000]:
            if r <= len(ranked):
                title, v = ranked[r - 1]
                print(f"  rank {r:7d}: {title!r:55s} ({v:,} views)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("rank\ttitle\tviews\n")
        for rank, (title, v) in enumerate(ranked, 1):
            f.write(f"{rank}\t{title}\t{v}\n")

    if verbose:
        print(f"\nSaved {len(ranked):,} entries to {output_path}", flush=True)

    return len(ranked)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", default="2023,2024",
                        help="Comma-separated years (default: 2023,2024)")
    parser.add_argument("--output", default="data/pageview_ranking.tsv")
    parser.add_argument("--download-workers", type=int, default=2,
                        help="Parallel download threads (Wikimedia limits ~2 concurrent)")
    parser.add_argument("--decompress-workers", type=int, default=6,
                        help="Parallel decompress threads (default: 6)")
    parser.add_argument("--keep-files", action="store_true",
                        help="Keep bz2 files in data/pageview_tmp/ after decompression")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print first 40 fr.wikipedia lines from first file, then exit")
    args = parser.parse_args()

    years = [int(y) for y in args.years.split(",")]

    if args.dry_run:
        _dry_run(years)
        return

    print(f"Building fr.wikipedia pageview ranking for {years} ...", flush=True)
    print(f"  Download workers : {args.download_workers}", flush=True)
    print(f"  Decompress workers: {args.decompress_workers}", flush=True)
    n = build_ranking(
        years,
        Path(args.output),
        download_workers=args.download_workers,
        decompress_workers=args.decompress_workers,
        keep_files=args.keep_files,
    )
    print(f"\nDone: {n:,} articles ranked.", flush=True)


if __name__ == "__main__":
    main()
