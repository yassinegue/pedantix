#!/usr/bin/env python3
"""
Re-fetch full Wikipedia introductory sections (all paragraphs before the first
section heading) for every page currently in D1, using the MediaWiki `extracts`
API with exintro=true&explaintext=true.

Updates both:
  - data/bulk_pages.jsonl  (patched in-place so downstream scripts stay consistent)
  - D1 (via wrangler d1 execute --remote)

Usage:
    python3 scripts/refetch_full_intros.py [--limit 2000] [--local]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"
JSONL_PATH = REPO_ROOT / "data" / "filtered_pages.jsonl"
DB_NAME = "pedantix"
WRANGLER = WEB_DIR / "node_modules" / ".bin" / "wrangler"
BATCH_SIZE = 20   # titles per API request
SLEEP = 0.5       # seconds between API calls

# Cleanup patterns applied to the extracted text
_BLANK_LINES = re.compile(r"\n{3,}")
_NOTES_RE = re.compile(r"\s*\(.*?\)\s*", re.DOTALL)  # aggressive — not used
_REF_RE = re.compile(r"\[\d+\]")                      # [1], [2], etc.
_IPA_RE = re.compile(r"\s*[(/\[][^)\]/]*[)\]/]")      # IPA and parenthetical asides near title


def q(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def fetch_extracts(titles: list[str], lang: str = "fr") -> dict[str, str]:
    """Return {title: extract_plain_text} for each found title."""
    joined = "|".join(t.replace(" ", "_") for t in titles)
    params = urllib.parse.urlencode({
        "action": "query",
        "prop": "extracts",
        "exintro": "true",
        "explaintext": "true",
        "redirects": "true",
        "format": "json",
        "titles": joined,
    })
    url = f"https://{lang}.wikipedia.org/w/api.php?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "palimot-refetch/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    result: dict[str, str] = {}
    pages = data.get("query", {}).get("pages", {})
    redirects = {r["from"]: r["to"] for r in data.get("query", {}).get("redirects", [])}

    # Build title → normalized mapping from redirects
    rev_redirect: dict[str, str] = {}
    for frm, to in redirects.items():
        rev_redirect[to] = frm   # wiki canonical → original query title

    for page in pages.values():
        extract: str = page.get("extract", "").strip()
        if not extract or page.get("missing") is not None:
            continue
        # Clean up
        extract = _REF_RE.sub("", extract)
        extract = _BLANK_LINES.sub("\n\n", extract).strip()
        if len(extract) < 100:
            continue
        wiki_title: str = page.get("title", "")
        # Map back to original query title via redirect table
        original = rev_redirect.get(wiki_title, wiki_title)
        result[original] = extract
        result[wiki_title] = extract  # also key by canonical
    return result


def run_d1(sql: str, target_flag: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sql", delete=False, encoding="utf-8") as f:
        f.write(sql)
        tmp = Path(f.name)
    cmd = [str(WRANGLER), "d1", "execute", DB_NAME, target_flag, "--yes", f"--file={tmp}"]
    res = subprocess.run(cmd, cwd=WEB_DIR, check=False)
    tmp.unlink()
    if res.returncode != 0:
        raise RuntimeError("wrangler d1 execute failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2000, help="Max pages to process")
    parser.add_argument("--local", action="store_true", help="Target local D1 (dev)")
    parser.add_argument("--skip-d1", action="store_true", help="Only update JSONL, skip D1")
    args = parser.parse_args()

    target_flag = "--local" if args.local else "--remote"

    # Load current JSONL
    print(f"Reading {JSONL_PATH} …", flush=True)
    rows: list[dict] = []
    with JSONL_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    rows = rows[: args.limit]
    print(f"Processing {len(rows)} pages", flush=True)

    # We process in batches of BATCH_SIZE
    updated_rows = list(rows)  # will be mutated
    total_updated = 0

    for batch_start in range(0, len(rows), BATCH_SIZE):
        batch = rows[batch_start: batch_start + BATCH_SIZE]
        titles = [r["title"] for r in batch]
        batch_ids = list(range(batch_start + 1, batch_start + len(batch) + 1))

        try:
            extracts = fetch_extracts(titles)
        except Exception as e:
            print(f"  WARN batch {batch_start}: {e}", flush=True)
            time.sleep(3)
            continue

        sql_stmts: list[str] = []
        for i, row in enumerate(batch):
            pid = batch_ids[i]
            title = row["title"]
            new_intro = extracts.get(title) or extracts.get(title.replace("_", " "))
            if not new_intro:
                continue
            # Update in-memory row
            updated_rows[batch_start + i] = dict(row, intro=new_intro)
            total_updated += 1
            if not args.skip_d1:
                wc = len(new_intro.split())
                sql_stmts.append(
                    f"UPDATE pages SET intro={q(new_intro)}, char_count={len(new_intro)}, "
                    f"word_count={wc} WHERE id={pid};"
                )

        if sql_stmts and not args.skip_d1:
            sql = "\n".join(sql_stmts)
            try:
                run_d1(sql, target_flag)
            except RuntimeError as e:
                print(f"  WARN D1 update failed for batch {batch_start}: {e}", flush=True)

        done = batch_start + len(batch)
        print(f"  {done}/{len(rows)} done, {total_updated} updated so far", flush=True)
        time.sleep(SLEEP)

    # Write updated JSONL back
    out_path = JSONL_PATH.with_name("filtered_pages_full.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for row in updated_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(updated_rows)} rows → {out_path}", flush=True)
    print(f"Updated {total_updated} intros.", flush=True)


if __name__ == "__main__":
    main()
