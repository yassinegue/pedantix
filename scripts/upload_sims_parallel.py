#!/usr/bin/env python3
"""Upload all sims/*.json files to R2 in parallel using a thread pool."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def upload_one(args: tuple[str, Path, str]) -> tuple[int, bool, str]:
    page_id, path, bucket = args
    local_wrangler = REPO_ROOT / "web" / "node_modules" / ".bin" / "wrangler"
    wrangler = str(local_wrangler) if local_wrangler.exists() else (shutil.which("wrangler") or "wrangler")
    key = f"sims/{path.name}"
    cmd = [wrangler, "r2", "object", "put", f"{bucket}/{key}",
           f"--file={path.resolve()}", "--remote"]
    for attempt in range(1, 4):
        res = subprocess.run(cmd, cwd=REPO_ROOT / "web",
                             capture_output=True, check=False)
        if res.returncode == 0:
            return page_id, True, ""
        err = res.stderr.decode(errors="replace").strip()
    return page_id, False, err


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sims-dir", default=REPO_ROOT / "web" / "sims", type=Path)
    p.add_argument("--bucket", default="pedantix-ranks")
    p.add_argument("--workers", default=12, type=int)
    p.add_argument("--ids", default="", type=str,
                   help="Comma-separated IDs to upload (default: all)")
    args = p.parse_args()

    all_files = sorted(args.sims_dir.glob("*.json"), key=lambda f: int(f.stem))
    if args.ids:
        wanted = {str(i) for i in args.ids.split(",")}
        all_files = [f for f in all_files if f.stem in wanted]

    total = len(all_files)
    print(f"Uploading {total} files with {args.workers} workers…")

    done = 0
    failed = []
    work = [(int(f.stem), f, args.bucket) for f in all_files]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(upload_one, item): item[0] for item in work}
        for fut in as_completed(futures):
            page_id, ok, err = fut.result()
            done += 1
            if ok:
                if done % 50 == 0 or done == total:
                    print(f"  {done}/{total} uploaded…")
            else:
                failed.append((page_id, err))
                print(f"  FAILED page {page_id}: {err[:120]}")

    if failed:
        print(f"\n{len(failed)} failures:")
        for pid, err in failed:
            print(f"  page {pid}: {err}")
        return 1
    print(f"\nDone: {total} files uploaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
