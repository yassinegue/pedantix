#!/usr/bin/env python3
"""Precompute per-page word ranks for the Cloudflare Pedantix Worker.

For each target page, ranks the entire fasttext vocabulary by max cosine
similarity to the page's normalized word set, keeps the top N (default 1000),
and writes a single JSON file `{normalized_word: rank}` to disk. Optionally
uploads each file to R2 via `wrangler r2 object put`.

Usage:
    python scripts/precompute_puzzle_ranks.py \
        --jsonl data/filtered_pages.jsonl \
        --ids 1,2,3 \
        --out web/ranks/ \
        --top-n 1000 \
        [--upload]

The Worker (web/src/scorer.ts) reads `ranks/{page_id}.json` from the R2
bucket bound as RANKS. Keys are normalized lowercase ASCII (no accents,
no apostrophes), matching the TS normalize() in web/src/text.ts.

CRITICAL: uses FastTextWikiModel (models/fasttext_wiki_model.npz) — never
TinyPedantixModel. See CLAUDE.md.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np

# Make pedantix_project importable when running from repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pedantix_project.model import FastTextWikiModel  # noqa: E402


# ---------------------------------------------------------------------------
# Normalization — must match web/src/text.ts `normalize(s).replace(/[^a-z0-9]/g, "")`
# ---------------------------------------------------------------------------

def normalize_for_web(s: str) -> str:
    """Lowercase, NFD-strip combining marks, oe/ae expansion, keep [a-z0-9]."""
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("œ", "oe").replace("æ", "ae")
    return "".join(c for c in s if c.isascii() and c.isalnum())


def page_word_set(intro: str, title: str) -> list[str]:
    """Deduplicated normalized words appearing in the page intro + title.

    These are the "page canons" we rank the vocabulary against (max-sim).
    """
    seen: set[str] = set()
    for chunk in (intro, title):
        # Split on anything non-letter/digit (in unicode, not just ASCII).
        buf: list[str] = []
        for ch in chunk:
            if ch.isalnum():
                buf.append(ch)
            else:
                if buf:
                    word = normalize_for_web("".join(buf))
                    if word and len(word) > 1:
                        seen.add(word)
                    buf.clear()
        if buf:
            word = normalize_for_web("".join(buf))
            if word and len(word) > 1:
                seen.add(word)
    return sorted(seen)


# ---------------------------------------------------------------------------
# Vocabulary preprocessing
# ---------------------------------------------------------------------------

def build_normalized_vocab(model: FastTextWikiModel) -> tuple[list[str], np.ndarray]:
    """Collapse the raw fasttext vocab to its normalized form.

    Multiple raw entries (e.g. "Napoléon", "napoléon") collapse to "napoleon".
    For each normalized form, we keep the average of the contributing vectors
    (then renormalize), which is a reasonable single representative.

    Returns (norm_vocab_list, norm_matrix) where norm_matrix is float32 (M, 300)
    with unit-norm rows, and M ≤ N.
    """
    n, d = model.matrix.shape
    buckets: dict[str, list[int]] = {}
    raw_vocab = [""] * n
    for w, i in model.word2idx.items():
        raw_vocab[i] = w
    for i, w in enumerate(raw_vocab):
        norm = normalize_for_web(w)
        if not norm or len(norm) < 2:
            continue
        buckets.setdefault(norm, []).append(i)

    norm_vocab = sorted(buckets.keys())
    M = len(norm_vocab)
    norm_matrix = np.zeros((M, d), dtype=np.float32)
    for j, w in enumerate(norm_vocab):
        idxs = buckets[w]
        v = model.matrix[idxs].mean(axis=0)
        nrm = float(np.linalg.norm(v))
        if nrm > 1e-9:
            v = v / nrm
        norm_matrix[j] = v
    return norm_vocab, norm_matrix


# ---------------------------------------------------------------------------
# Per-page ranking
# ---------------------------------------------------------------------------

def rank_page(
    page_canons: list[str],
    norm_vocab: list[str],
    norm_matrix: np.ndarray,
    word_to_idx: dict[str, int],
    top_n: int,
) -> dict[str, int]:
    """Return {normalized_word: rank} for the top N most-similar vocabulary words.

    Similarity = max cosine sim to any word in page_canons (matches the
    Pedantix scoring semantics). Page words themselves get rank 1..K naturally.
    """
    valid_canon_idxs = [word_to_idx[w] for w in page_canons if w in word_to_idx]
    if not valid_canon_idxs:
        return {}
    page_vecs = norm_matrix[valid_canon_idxs]              # (K, 300)
    # (M, 300) @ (300, K) -> (M, K), then max over K.
    # Chunk over M to keep peak memory bounded (each row is K*4 bytes).
    M = norm_matrix.shape[0]
    K = page_vecs.shape[0]
    chunk = max(1, min(M, 200_000 // max(1, K)))
    word_max = np.empty(M, dtype=np.float32)
    page_vecs_t = page_vecs.T.astype(np.float32, copy=False)
    for start in range(0, M, chunk):
        end = min(M, start + chunk)
        sims = norm_matrix[start:end] @ page_vecs_t       # (chunk, K)
        word_max[start:end] = sims.max(axis=1)

    # argsort descending; take top_n.
    n_take = min(top_n, M)
    # argpartition is O(M); a full sort over only the top-n window keeps order.
    cut = np.argpartition(-word_max, n_take - 1)[:n_take]
    cut_sorted = cut[np.argsort(-word_max[cut])]

    ranks: dict[str, int] = {}
    for rank0, idx in enumerate(cut_sorted):
        ranks[norm_vocab[idx]] = rank0 + 1
    return ranks


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def read_pages(jsonl_path: Path, wanted_ids: set[int]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    seq = 0  # mirrors ingest.mjs: id = row.id ?? row.page_id ?? total + 1
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            title = row.get("title")
            intro = row.get("intro") or row.get("summary") or row.get("text")
            if not title or not intro:
                continue
            seq += 1
            pid = int(row.get("id") or row.get("page_id") or seq)
            if pid in wanted_ids:
                out[pid] = row
                if len(out) == len(wanted_ids):
                    break
    return out


def upload_to_r2(bucket: str, key: str, file_path: Path, remote: bool) -> None:
    """Shell out to wrangler. Requires the user to have run `wrangler login`."""
    # Prefer local node_modules/.bin/wrangler so no global install is needed.
    local_wrangler = REPO_ROOT / "web" / "node_modules" / ".bin" / "wrangler"
    wrangler_bin = str(local_wrangler) if local_wrangler.exists() else (shutil.which("wrangler") or "wrangler")
    abs_file = str(file_path.resolve())
    cmd = [wrangler_bin, "r2", "object", "put", f"{bucket}/{key}", f"--file={abs_file}"]
    if remote:
        cmd.append("--remote")
    else:
        cmd.append("--local")
    print(f"  uploading: {' '.join(cmd)}")
    for attempt in range(1, 4):
        res = subprocess.run(cmd, cwd=REPO_ROOT / "web", check=False)
        if res.returncode == 0:
            return
        print(f"  upload failed (attempt {attempt}/3), retrying…")
    raise RuntimeError(f"wrangler upload failed for {key}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True, type=Path,
                   help="Path to pages JSONL with {id, title, intro|summary|text}")
    p.add_argument("--ids", required=True, type=str,
                   help="Comma-separated page ids to precompute")
    p.add_argument("--out", default=REPO_ROOT / "web" / "ranks", type=Path,
                   help="Directory to write rank JSONs into")
    p.add_argument("--top-n", default=1000, type=int)
    p.add_argument("--model", default=REPO_ROOT / "models" / "fasttext_wiki_model.npz", type=Path)
    p.add_argument("--upload", action="store_true",
                   help="Upload each generated file to R2 via wrangler")
    p.add_argument("--bucket", default="pedantix-ranks",
                   help="R2 bucket name (must match wrangler.toml)")
    p.add_argument("--local", action="store_true",
                   help="Upload to the local wrangler dev R2 (default: --remote)")
    args = p.parse_args()

    wanted_ids = {int(x) for x in args.ids.split(",") if x.strip()}
    if not wanted_ids:
        print("no ids provided", file=sys.stderr)
        return 2

    print(f"reading pages from {args.jsonl}…")
    pages = read_pages(args.jsonl, wanted_ids)
    missing = wanted_ids - set(pages.keys())
    if missing:
        print(f"warning: {len(missing)} ids not found in jsonl: {sorted(missing)[:10]}…",
              file=sys.stderr)
    if not pages:
        print("no matching pages", file=sys.stderr)
        return 2

    print(f"loading fasttext model from {args.model}…")
    model = FastTextWikiModel.load(args.model)
    print(f"  raw vocab: {len(model.word2idx):,}  matrix: {model.matrix.shape}")

    print("collapsing vocab to normalized forms…")
    norm_vocab, norm_matrix = build_normalized_vocab(model)
    print(f"  normalized vocab: {len(norm_vocab):,}")
    word_to_idx = {w: i for i, w in enumerate(norm_vocab)}

    args.out.mkdir(parents=True, exist_ok=True)

    for pid in sorted(pages.keys()):
        row = pages[pid]
        intro = row.get("intro") or row.get("summary") or row.get("text") or ""
        title = row.get("title") or ""
        canons = page_word_set(intro, title)
        if not canons:
            print(f"  page {pid}: empty canons, skipping")
            continue
        print(f"  page {pid} ({title[:40]}): {len(canons)} canons → ranking…")
        ranks = rank_page(canons, norm_vocab, norm_matrix, word_to_idx, args.top_n)
        out_path = args.out / f"{pid}.json"
        out_path.write_text(json.dumps(ranks, separators=(",", ":")), encoding="utf-8")
        print(f"    wrote {out_path} ({len(ranks)} ranks, {out_path.stat().st_size // 1024} KB)")
        if args.upload:
            local_bin = REPO_ROOT / "web" / "node_modules" / ".bin" / "wrangler"
            if not local_bin.exists() and not shutil.which("wrangler"):
                raise RuntimeError("--upload requires `wrangler` on PATH or in web/node_modules/.bin/")
            upload_to_r2(args.bucket, f"ranks/{pid}.json", out_path, remote=not args.local)

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
