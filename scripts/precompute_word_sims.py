#!/usr/bin/env python3
"""Precompute per-guess article-word similarity tables.

For each article page, builds an inverted index:
  vocab_word → {article_word: cosine_similarity}

No rank-table restriction: computes against the FULL fasttext vocab so any
guess (even outside the topic's top-1000) can produce highlights.

Similarity threshold is low (default 0.35) so even distant guesses produce
subtle hints. The frontend scales opacity smoothly with the sim value.

Output: R2 key  sims/{page_id}.json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pedantix_project.model import FastTextWikiModel  # noqa: E402


def normalize_for_web(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("œ", "oe").replace("æ", "ae")
    return "".join(c for c in s if c.isascii() and c.isalnum())


_APOSTROPHES = re.compile(r"['’ʼʹ`]")
# French clitics that prefix a noun after an apostrophe (mirrors web/src/text.ts).
_CLITICS = {"l", "d", "qu", "n", "m", "t", "s", "j", "c", "lorsqu", "puisqu", "jusqu"}


def article_words(intro: str, title: str) -> list[str]:
    """Return deduplicated normalized article words, compatible with the TypeScript tokenizer.

    The TypeScript tokenizer splits French clitics on apostrophes (l'étoile → "l", "etoile").
    We do the same so sim-table article-word keys match the runtime token norms.
    """
    seen: set[str] = set()
    for chunk in (intro + " " + title).split():
        # Split on apostrophes (same logic as TypeScript FRENCH_CLITICS).
        parts = _APOSTROPHES.split(chunk)
        for i, part in enumerate(parts):
            w = normalize_for_web(part)
            if not w or len(w) <= 1:
                continue
            # Skip clitic prefix (e.g. "l" from "l'étoile") — keep only the main word.
            if i < len(parts) - 1 and w in _CLITICS:
                continue
            seen.add(w)
    return sorted(seen)


def upload_to_r2(bucket: str, key: str, file_path: Path, remote: bool) -> None:
    local_wrangler = REPO_ROOT / "web" / "node_modules" / ".bin" / "wrangler"
    wrangler_bin = str(local_wrangler) if local_wrangler.exists() else (shutil.which("wrangler") or "wrangler")
    cmd = [wrangler_bin, "r2", "object", "put", f"{bucket}/{key}",
           f"--file={str(file_path.resolve())}"]
    cmd.append("--remote" if remote else "--local")
    for attempt in range(1, 4):
        res = subprocess.run(cmd, cwd=REPO_ROOT / "web", check=False, capture_output=True)
        if res.returncode == 0:
            return
        print(f"    upload failed (attempt {attempt}/3), retrying…")
    raise RuntimeError(f"wrangler upload failed for {key}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True, type=Path)
    p.add_argument("--ids", required=True, type=str)
    p.add_argument("--out", default=REPO_ROOT / "web" / "sims", type=Path)
    p.add_argument("--model", default=REPO_ROOT / "models" / "fasttext_wiki_model.npz", type=Path)
    p.add_argument("--top-k", default=500, type=int,
                   help="Max similar vocab words stored per article word")
    p.add_argument("--min-sim", default=0.1, type=float,
                   help="Minimum cosine similarity to store")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--bucket", default="pedantix-ranks")
    p.add_argument("--local", action="store_true")
    args = p.parse_args()

    target_ids = {int(x) for x in args.ids.split(",")}
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model}…")
    model = FastTextWikiModel.load(args.model)
    n_vocab, d = model.matrix.shape
    print(f"  {n_vocab:,} words × {d} dims")

    # Build normalized vocab list + unit-vector matrix (computed once, shared across pages)
    print("  Building normalized vocab matrix…")
    buckets: dict[str, list[int]] = {}
    for w, i in model.word2idx.items():
        norm = normalize_for_web(w)
        if norm and len(norm) > 1:
            buckets.setdefault(norm, []).append(i)

    norm_vocab: list[str] = sorted(buckets.keys())
    norm_matrix = np.zeros((len(norm_vocab), d), dtype=np.float32)
    for j, w in enumerate(norm_vocab):
        idxs = buckets[w]
        v = model.matrix[idxs].mean(axis=0).astype(np.float32)
        nrm = float(np.linalg.norm(v))
        if nrm > 1e-9:
            norm_matrix[j] = v / nrm

    norm_word2idx = {w: i for i, w in enumerate(norm_vocab)}
    print(f"  {len(norm_vocab):,} normalized vocab vectors ready")

    # Load target pages
    pages: dict[int, dict] = {}
    seq = 0
    with args.jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not (row.get("title") and (row.get("intro") or row.get("summary") or row.get("text"))):
                continue
            seq += 1
            pid = int(row.get("id") or row.get("page_id") or seq)
            if pid in target_ids:
                pages[pid] = row
            if len(pages) == len(target_ids):
                break

    processed = 0
    for pid in sorted(target_ids):
        row = pages.get(pid)
        if not row:
            continue
        title = row.get("title", "")
        intro = row.get("intro") or row.get("summary") or row.get("text") or ""

        # Article words that have vectors
        art_ws = [w for w in article_words(intro, title) if w in norm_word2idx]
        if not art_ws:
            continue

        art_idxs = [norm_word2idx[w] for w in art_ws]
        art_mat = norm_matrix[art_idxs]  # (N_art, d), already unit vectors

        # One-shot similarity matrix: (N_art, V) — all article words vs full vocab.
        # Memory: ~140 MB for N_art=100, V=350k, d=300 — fine for 54 GB nodes.
        V = len(norm_vocab)
        S = art_mat @ norm_matrix.T  # (N_art, V), cosine sim (both unit vectors)

        # Top-K per article word via argpartition — no Python loop over vocab.
        k = min(args.top_k, V)
        top_k_cols = np.argpartition(S, -k, axis=1)[:, -k:]          # (N_art, k)
        top_k_sims = S[np.arange(len(art_ws))[:, None], top_k_cols]  # (N_art, k)

        # Filter by min_sim and build inverted index.
        inverted: dict[str, dict[str, float]] = {}
        valid_art, valid_col = np.where(top_k_sims >= args.min_sim)
        if len(valid_art):
            sim_vals = top_k_sims[valid_art, valid_col]
            vocab_idxs = top_k_cols[valid_art, valid_col]
            for idx in range(len(valid_art)):
                art_w = art_ws[int(valid_art[idx])]
                vocab_w = norm_vocab[int(vocab_idxs[idx])]
                sv = round(float(sim_vals[idx]), 3)
                entry = inverted.setdefault(vocab_w, {})
                if art_w not in entry or entry[art_w] < sv:
                    entry[art_w] = sv

        out_path = args.out / f"{pid}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(inverted, f, ensure_ascii=False, separators=(",", ":"))

        kb = out_path.stat().st_size // 1024
        print(f"  page {pid} ({title}): {len(art_ws)} art words → {len(inverted)} vocab mappings ({kb} KB)")

        if args.upload:
            upload_to_r2(args.bucket, f"sims/{pid}.json", out_path, remote=not args.local)

        processed += 1

    print(f"\nDone: {processed}/{len(target_ids)} pages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
