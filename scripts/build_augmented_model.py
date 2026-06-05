#!/usr/bin/env python3
"""Build an augmented FastText model that includes synthetic vectors for OOV words.

For words present in our Wikipedia corpus but missing from the fasttext model
(e.g. "alors", "dans", "apres") we compute a context vector:
  V(w) = mean of article-level mean-vectors across all articles containing w

This positions OOV words in embedding space based on what they co-occur with.
Common stopwords ("de", "un") end up near the centroid (rank ~900+).
Semi-specific words ("alors", "apres") end up moderately positioned (rank ~400-700).
Topic-specific OOV words end up near the relevant topic cluster.

Output: models/fasttext_wiki_model_augmented.npz  (same format as the original)
"""
from __future__ import annotations

import sys
import json
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


def main():
    jsonl = REPO_ROOT / "data" / "filtered_pages.jsonl"
    orig_npz = REPO_ROOT / "models" / "fasttext_wiki_model.npz"
    out_npz  = REPO_ROOT / "models" / "fasttext_wiki_model_augmented.npz"
    max_articles = 2000
    min_article_count = 3   # OOV word must appear in ≥3 articles

    print(f"Loading model from {orig_npz}…")
    model = FastTextWikiModel.load(orig_npz)
    n, d = model.matrix.shape
    print(f"  {n:,} words × {d} dims")

    # Build normalized vocab → row indices (same as precompute script)
    raw_vocab = [""] * n
    for w, i in model.word2idx.items():
        raw_vocab[i] = w
    buckets: dict[str, list[int]] = {}
    for i, w in enumerate(raw_vocab):
        norm = normalize_for_web(w)
        if not norm or len(norm) < 2:
            continue
        buckets.setdefault(norm, []).append(i)

    in_norm: set[str] = set(buckets.keys())
    print(f"  {len(in_norm):,} normalized forms in model")

    # Build normalized word matrix (same as precompute)
    norm_vocab_list = sorted(buckets.keys())
    M = len(norm_vocab_list)
    norm_word2idx = {w: i for i, w in enumerate(norm_vocab_list)}
    norm_matrix = np.zeros((M, d), dtype=np.float32)
    for j, w in enumerate(norm_vocab_list):
        idxs = buckets[w]
        v = model.matrix[idxs].mean(axis=0)
        nrm = float(np.linalg.norm(v))
        if nrm > 1e-9:
            v = v / nrm
        norm_matrix[j] = v

    # --- Pass 1: build per-article mean vectors and OOV word→article mapping ---
    print(f"\nReading {jsonl} (up to {max_articles} articles)…")
    article_mean_vecs: list[np.ndarray | None] = []  # index = seq-1
    oov_articles: dict[str, list[int]] = {}          # oov_norm → [article_idxs]

    seq = 0
    with jsonl.open("r", encoding="utf-8") as f:
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
            if seq > max_articles:
                break

            art_idx = seq - 1  # 0-based

            # Collect in-model words and OOV words from intro+title
            in_model_idxs: list[int] = []
            oov_words_here: set[str] = set()
            for chunk in (intro + " " + title).split():
                n_w = normalize_for_web(chunk)
                if not n_w or len(n_w) < 2:
                    continue
                if n_w in norm_word2idx:
                    in_model_idxs.append(norm_word2idx[n_w])
                else:
                    oov_words_here.add(n_w)

            # Article mean vector from in-model words
            if in_model_idxs:
                v = norm_matrix[in_model_idxs].mean(axis=0)
                nrm = float(np.linalg.norm(v))
                if nrm > 1e-9:
                    v = v / nrm
                article_mean_vecs.append(v)
            else:
                article_mean_vecs.append(None)

            for w in oov_words_here:
                oov_articles.setdefault(w, []).append(art_idx)

    print(f"  {seq} articles read, {len(oov_articles):,} unique OOV normalized words")

    # --- Pass 2: compute synthetic vectors for qualifying OOV words ---
    qualified = {w: arts for w, arts in oov_articles.items()
                 if len(arts) >= min_article_count}
    print(f"  {len(qualified):,} OOV words appear in ≥{min_article_count} articles → computing synthetic vectors")

    oov_norm_list = sorted(qualified.keys())
    oov_matrix = np.zeros((len(oov_norm_list), d), dtype=np.float32)

    for j, w in enumerate(oov_norm_list):
        art_idxs = qualified[w]
        vecs = [article_mean_vecs[i] for i in art_idxs if article_mean_vecs[i] is not None]
        if not vecs:
            continue
        v = np.mean(vecs, axis=0).astype(np.float32)
        nrm = float(np.linalg.norm(v))
        if nrm > 1e-9:
            v = v / nrm
        oov_matrix[j] = v

    # Show a few examples
    print("\n  Sample OOV word coverage:")
    samples = ["alors", "dans", "apres", "deux", "annees", "lors", "films", "tous", "autres", "de", "un"]
    for w in samples:
        if w in qualified:
            print(f"    {w!r}: {len(qualified[w])} articles → synthetic vector ✓")
        else:
            print(f"    {w!r}: not in corpus / < {min_article_count} articles")

    # --- Merge original normalized vocab + OOV words ---
    combined_words = norm_vocab_list + oov_norm_list
    combined_matrix = np.concatenate([norm_matrix, oov_matrix], axis=0)
    print(f"\nAugmented model: {len(combined_words):,} words ({len(oov_norm_list):,} synthetic OOV vectors added)")

    # Save as NPZ using the same format as FastTextWikiModel
    # The model's word2idx uses RAW words; we save normalized words here
    # since precompute_puzzle_ranks.py uses build_normalized_vocab() which
    # collapses raw→norm anyway. We'll store them in the "vocab" array
    # and trust precompute to read it.
    # But FastTextWikiModel.load() uses raw word2idx from the NPZ's "vocab" key.
    # So we save combined_words as the "vocab" key.
    vocab_arr = np.array(combined_words, dtype=object)
    np.savez_compressed(
        str(out_npz),
        matrix=combined_matrix,
        vocab=vocab_arr,
    )
    print(f"Saved → {out_npz} ({out_npz.stat().st_size // 1024 // 1024} MB)")


if __name__ == "__main__":
    main()
