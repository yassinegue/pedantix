#!/usr/bin/env python3
"""
Build a French lemma lookup table from our FastText vocabulary.
Outputs: lemmas.json  — {word_form: lemma} for forms where lemma != form.
Uploads to R2 as lemmas.json.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import simplemma

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"
MODEL_PATH = REPO_ROOT / "models" / "fasttext_wiki_model.npz"


def load_vocab() -> list[str]:
    data = np.load(MODEL_PATH, allow_pickle=True)
    vocab = list(data["vocab"])
    print(f"Loaded {len(vocab):,} vocab words", flush=True)
    return vocab


# simplemma incorrectly maps pronouns/determiners to other pronouns/determiners
# (e.g. elles→il, leur→son). Block any entry whose lemma is a common function word
# that has no morphological relationship to the inflected form.
_FUNCTION_WORD_LEMMAS = {
    "il", "elle", "ils", "elles", "on", "y", "en",
    "le", "la", "les", "un", "une", "des",
    "au", "aux", "du",
    "ce", "se", "ne", "me", "te", "lui",
    "moi", "toi", "soi",
    "son", "sa", "ses",
    "mon", "ma", "mes",
    "ton", "ta", "tes",
    "notre", "votre",
}


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def build_lemma_table(vocab: list[str]) -> dict[str, str]:
    table = {}
    errors = 0
    skipped = 0
    for i, word in enumerate(vocab):
        if i % 50000 == 0:
            print(f"  {i:,}/{len(vocab):,} …", flush=True)
        try:
            lemma = simplemma.lemmatize(word, lang="fr")
            if not lemma or lemma == word:
                continue
            # Skip mappings to common function words (simplemma false positives)
            if lemma in _FUNCTION_WORD_LEMMAS:
                skipped += 1
                continue
            table[word] = lemma
        except Exception:
            errors += 1
    print(f"Built table: {len(table):,} entries (skipped: {skipped}, errors: {errors})", flush=True)
    return table


def upload_to_r2(local_path: Path, r2_key: str):
    cmd = [
        "npx", "wrangler", "r2", "object", "put",
        f"pedantix-ranks/{r2_key}",
        "--file", str(local_path),
        "--content-type", "application/json",
        "--remote",
    ]
    for attempt in range(1, 4):
        res = subprocess.run(cmd, cwd=WEB_DIR, check=False)
        if res.returncode == 0:
            return
        print(f"  upload failed (attempt {attempt}/3), retrying…", flush=True)
    raise RuntimeError(f"wrangler upload failed for {r2_key}")


def main():
    if not MODEL_PATH.exists():
        print(f"ERROR: model not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    vocab = load_vocab()
    table = build_lemma_table(vocab)

    payload = json.dumps(table, ensure_ascii=False, separators=(",", ":"))
    print(f"JSON size: {len(payload) / 1024:.1f} KB", flush=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        f.write(payload)
        tmp = Path(f.name)

    print("Uploading to R2 as lemmas.json …", flush=True)
    upload_to_r2(tmp, "lemmas.json")
    tmp.unlink()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
