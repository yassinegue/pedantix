#!/usr/bin/env bash
# Download FastText cc.fr.300 vectors and filter to Wikipedia vocabulary.
#
# Step 1: download cc.fr.300.vec.gz (~1.6 GB) from Facebook AI
# Step 2: stream through it, keep only words in clean_pages.jsonl corpus
# Step 3: save compact JSON model (~120 MB, ~100K words)
#
# Usage:
#   bash scripts/build_fasttext_model.sh
#   Skip download if already present: VEC_GZ=models/fasttext/cc.fr.300.vec.gz bash scripts/build_fasttext_model.sh
set -euo pipefail

cd /Data/yassine.guennoun/pedantix

PYTHON="${PYTHON:-/users/eleves-b/2022/yassine.guennoun/.conda/envs/py310/bin/python}"
VEC_GZ="${VEC_GZ:-models/fasttext/cc.fr.300.vec.gz}"
OUTPUT="${OUTPUT:-models/fasttext_wiki_model.npz}"

mkdir -p models/fasttext

# Step 1: download if not present
if [ ! -f "$VEC_GZ" ]; then
  echo "Downloading FastText French vectors..."
  "$PYTHON" -m pedantix_project.cli fasttext-download --dest-dir models/fasttext
else
  echo "Using existing: $VEC_GZ"
fi

# Step 2+3: filter to Wikipedia vocab and save
echo "Building compact model..."
"$PYTHON" -m pedantix_project.cli fasttext-build \
  --pages data/clean_pages.jsonl \
  --vec-gz "$VEC_GZ" \
  --output "$OUTPUT"

echo "Done: $OUTPUT"
ls -lh "$OUTPUT"
