#!/usr/bin/env bash
# Generate SFT training data from Claude gameplay.
# Starts with N_PAGES=1000 to validate, then scale up.
# Usage:
#   ANTHROPIC_API_KEY=sk-ant-... bash scripts/run_claude_sft_gen.sh
#   or: N_PAGES=5000 API_KEY=sk-ant-... bash scripts/run_claude_sft_gen.sh
set -euo pipefail

cd /Data/yassine.guennoun/pedantix

PYTHON="${PYTHON:-/users/eleves-b/2022/yassine.guennoun/.conda/envs/py310/bin/python}"
N_PAGES="${N_PAGES:-1000}"
MAX_STEPS="${MAX_STEPS:-30}"
CLAUDE_MODEL="${CLAUDE_MODEL:-claude-haiku-4-5-20251001}"
OUTPUT="${OUTPUT:-data/claude_sft_${N_PAGES}.jsonl}"
SEED="${SEED:-42}"

API_KEY="${API_KEY:-${ANTHROPIC_API_KEY:-}}"
if [ -z "$API_KEY" ]; then
  echo "ERROR: set ANTHROPIC_API_KEY or API_KEY"
  exit 1
fi

mkdir -p data models/logs

"$PYTHON" -m pedantix_project.cli claude-sft-gen \
  --pages data/clean_pages.jsonl \
  --tiny-model models/tiny_model.json \
  --n-pages "$N_PAGES" \
  --max-steps "$MAX_STEPS" \
  --claude-model "$CLAUDE_MODEL" \
  --output "$OUTPUT" \
  --seed "$SEED" \
  --api-key "$API_KEY"

echo "Done. Training data: $OUTPUT"
wc -l "$OUTPUT"
