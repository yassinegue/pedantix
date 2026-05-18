#!/usr/bin/env bash
# Baseline: play 100 Pedantix games with Claude Haiku and measure solve rate.
# Usage:
#   ANTHROPIC_API_KEY=sk-ant-... bash scripts/run_claude_eval.sh
#   or: API_KEY=sk-ant-... bash scripts/run_claude_eval.sh
set -euo pipefail

cd /Data/yassine.guennoun/pedantix

PYTHON="${PYTHON:-/users/eleves-b/2022/yassine.guennoun/.conda/envs/py310/bin/python}"
N_PAGES="${N_PAGES:-100}"
MAX_STEPS="${MAX_STEPS:-30}"
CLAUDE_MODEL="${CLAUDE_MODEL:-claude-haiku-4-5-20251001}"
OUTPUT="${OUTPUT:-models/claude_baseline.jsonl}"
SEED="${SEED:-42}"

API_KEY="${API_KEY:-${ANTHROPIC_API_KEY:-}}"
if [ -z "$API_KEY" ]; then
  echo "ERROR: set ANTHROPIC_API_KEY or API_KEY"
  exit 1
fi

"$PYTHON" -m pedantix_project.cli claude-eval \
  --pages data/clean_pages.jsonl \
  --tiny-model models/tiny_model.json \
  --n-pages "$N_PAGES" \
  --max-steps "$MAX_STEPS" \
  --claude-model "$CLAUDE_MODEL" \
  --output "$OUTPUT" \
  --seed "$SEED" \
  --api-key "$API_KEY"
