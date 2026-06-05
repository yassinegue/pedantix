#!/usr/bin/env bash
# Relaunch deepseek-v4-pro trajectory generation with full warm_start coverage.
# Schedule: 0 (cold), 5, 10, 15, 20, 25 (near-solve) — uniform cycling.
# Saves score field in each JSONL row for future reward-weighted training.
set -euo pipefail

cd /Data/yassine.guennoun/pedantix

PYTHON="${PYTHON:-/users/eleves-b/2022/yassine.guennoun/.conda/envs/py310/bin/python}"
API_KEY="${DO_INFERENCE_KEY:-$(grep '^DO_INFERENCE_KEY' .env 2>/dev/null | awk '{print $3}')}"
if [ -z "$API_KEY" ]; then
  echo "ERROR: DO_INFERENCE_KEY not set"
  exit 1
fi

OUT="data/sft_deepseek_v5.jsonl"
LOG="/tmp/sft_deepseek_v5.log"

echo "Launching deepseek-v4-pro generation → $OUT"
echo "Log: $LOG"

"$PYTHON" -u -m pedantix_project.cli do-traj-gen \
  --pages data/clean_pages.jsonl \
  --tiny-model models/tiny_model.json \
  --n-pages 800 \
  --max-steps 30 \
  --do-model deepseek-r1-distill-qwen-14b \
  --output "$OUT" \
  --seed 9999 \
  --warm-start-schedule "0,5,10,15,20,25" \
  --api-key "$API_KEY" \
  --workers 1 \
  --request-delay 1.5 \
  > "$LOG" 2>&1 &

echo "PID $!"
echo "Monitor: tail -f $LOG"
