#!/usr/bin/env bash
# Generate CoT SFT trajectories — DeepSeek reasons in <think>...</think> before each guess.
# Chess distillation recipe: 4B model distilled from CoT expert >> plain imitation.
set -euo pipefail
cd /Data/yassine.guennoun/pedantix

PYTHON="/users/eleves-b/2022/yassine.guennoun/.conda/envs/py310/bin/python"
API_KEY="${API_KEY:-${DO_INFERENCE_KEY:-}}"
if [ -z "$API_KEY" ]; then
  echo "ERROR: set DO_INFERENCE_KEY or API_KEY"; exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="data/sft_deepseek_cot_${TIMESTAMP}.jsonl"
LOG="/tmp/do_cot_traj_${TIMESTAMP}.log"
N_PAGES="${N_PAGES:-800}"

echo "Launching CoT trajectory gen → $OUTPUT  log: $LOG"

nohup "$PYTHON" -u -m pedantix_project.cli do-traj-gen \
  --pages data/clean_pages.jsonl \
  --tiny-model models/tiny_model.json \
  --n-pages "$N_PAGES" \
  --max-steps 30 \
  --do-model deepseek-v4-pro \
  --output "$OUTPUT" \
  --seed 55555 \
  --warm-start-schedule 0,5,10,15,20,25 \
  --api-key "$API_KEY" \
  --workers 3 \
  --request-delay 1.0 \
  --cot \
  > "$LOG" 2>&1 &

echo "PID: $!  Output: $OUTPUT"
echo "Tail: tail -f $LOG"
