#!/usr/bin/env bash
# Generate GRPO-ready training data with the current make_prompt format.
# Includes title/intro/history columns needed by the reward function,
# and qwen-chat-formatted prompts matching v4_sft_4b's training format.
set -euo pipefail
cd /Data/yassine.guennoun/pedantix

PYTHON="/users/eleves-b/2022/yassine.guennoun/.conda/envs/py310/bin/python"
API_KEY="${API_KEY:-${DO_INFERENCE_KEY:-}}"
if [ -z "$API_KEY" ]; then
  echo "ERROR: set DO_INFERENCE_KEY or API_KEY"; exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="data/grpo_ready_${TIMESTAMP}.jsonl"
LOG="/tmp/grpo_data_gen_${TIMESTAMP}.log"
N_PAGES="${N_PAGES:-300}"

echo "Generating GRPO data → $OUTPUT  log: $LOG"

nohup "$PYTHON" -u -m pedantix_project.cli do-traj-gen \
  --pages data/clean_pages.jsonl \
  --tiny-model models/tiny_model.json \
  --n-pages "$N_PAGES" \
  --max-steps 30 \
  --do-model deepseek-v4-pro \
  --output "$OUTPUT" \
  --seed 11111 \
  --warm-start-schedule 0,5,10,15,20,25 \
  --api-key "$API_KEY" \
  --workers 3 \
  --request-delay 1.0 \
  --grpo-format \
  > "$LOG" 2>&1 &

echo "PID: $!  Output: $OUTPUT"
echo "Tail: tail -f $LOG"
