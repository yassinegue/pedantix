#!/usr/bin/env bash
# Wait for PID 1061814 (other user's GPU job) to finish, then launch Qwen3-4B SFT.
# Run this in the background: nohup bash scripts/wait_and_train_4b.sh &
set -euo pipefail

cd /Data/yassine.guennoun/pedantix

PYTHON="/users/eleves-b/2022/yassine.guennoun/.conda/envs/py310/bin/python"
BLOCKING_PID=1061814
OUT_DIR="models/v3_sft_poc_4b"
TRAIN_DATA="data/sft_poc_chat.jsonl"
LOG="/tmp/sft_poc_4b_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date)] Waiting for PID $BLOCKING_PID to release GPU..."
while kill -0 $BLOCKING_PID 2>/dev/null; do
    sleep 60
done

echo "[$(date)] PID $BLOCKING_PID done. Waiting 30s for GPU memory to clear..."
sleep 30

echo "[$(date)] Launching Qwen3-4B SFT → $OUT_DIR"
mkdir -p "$OUT_DIR"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" -u -m pedantix_project.cli llm-sft \
  --train "$TRAIN_DATA" \
  --model Qwen/Qwen3-4B \
  --output-dir "$OUT_DIR" \
  --max-steps 600 \
  --batch-size 1 \
  --gradient-accumulation-steps 32 \
  --learning-rate 2e-5 \
  --lora-rank 32 \
  --seed 42 \
  --save-steps 100 \
  --save-total-limit 3 \
  --eval-every 200 \
  --eval-pages 50 \
  --eval-corpus data/clean_pages.jsonl \
  --tiny-model models/tiny_model.json \
  --eval-chat-format qwen \
  --eval-num-generations 4 \
  --eval-batch-size 4 \
  --score-min 20 \
  --log "$OUT_DIR/training_log.json" \
  2>&1 | tee "$LOG"

echo "[$(date)] Done. Model → $OUT_DIR"
