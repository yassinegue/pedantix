#!/usr/bin/env bash
set -euo pipefail
cd /Data/yassine.guennoun/pedantix

PYTHON="/users/eleves-b/2022/yassine.guennoun/.conda/envs/py310/bin/python"
OUT_DIR="models/v4_sft_4b"
TRAIN_DATA="data/sft_poc_chat.jsonl"
LOG="/tmp/sft_v4_4b_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$OUT_DIR"
echo "[$(date)] Launching Qwen3-4B SFT (fixed completion-only loss) → $OUT_DIR"

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
  --save-steps 200 \
  --save-total-limit 3 \
  --eval-every 200 \
  --eval-pages 50 \
  --eval-corpus data/clean_pages.jsonl \
  --tiny-model models/tiny_model.json \
  --eval-chat-format qwen \
  --eval-num-generations 4 \
  --eval-batch-size 4 \
  --score-min 0 \
  --log "$OUT_DIR/training_log.json" \
  2>&1 | tee "$LOG"

echo "[$(date)] Done → $OUT_DIR"
