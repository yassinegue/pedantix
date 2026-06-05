#!/usr/bin/env bash
# SFT POC: train Qwen3-4B on merged T1+T2 deepseek/cerebras trajectories.
#
# 3,164 examples from deepseek-v4-pro + deepseek32 + cerebras.
# Score-weighted loss: completion gradient scales by game score / 100.
# Warm-start coverage: 0 (cold) to 25 (near-solve), cycles through all phases.
#
# Expected runtime: ~3-4 hours on RTX A5000.
# Wake up to eval results at step 250 / 500.
set -euo pipefail

cd /Data/yassine.guennoun/pedantix

PYTHON="${PYTHON:-/users/eleves-b/2022/yassine.guennoun/.conda/envs/py310/bin/python}"
TRAIN_DATA="data/sft_poc_text.jsonl"
OUT_DIR="models/v3_sft_poc_06b"
LOG="/tmp/sft_poc_$(date +%Y%m%d_%H%M%S).log"

echo "Training SFT POC → $OUT_DIR"
echo "Data: $(wc -l < $TRAIN_DATA) examples in $TRAIN_DATA"
echo "Log: $LOG"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" -u -m pedantix_project.cli llm-sft \
  --train "$TRAIN_DATA" \
  --model Qwen/Qwen3-0.6B \
  --output-dir "$OUT_DIR" \
  --max-steps 800 \
  --batch-size 4 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-5 \
  --lora-rank 32 \
  --seed 42 \
  --save-steps 100 \
  --save-total-limit 3 \
  --eval-every 250 \
  --eval-pages 50 \
  --eval-corpus data/clean_pages.jsonl \
  --tiny-model models/tiny_model.json \
  --eval-chat-format qwen \
  --eval-num-generations 4 \
  --eval-batch-size 4 \
  --score-min 20 \
  --log "$OUT_DIR/training_log.json" \
  2>&1 | tee "$LOG"

echo "Done. Model saved → $OUT_DIR"
