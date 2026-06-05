#!/usr/bin/env bash
# CoT SFT: distill DeepSeek's chain-of-thought reasoning into Qwen3-4B.
# Training data: DeepSeek-generated <think>...reasoning...</think>\nMOT: word traces.
# The model learns WHEN to pivot domains, HOW to follow semantic scores — not just format.
set -euo pipefail
cd /Data/yassine.guennoun/pedantix

PYTHON="${PYTHON:-/users/eleves-b/2022/yassine.guennoun/.conda/envs/py310/bin/python}"
HF_ROOT="models/hf_cache"
export HF_HOME="$HF_ROOT" HF_HUB_CACHE="$HF_ROOT/hub"
export TRANSFORMERS_CACHE="$HF_ROOT/transformers" HF_DATASETS_CACHE="$HF_ROOT/datasets"
export TORCH_HOME="$HF_ROOT/torch" XDG_CACHE_HOME="$HF_ROOT/xdg"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN="${TRAIN:-data/sft_deepseek_cot_20260526_005410.jsonl}"
OUT_DIR="${OUT_DIR:-models/v8_cot_sft}"
LOG="/tmp/cot_sft_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$OUT_DIR"
echo "[$(date)] Launching CoT SFT → $OUT_DIR  log: $LOG"
echo "  Train: $TRAIN"

nohup "$PYTHON" -u -m pedantix_project.cli llm-sft \
  --train "$TRAIN" \
  --model Qwen/Qwen3-4B \
  --output-dir "$OUT_DIR" \
  --max-steps 800 \
  --batch-size 1 \
  --gradient-accumulation-steps 32 \
  --learning-rate 2e-5 \
  --lora-rank 32 \
  --seed 42 \
  --save-steps 400 \
  --eval-every 100 \
  --eval-pages 12 \
  --eval-corpus data/clean_pages.jsonl \
  --tiny-model models/tiny_model.json \
  --eval-chat-format qwen \
  --eval-num-generations 4 \
  --eval-batch-size 4 \
  --log "$OUT_DIR/training_log.json" \
  --no-stop-on-garbage \
  --max-seq-length 3000 \
  > "$LOG" 2>&1 &

echo "PID: $!  Log: $LOG"
echo "Tail: tail -f $LOG"
