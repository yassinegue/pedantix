#!/usr/bin/env bash
# SFT Qwen3-4B on oracle trajectories (no API needed).
# Teaches state-conditional word choice from local TinyModel oracle.
set -euo pipefail
cd /Data/yassine.guennoun/pedantix

PYTHON="${PYTHON:-/users/eleves-b/2022/yassine.guennoun/.conda/envs/py310/bin/python}"
HF_ROOT="models/hf_cache"
export HF_HOME="$HF_ROOT" HF_HUB_CACHE="$HF_ROOT/hub"
export TRANSFORMERS_CACHE="$HF_ROOT/transformers" HF_DATASETS_CACHE="$HF_ROOT/datasets"
export TORCH_HOME="$HF_ROOT/torch" XDG_CACHE_HOME="$HF_ROOT/xdg"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT_DIR="models/v7_oracle_sft"
LOG="/tmp/oracle_sft_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$OUT_DIR"
echo "[$(date)] Launching oracle SFT → $OUT_DIR  log: $LOG"

nohup "$PYTHON" -u -m pedantix_project.cli llm-sft \
  --train data/sft_stochastic_oracle.jsonl \
  --model Qwen/Qwen3-4B \
  --output-dir "$OUT_DIR" \
  --max-steps 800 \
  --batch-size 1 \
  --gradient-accumulation-steps 32 \
  --learning-rate 2e-5 \
  --lora-rank 32 \
  --seed 42 \
  --save-steps 300 \
  --eval-every 100 \
  --eval-pages 12 \
  --eval-corpus data/clean_pages.jsonl \
  --tiny-model models/tiny_model.json \
  --eval-chat-format qwen \
  --eval-num-generations 4 \
  --eval-batch-size 4 \
  --log "$OUT_DIR/training_log.json" \
  --no-stop-on-garbage \
  > "$LOG" 2>&1 &

echo "PID: $!  Log: $LOG"
echo "Tail: tail -f $LOG"
echo ""
echo "Note: --save-steps 300 avoids coinciding with --eval-every 100 to prevent step-200 OOM."
