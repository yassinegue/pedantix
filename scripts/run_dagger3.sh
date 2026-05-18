#!/usr/bin/env bash
# dagger4: more frequent DAgger (every 50), more pairs (40p/160bc), G=16, rollout=4, near-solve oversampling
set -euo pipefail

cd /Data/yassine.guennoun/pedantix

PYTHON="${PYTHON:-/users/eleves-b/2022/yassine.guennoun/.conda/envs/py310/bin/python}"
HF_ROOT="/Data/yassine.guennoun/pedantix/models/hf_cache"

export HF_HOME="$HF_ROOT"
export HF_HUB_CACHE="$HF_ROOT/hub"
export TRANSFORMERS_CACHE="$HF_ROOT/transformers"
export HF_DATASETS_CACHE="$HF_ROOT/datasets"
export TORCH_HOME="$HF_ROOT/torch"
export XDG_CACHE_HOME="$HF_ROOT/xdg"

RUN_NAME="${RUN_NAME:-v7_rollout4_bs16g16_dagger4}"
MODEL="${MODEL:-models/v7_rollout2_bs16g8_dagger3_kl_fix}"
TRAIN_JSONL="data/oracle_combined_v7.jsonl"
OUT_DIR="models/${RUN_NAME}"
LOG_DIR="logs/${RUN_NAME}"
TRAIN_STEPS="${TRAIN_STEPS:-1500}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

"$PYTHON" -m pedantix_project.cli llm-vocab-grpo \
  --train "$TRAIN_JSONL" \
  --pages data/clean_pages.jsonl \
  --model "$MODEL" \
  --tiny-model models/tiny_model.json \
  --output-dir "$OUT_DIR" \
  --log "${OUT_DIR}_training.jsonl" \
  --plot "${LOG_DIR}/rewards.png" \
  --max-steps "$TRAIN_STEPS" \
  --batch-size 16 \
  --num-generations 16 \
  --action-size 8000 \
  --learning-rate 2e-5 \
  --lora-rank 32 \
  --rollout-steps 4 \
  --kl-ref-coef 0.2 \
  --dagger-every 50 \
  --dagger-pages 40 \
  --dagger-bc-steps 160 \
  --seed 401 \
  2>&1 | tee "${LOG_DIR}/stdout.txt"
