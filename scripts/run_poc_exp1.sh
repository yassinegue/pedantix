#!/usr/bin/env bash
# Exp 1: single-word-title curriculum, no code changes
# Hypothesis: collapsing to 1-word titles forces 1 title decision → solve_rate > 0
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

RUN_NAME="${RUN_NAME:-poc_exp1_frozen_200}"
MODEL="${MODEL:-models/v7_rollout2_bs16g8_dagger3_kl_fix}"
TRAIN_JSONL="data/oracle_v7_1word.jsonl"
OUT_DIR="models/${RUN_NAME}"
LOG_DIR="logs/${RUN_NAME}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

"$PYTHON" -m pedantix_project.cli llm-vocab-grpo \
  --train "$TRAIN_JSONL" \
  --pages data/clean_pages.jsonl \
  --model "$MODEL" \
  --tiny-model models/tiny_model.json \
  --output-dir "$OUT_DIR" \
  --log "${OUT_DIR}_training.jsonl" \
  --plot "${LOG_DIR}/rewards.png" \
  --plot-every 10 \
  --max-steps 50 \
  --batch-size 16 \
  --num-generations 32 \
  --action-size 200 \
  --freeze-backbone \
  --learning-rate 2e-4 \
  --rollout-steps 2 \
  --dagger-every 10 \
  --dagger-pages 4 \
  --dagger-bc-steps 10 \
  --seed 411 \
  2>&1 | tee "${LOG_DIR}/stdout.txt"
