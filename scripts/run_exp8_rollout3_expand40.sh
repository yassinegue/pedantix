#!/usr/bin/env bash
# Exp 8: rollout_steps=3, dynamic_expand_k=40
#
# Tests two things vs exp7 (rollout=2, expand=20):
#   - Longer rollout: 3 gradient steps per episode → richer credit assignment signal
#   - Wider vocab expansion: top-40 TinyModel neighbors → more title words reachable
#
# Includes bc_loss fix: oracle target is force-unmasked in ep_mask before BC loss,
# eliminating the 62.5M spikes seen in exp7.
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

RUN_NAME="${RUN_NAME:-exp8_rollout3_expand40}"
MODEL="${MODEL:-models/v7_rollout4_bs16g16_dagger4}"
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
  --plot-every 20 \
  --max-steps 1000 \
  --batch-size 16 \
  --num-generations 16 \
  --action-size 8000 \
  --dynamic-expand-k 40 \
  --freeze-backbone \
  --learning-rate 1e-4 \
  --rollout-steps 3 \
  --dagger-every 10 \
  --dagger-pages 16 \
  --dagger-bc-steps 50 \
  --kl-ref-coef 0.01 \
  --min-entropy 3.0 \
  --min-entropy-coef 2.0 \
  --bc-coef 1.0 \
  --bc-warmup-steps 50 \
  --seed 42 \
  2>&1 | tee "${LOG_DIR}/stdout.txt"
