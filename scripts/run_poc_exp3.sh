#!/usr/bin/env bash
# Exp 3: free-generation LLM-GRPO control on single-word titles
# Hypothesis: pretrained LM prior over French words solves 1-word titles within 20 steps
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

RUN_NAME="${RUN_NAME:-poc_exp3_freegen_v2}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
TRAIN_JSONL="data/oracle_v7_1word.jsonl"
OUT_DIR="models/${RUN_NAME}"
LOG_DIR="logs/${RUN_NAME}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

"$PYTHON" -m pedantix_project.cli llm-grpo \
  --train "$TRAIN_JSONL" \
  --model "$MODEL" \
  --tiny-model models/tiny_model.json \
  --output-dir "$OUT_DIR" \
  --log "${OUT_DIR}_training.jsonl" \
  --plot "${LOG_DIR}/rewards.png" \
  --max-steps 50 \
  --batch-size 16 \
  --num-generations 16 \
  --max-completion-length 12 \
  --temperature 0.7 \
  --lora-rank 32 \
  --learning-rate 2e-5 \
  --logging-steps 1 \
  --seed 413 \
  2>&1 | tee "${LOG_DIR}/stdout.txt"
