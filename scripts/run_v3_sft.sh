#!/usr/bin/env bash
# v3 SFT: fine-tune Qwen3-4B on Claude-generated gameplay data.
# Run after scripts/run_claude_sft_gen.sh has produced data/claude_sft_1000.jsonl.
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_NAME="${RUN_NAME:-v3_sft_claude}"
TRAIN_JSONL="${TRAIN_JSONL:-data/claude_sft_1000.jsonl}"
OUT_DIR="models/${RUN_NAME}"
LOG_DIR="logs/${RUN_NAME}"
MAX_STEPS="${MAX_STEPS:-1000}"
EVAL_EVERY="${EVAL_EVERY:-200}"
SAVE_EVERY="${SAVE_EVERY:-$EVAL_EVERY}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
LORA_RANK="${LORA_RANK:-32}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
RESUME="${RESUME:-}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

RESUME_ARG=()
if [ -n "$RESUME" ]; then
  RESUME_ARG=(--resume-from-checkpoint "$RESUME")
fi

"$PYTHON" -m pedantix_project.cli llm-sft \
  --train "$TRAIN_JSONL" \
  --model Qwen/Qwen3-4B \
  --output-dir "$OUT_DIR" \
  --log "${OUT_DIR}_training.jsonl" \
  --max-steps "$MAX_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --learning-rate "$LEARNING_RATE" \
  --lora-rank "$LORA_RANK" \
  --tiny-model models/tiny_model.json \
  --eval-every "$EVAL_EVERY" \
  --eval-corpus data/clean_pages.jsonl \
  --eval-pages 50 \
  --eval-max-game-steps 30 \
  --eval-chat-format none \
  --eval-num-generations 4 \
  --eval-batch-size 4 \
  --save-steps "$SAVE_EVERY" \
  --save-total-limit 3 \
  "${RESUME_ARG[@]}" \
  --seed 42 \
  2>&1 | tee -a "${LOG_DIR}/stdout.txt"
