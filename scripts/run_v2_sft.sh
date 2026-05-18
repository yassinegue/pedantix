#!/usr/bin/env bash
# v2 Stage 1: SFT warmup of Qwen3-4B + LoRA on the two-tier oracle mix
# (80% oracle_v7_1word + 20% oracle_20k_10state_v7, built via build_v2_sft_mix.py).
#
# Goal: teach the model to emit "MOT: <word>" in the right format, biased
# toward title words on near-solve states and sensible French words on early
# states. Target acceptance gate: >=10% solve_rate on the 50-page held-out eval.
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
# Reduce VRAM fragmentation that triggered eval-time OOM on the 24GB A5000.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_NAME="${RUN_NAME:-v2_sft_qwen4b_r16}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
TRAIN_JSONL="${TRAIN_JSONL:-data/v2_sft_mix.jsonl}"
EVAL_CORPUS="${EVAL_CORPUS:-data/clean_pages.jsonl}"
OUT_DIR="models/${RUN_NAME}"
LOG_DIR="logs/${RUN_NAME}"
MAX_STEPS="${MAX_STEPS:-3000}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SAVE_EVERY="${SAVE_EVERY:-$EVAL_EVERY}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
RESUME="${RESUME:-auto}"
# eval VRAM tuning: at batch=8 × num_return=4 = 32 seqs, generate's KV-cache
# allocation overflows the ~500MB headroom left on a 24GB A5000 after training
# state. Halving the batch keeps the eval within the available free pool.
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_NUM_GENERATIONS="${EVAL_NUM_GENERATIONS:-4}"
EVAL_PAGES="${EVAL_PAGES:-50}"
EVAL_MAX_GAME_STEPS="${EVAL_MAX_GAME_STEPS:-30}"
EVAL_CHAT_FORMAT="${EVAL_CHAT_FORMAT:-none}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
LORA_RANK="${LORA_RANK:-16}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

if [ ! -f "$TRAIN_JSONL" ]; then
  echo "Building SFT mix at $TRAIN_JSONL"
  "$PYTHON" scripts/build_v2_sft_mix.py --output "$TRAIN_JSONL"
fi

"$PYTHON" -m pedantix_project.cli llm-sft \
  --train "$TRAIN_JSONL" \
  --model "$MODEL" \
  --output-dir "$OUT_DIR" \
  --max-steps "$MAX_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --learning-rate "$LEARNING_RATE" \
  --lora-rank "$LORA_RANK" \
  --log "${OUT_DIR}_training.jsonl" \
  --tiny-model models/tiny_model.json \
  --eval-corpus "$EVAL_CORPUS" \
  --eval-every "$EVAL_EVERY" \
  --eval-pages "$EVAL_PAGES" \
  --eval-max-game-steps "$EVAL_MAX_GAME_STEPS" \
  --eval-num-generations "$EVAL_NUM_GENERATIONS" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --eval-chat-format "$EVAL_CHAT_FORMAT" \
  --save-steps "$SAVE_EVERY" \
  --save-total-limit "$SAVE_TOTAL_LIMIT" \
  --resume-from-checkpoint "$RESUME" \
  --seed 42 \
  2>&1 | tee -a "${LOG_DIR}/stdout.txt"
