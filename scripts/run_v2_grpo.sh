#!/usr/bin/env bash
# v2 Stage 2: GRPO from the SFT checkpoint, with KL anchor (beta=0.04) and
# clipped solve bonus (+1000 -> +50 via solve-bonus-scale 0.05).
#
# Resumes the LoRA adapter saved at models/v2_sft_qwen4b_r16 by default —
# train_llm_grpo auto-loads adapters via _load_peft_model_if_adapter.
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

RUN_NAME="${RUN_NAME:-v2_grpo_qwen4b}"
SFT_MODEL="${SFT_MODEL:-models/sft_mini_test/checkpoint-200}"
TRAIN_JSONL="${TRAIN_JSONL:-data/v2_sft_mix.jsonl}"
EVAL_CORPUS="${EVAL_CORPUS:-data/clean_pages.jsonl}"
OUT_DIR="models/${RUN_NAME}"
LOG_DIR="logs/${RUN_NAME}"
MAX_STEPS="${MAX_STEPS:-1500}"
EVAL_EVERY="${EVAL_EVERY:-250}"
SAVE_EVERY="${SAVE_EVERY:-$EVAL_EVERY}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
RESUME="${RESUME:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_NUM_GENERATIONS="${EVAL_NUM_GENERATIONS:-4}"
EVAL_PAGES="${EVAL_PAGES:-50}"
EVAL_MAX_GAME_STEPS="${EVAL_MAX_GAME_STEPS:-30}"
EVAL_CHAT_FORMAT="${EVAL_CHAT_FORMAT:-none}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
# TRL constraint: BATCH_SIZE must be divisible by NUM_GENERATIONS.
LEARNING_RATE="${LEARNING_RATE:-5e-7}"
LORA_RANK="${LORA_RANK:-32}"
DAGGER_EVERY="${DAGGER_EVERY:-50}"
DAGGER_PAGES="${DAGGER_PAGES:-32}"
DAGGER_ROLLOUT_STEPS="${DAGGER_ROLLOUT_STEPS:-10}"
DAGGER_MICROSTEPS="${DAGGER_MICROSTEPS:-16}"
DAGGER_BC_BATCH_SIZE="${DAGGER_BC_BATCH_SIZE:-2}"
DAGGER_HISTORY_MAX_STEPS="${DAGGER_HISTORY_MAX_STEPS:-30}"
DAGGER_ORACLE_MODE="${DAGGER_ORACLE_MODE:-soft}"
DAGGER_ORACLE_TOP_K="${DAGGER_ORACLE_TOP_K:-8}"
DAGGER_ORACLE_TEMPERATURE="${DAGGER_ORACLE_TEMPERATURE:-1.0}"
DAGGER_ORACLE_MIN_IDF="${DAGGER_ORACLE_MIN_IDF:-2.0}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

if [ ! -d "$SFT_MODEL" ]; then
  echo "ERROR: SFT checkpoint not found at $SFT_MODEL — run scripts/run_v2_sft.sh first."
  exit 1
fi

RESUME_ARG=()
if [ -n "$RESUME" ]; then
  RESUME_ARG=(--resume-from-checkpoint "$RESUME")
fi

"$PYTHON" -m pedantix_project.cli llm-grpo \
  --train "$TRAIN_JSONL" \
  --model "$SFT_MODEL" \
  --tiny-model models/tiny_model.json \
  --output-dir "$OUT_DIR" \
  --log "${OUT_DIR}_training.jsonl" \
  --plot "${LOG_DIR}/rewards.png" \
  --max-steps "$MAX_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --num-generations "$NUM_GENERATIONS" \
  --max-completion-length 8 \
  --learning-rate "$LEARNING_RATE" \
  --lora-rank "$LORA_RANK" \
  --temperature 0.9 \
  --top-p 0.9 \
  --beta 0.04 \
  --solve-bonus-scale 0.05 \
  --logging-steps 1 \
  --save-steps "$SAVE_EVERY" \
  --eval-corpus "$EVAL_CORPUS" \
  --eval-every "$EVAL_EVERY" \
  --eval-pages "$EVAL_PAGES" \
  --eval-max-game-steps "$EVAL_MAX_GAME_STEPS" \
  --eval-num-generations "$EVAL_NUM_GENERATIONS" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --eval-chat-format "$EVAL_CHAT_FORMAT" \
  --dagger-every "$DAGGER_EVERY" \
  --dagger-pages "$DAGGER_PAGES" \
  --dagger-rollout-steps "$DAGGER_ROLLOUT_STEPS" \
  --dagger-microsteps "$DAGGER_MICROSTEPS" \
  --dagger-bc-batch-size "$DAGGER_BC_BATCH_SIZE" \
  --dagger-history-max-steps "$DAGGER_HISTORY_MAX_STEPS" \
  --dagger-oracle-mode "$DAGGER_ORACLE_MODE" \
  --dagger-oracle-top-k "$DAGGER_ORACLE_TOP_K" \
  --dagger-oracle-temperature "$DAGGER_ORACLE_TEMPERATURE" \
  --dagger-oracle-min-idf "$DAGGER_ORACLE_MIN_IDF" \
  "${RESUME_ARG[@]}" \
  --seed 42 \
  2>&1 | tee -a "${LOG_DIR}/stdout.txt"
