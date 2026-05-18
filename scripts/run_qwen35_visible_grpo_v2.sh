#!/usr/bin/env bash
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

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
RUN_NAME="${RUN_NAME:-qwen35_4b_visible_grpo_info_v2}"
PREP_SAMPLE_PAGES="${PREP_SAMPLE_PAGES:-100000}"
PREP_STATES_PER_PAGE="${PREP_STATES_PER_PAGE:-4}"
PREP_ACTION_SIZE="${PREP_ACTION_SIZE:-8000}"
TRAIN_STEPS="${TRAIN_STEPS:-3000}"
TRAIN_JSONL="data/llm_grpo_visible_info_v2.jsonl"
OUT_DIR="models/${RUN_NAME}"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$OUT_DIR"

"$PYTHON" -m unittest discover -s tests -v

"$PYTHON" -m pedantix_project.cli llm-prepare \
  --pages data/clean_pages.jsonl \
  --tiny-model models/tiny_model.json \
  --output "$TRAIN_JSONL" \
  --sample-pages "$PREP_SAMPLE_PAGES" \
  --states-per-page "$PREP_STATES_PER_PAGE" \
  --curriculum-max-title-words 3 \
  --action-size "$PREP_ACTION_SIZE" \
  --max-game-steps 100 \
  --chat-format qwen \
  --seed 301

"$PYTHON" -m pedantix_project.cli llm-grpo \
  --train "$TRAIN_JSONL" \
  --model "$MODEL" \
  --tiny-model models/tiny_model.json \
  --output-dir "$OUT_DIR" \
  --log "${OUT_DIR}_training.jsonl" \
  --plot "${OUT_DIR}_reward.png" \
  --max-steps "$TRAIN_STEPS" \
  --batch-size 16 \
  --gradient-accumulation-steps 1 \
  --num-generations 8 \
  --max-completion-length 8 \
  --learning-rate 7e-7 \
  --lora-rank 32 \
  --logging-steps 5 \
  --save-steps 500 \
  --seed 311

"$PYTHON" -m pedantix_project.cli llm-eval \
  --pages data/clean_pages.jsonl \
  --model "$OUT_DIR" \
  --tiny-model models/tiny_model.json \
  --sample-pages 200 \
  --max-game-steps 100 \
  --eval-batch-size 16 \
  --output "${OUT_DIR}_eval_100.jsonl" \
  --chat-format qwen \
  --seed 321

head -20 "${OUT_DIR}_eval_100.jsonl"
