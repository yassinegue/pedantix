#!/usr/bin/env bash
# v2 eval: run llm-eval against any v2 checkpoint (SFT or GRPO) on fresh pages.
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

MODEL="${MODEL:-models/v2_grpo_qwen4b}"
PAGES="${PAGES:-data/clean_pages.jsonl}"
SAMPLE_PAGES="${SAMPLE_PAGES:-200}"
MAX_GAME_STEPS="${MAX_GAME_STEPS:-30}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OUTPUT="${OUTPUT:-models/$(basename "$MODEL")_eval.jsonl}"
SEED="${SEED:-42}"

"$PYTHON" -m pedantix_project.cli llm-eval \
  --pages "$PAGES" \
  --model "$MODEL" \
  --tiny-model models/tiny_model.json \
  --sample-pages "$SAMPLE_PAGES" \
  --max-game-steps "$MAX_GAME_STEPS" \
  --output "$OUTPUT" \
  --chat-format qwen \
  --eval-batch-size "$BATCH_SIZE" \
  --eval-num-generations "$NUM_GENERATIONS" \
  --seed "$SEED"
