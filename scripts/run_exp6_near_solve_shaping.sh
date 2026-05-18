#!/usr/bin/env bash
# Exp 6: Near-solve embedding shaping reward (Option B POC)
#
# Hypothesis: adding a bonus = shaping_coef * max_sim(guess, unrevealed_title_words)
# gives GRPO a dense reward signal even at 0% exact solve rate, breaking the cold
# start that killed both v7 (no solves in 1500 steps) and exp5 (flat entropy/KL).
#
# Scale: ~200 steps, rollout_steps=2, frozen backbone from v7 → ~2h wall clock.
# Backbone is frozen so only the action head trains (fast iteration, clean ablation).
# Compare solve_rate and shaped_bonus trajectory vs v7 (same oracle data, no shaping).
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

RUN_NAME="${RUN_NAME:-exp6_near_solve_shaping}"
# Start from v7's trained backbone (frozen) — already learned good content-word representations
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
  --freeze-backbone \
  --learning-rate 1e-4 \
  --rollout-steps 2 \
  --dagger-every 10 \
  --dagger-pages 16 \
  --dagger-bc-steps 50 \
  --near-solve-shaping-coef 30.0 \
  --kl-ref-coef 0.01 \
  --min-entropy 3.0 \
  --min-entropy-coef 2.0 \
  --seed 42 \
  2>&1 | tee "${LOG_DIR}/stdout.txt"
