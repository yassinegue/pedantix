#!/usr/bin/env bash
# v7 Trajectory GRPO: model plays full games from turn 0 on its own rollouts.
# Fixes the bandit GRPO state-distribution mismatch (eval at turn 0 vs train at turn 18).
#
# Key differences from v6_grpo:
#   - No static dataset — live OnlineGameBuffer of 64 active games
#   - Model experiences cold-start (turn 0) and its own causal chain
#   - solve_bonus_scale=1.0 (solving is the main signal, not shaping)
#   - beta=0.05 (lower KL — model needs to explore strategy)
#   - lr=1e-6 (slow, on-policy)
#   - eval-every 50 per CLAUDE.md
set -euo pipefail
cd /Data/yassine.guennoun/pedantix

PYTHON="${PYTHON:-/users/eleves-b/2022/yassine.guennoun/.conda/envs/py310/bin/python}"
HF_ROOT="models/hf_cache"
export HF_HOME="$HF_ROOT" HF_HUB_CACHE="$HF_ROOT/hub"
export TRANSFORMERS_CACHE="$HF_ROOT/transformers" HF_DATASETS_CACHE="$HF_ROOT/datasets"
export TORCH_HOME="$HF_ROOT/torch" XDG_CACHE_HOME="$HF_ROOT/xdg"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_NAME="${RUN_NAME:-v8_grpo_cot}"
SFT_MODEL="${SFT_MODEL:-models/v8_cot_sft/checkpoint-400}"
PAGES="${PAGES:-data/filtered_pages.jsonl}"
EVAL_CORPUS="${EVAL_CORPUS:-data/clean_pages.jsonl}"
OUT_DIR="models/${RUN_NAME}"
LOG_DIR="logs/${RUN_NAME}"
MAX_STEPS="${MAX_STEPS:-3000}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

if [ ! -d "$SFT_MODEL" ]; then
  echo "ERROR: SFT checkpoint not found at $SFT_MODEL"; exit 1
fi
if [ ! -f "$PAGES" ]; then
  echo "ERROR: pages file not found at $PAGES"; exit 1
fi

echo "[$(date)] Launching v7 trajectory GRPO → $OUT_DIR"
echo "  Base: $SFT_MODEL"
echo "  Pages: $PAGES"

nohup "$PYTHON" -u -m pedantix_project.cli llm-grpo-trajectory \
  --pages "$PAGES" \
  --model "$SFT_MODEL" \
  --tiny-model models/tiny_model.json \
  --output-dir "$OUT_DIR" \
  --log "$OUT_DIR/training_log.jsonl" \
  --plot "$LOG_DIR/rewards.png" \
  --max-steps "$MAX_STEPS" \
  --batch-size 2 \
  --num-generations 4 \
  --max-completion-length 8 \
  --learning-rate 1e-6 \
  --beta 0.05 \
  --pool-size 64 \
  --game-max-steps 30 \
  --temperature 0.9 \
  --top-p 0.9 \
  --lora-rank 32 \
  --solve-bonus-scale 1.0 \
  --top-k-pages 20000 \
  --logging-steps 1 \
  --save-steps 200 \
  --eval-corpus "$EVAL_CORPUS" \
  --eval-every 50 \
  --eval-pages 20 \
  --eval-max-game-steps 30 \
  --eval-num-generations 2 \
  --eval-batch-size 2 \
  --eval-chat-format qwen \
  --chat-format qwen \
  --contrastive-weight 30.0 \
  --seed 42 \
  > "$LOG_DIR/stdout.txt" 2>&1 &

echo "PID: $!  Log: $LOG_DIR/stdout.txt"
echo "Tail: tail -f $LOG_DIR/stdout.txt"
