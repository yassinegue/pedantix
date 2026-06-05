#!/usr/bin/env bash
# v6 GRPO: from oracle-SFT checkpoint.
# Key improvements over v5:
#   - Start from v6_oracle_sft (state-conditional, not template-memorizing)
#   - No DAgger (v7 experiments showed it fights GRPO)
#   - Higher KL beta (0.1) to stay close to oracle-SFT distribution
#   - Near-solve shaping already in reward function (coef=0.3)
#   - eval-every 50 per CLAUDE.md rules
set -euo pipefail
cd /Data/yassine.guennoun/pedantix

PYTHON="${PYTHON:-/users/eleves-b/2022/yassine.guennoun/.conda/envs/py310/bin/python}"
HF_ROOT="models/hf_cache"
export HF_HOME="$HF_ROOT" HF_HUB_CACHE="$HF_ROOT/hub"
export TRANSFORMERS_CACHE="$HF_ROOT/transformers" HF_DATASETS_CACHE="$HF_ROOT/datasets"
export TORCH_HOME="$HF_ROOT/torch" XDG_CACHE_HOME="$HF_ROOT/xdg"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_NAME="${RUN_NAME:-v6_grpo}"
SFT_MODEL="${SFT_MODEL:-models/v7_oracle_sft}"
TRAIN_JSONL="${TRAIN_JSONL:-data/grpo_ready_20260525_014855.jsonl}"
EVAL_CORPUS="${EVAL_CORPUS:-data/clean_pages.jsonl}"
OUT_DIR="models/${RUN_NAME}"
LOG_DIR="logs/${RUN_NAME}"
MAX_STEPS="${MAX_STEPS:-1500}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

if [ ! -d "$SFT_MODEL" ]; then
  echo "ERROR: SFT checkpoint not found at $SFT_MODEL"; exit 1
fi

echo "[$(date)] Launching v6 GRPO → $OUT_DIR"
echo "  SFT base: $SFT_MODEL"

nohup "$PYTHON" -u -m pedantix_project.cli llm-grpo \
  --train "$TRAIN_JSONL" \
  --model "$SFT_MODEL" \
  --tiny-model models/tiny_model.json \
  --output-dir "$OUT_DIR" \
  --log "${OUT_DIR}_training.jsonl" \
  --plot "${LOG_DIR}/rewards.png" \
  --max-steps "$MAX_STEPS" \
  --batch-size 4 \
  --gradient-accumulation-steps 2 \
  --num-generations 4 \
  --max-completion-length 8 \
  --learning-rate 3e-6 \
  --lora-rank 32 \
  --temperature 0.9 \
  --top-p 0.9 \
  --beta 0.1 \
  --solve-bonus-scale 0.05 \
  --logging-steps 1 \
  --save-steps 100 \
  --eval-corpus "$EVAL_CORPUS" \
  --eval-every 100 \
  --eval-pages 20 \
  --eval-max-game-steps 30 \
  --eval-num-generations 2 \
  --eval-batch-size 2 \
  --eval-chat-format qwen \
  --seed 42 \
  --constrained \
  --no-stop-on-garbage \
  > "$LOG_DIR/stdout.txt" 2>&1 &

echo "PID: $!  Log: $LOG_DIR/stdout.txt"
echo "Tail: tail -f $LOG_DIR/stdout.txt"
