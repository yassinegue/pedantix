#!/usr/bin/env bash
# Exp 11: Qwen3-4B frozen backbone (7× larger than exp7's 0.6B)
#
# Why: exp10 (LoRA on 0.6B) confirmed LoRA B-matrices did update (norm 0.13-0.53,
# grad_norm +20% in phase 2) but solve_rate never improved beyond exp7's 0.042 avg.
# The bottleneck is NOT backbone capacity per se — it's that GRPO's sparse signal
# (~0.045 solve rate) can't guide 196 LoRA matrices toward anything useful.
#
# Hypothesis: Qwen3-4B's frozen representations are inherently richer than 0.6B's
# for French Wikipedia game states (hidden_size 2560 vs 1536, 36 vs 28 layers,
# 7× total parameters). A better backbone means the linear action head gets
# better-separated hidden states for "game about Napoleon" vs "game about Paris",
# without needing GRPO to guide backbone training.
#
# Key differences vs exp7:
#   - Model: Qwen/Qwen3-4B (frozen, never trained on Pedantix before)
#   - No warm-start: action head shape mismatch (2560 vs 1536 hidden_size)
#   - bc-warmup-steps 100: 100 steps of pure BC to seed the fresh action head
#     before sparse GRPO signal kicks in (avoids cold-start noise)
#   - dagger-every 20 / dagger-bc-steps 25: reduced DAgger from exp9 findings
#   - batch-size 16: same as exp7, should fit in 24GB with frozen 4B
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

RUN_NAME="${RUN_NAME:-exp11_qwen4b_frozen}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
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
  --dynamic-expand-k 20 \
  --freeze-backbone \
  --learning-rate 1e-4 \
  --rollout-steps 2 \
  --dagger-every 20 \
  --dagger-pages 16 \
  --dagger-bc-steps 25 \
  --kl-ref-coef 0.01 \
  --min-entropy 3.0 \
  --min-entropy-coef 2.0 \
  --bc-coef 1.0 \
  --bc-warmup-steps 100 \
  --seed 42 \
  2>&1 | tee "${LOG_DIR}/stdout.txt"
