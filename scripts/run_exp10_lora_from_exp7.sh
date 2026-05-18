#!/usr/bin/env bash
# Exp 10: LoRA backbone + warm-start from exp7 trained action head
#
# Key differences vs exp7:
#   - No --freeze-backbone: fresh LoRA applied to Qwen3-0.6B backbone
#   - --model points to exp7 checkpoint: action head weights remapped by word identity
#   - --freeze-lora-steps 100: phase 1 (steps 1-100) trains action head only,
#     letting it adapt to LoRA-modified hidden states; phase 2 (steps 101+) trains
#     LoRA + action head jointly, allowing backbone representations to improve
#   - --dagger-every 20 / --dagger-bc-steps 25: reduced DAgger (exp9 showed this
#     reduces oscillation without hurting solve_rate)
#   - --bc-warmup-steps 0: action head already trained, no pure-BC warmup needed
#   - --learning-rate 5e-5: conservative for LoRA to avoid disrupting exp7's
#     learned action-head weights
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

RUN_NAME="${RUN_NAME:-exp10_lora_from_exp7}"
MODEL="${MODEL:-models/exp7_dynamic_vocab}"
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
  --lora-rank 16 \
  --freeze-lora-steps 100 \
  --learning-rate 5e-5 \
  --rollout-steps 2 \
  --dagger-every 20 \
  --dagger-pages 16 \
  --dagger-bc-steps 25 \
  --kl-ref-coef 0.01 \
  --min-entropy 3.0 \
  --min-entropy-coef 2.0 \
  --bc-coef 1.0 \
  --bc-warmup-steps 0 \
  --seed 42 \
  2>&1 | tee "${LOG_DIR}/stdout.txt"
