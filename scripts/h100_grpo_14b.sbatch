v#!/bin/bash
#SBATCH --job-name=pedantix-grpo
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=0:30:00
#SBATCH --output=/home/yguenn/pedantix/logs/grpo-%j.out
# NOTE: container-image must be passed on the CLI (# is stripped in #SBATCH headers).
# Submit via: sbatch --container-image='nvcr.io#nvidia/pytorch:25.06-py3' scripts/h100_grpo_14b.sbatch

# ── paths ────────────────────────────────────────────────────────────────────
PROJ=/home/yguenn/pedantix
export HF_HOME=$PROJ/models/hf_cache
export HF_HUB_CACHE=$PROJ/models/hf_cache/hub
export TRANSFORMERS_CACHE=$PROJ/models/hf_cache/transformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p $PROJ/logs $PROJ/models/hf_cache

echo "[setup] installing/checking Python deps..."
pip install --quiet --upgrade \
  "transformers>=4.51.0" \
  "peft>=0.14.0" \
  "accelerate>=0.34.0" \
  "torchao>=0.16.0" \
  einops sentencepiece 2>&1 | tail -3

python -c "
import torch
print(f'[gpu] CUDA={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')
print(f'[gpu] VRAM={torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
"

cd $PROJ

RUN_NAME="${RUN_NAME:-v9_h100_14b}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
OUT_DIR="$PROJ/models/$RUN_NAME"
mkdir -p "$OUT_DIR"

echo "[grpo] run=$RUN_NAME model=$MODEL output=$OUT_DIR"
echo "[grpo] $(date)"

python -u -m pedantix_project.cli llm-grpo-trajectory \
  --pages           data/filtered_pages.jsonl \
  --model           "$MODEL" \
  --tiny-model      models/tiny_model.json \
  --output-dir      "$OUT_DIR" \
  --log             "$OUT_DIR/training_log.jsonl" \
  --plot            "$OUT_DIR/reward.png" \
  --max-steps       2000 \
  --batch-size      4 \
  --num-generations 4 \
  --max-completion-length 128 \
  --lora-rank       16 \
  --learning-rate   5e-7 \
  --beta            0.1 \
  --temperature     0.9 \
  --top-p           0.9 \
  --pool-size       32 \
  --game-max-steps  50 \
  --top-k-pages     10000 \
  --min-intro-chars 300 \
  --chat-format     qwen-think \
  --contrastive-weight 0.0 \
  --kl-alarm-threshold 50.0 \
  --kl-lr-cut-factor   0.5 \
  --save-steps      100 \
  --eval-every      50 \
  --eval-corpus     data/filtered_pages.jsonl \
  --eval-pages      10 \
  --eval-max-game-steps 50 \
  --eval-chat-format qwen-think \
  --eval-num-generations 1 \
  --eval-batch-size 1 \
  --logging-steps   1 \
  --seed            42

echo "[grpo] done at $(date)"
