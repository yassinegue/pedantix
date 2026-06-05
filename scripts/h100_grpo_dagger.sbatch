#!/bin/bash
#SBATCH --job-name=pedantix-grpo-dagger
#SBATCH --partition=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --output=/home/yguenn/pedantix/logs/grpo-%j.out
# DAgger comparison vs job 585. Same hyperparams; only difference is
# --dagger-rescue-threshold 3 (oracle hint when game has 3 consecutive
# negative-reward turns). Submit with:
#   sbatch --container-image='nvcr.io#nvidia/pytorch:25.06-py3' scripts/h100_grpo_dagger.sbatch

# ── paths ────────────────────────────────────────────────────────────────────
PROJ=/home/yguenn/pedantix
export HF_HOME=$PROJ/models/hf_cache
export HF_HUB_CACHE=$PROJ/models/hf_cache/hub
export TRANSFORMERS_CACHE=$PROJ/models/hf_cache/transformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p $PROJ/logs $PROJ/models/hf_cache

# ── install deps ─────────────────────────────────────────────────────────────
echo "[setup] installing/checking Python deps..."
pip install --quiet --upgrade --break-system-packages \
  "transformers>=4.51.0" \
  "peft>=0.14.0" \
  "accelerate>=0.34.0" \
  "torchao>=0.16.0" \
  einops sentencepiece 2>&1 | tail -3

# ── verify GPU ───────────────────────────────────────────────────────────────
python -c "
import torch
print(f'[gpu] CUDA={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')
print(f'[gpu] VRAM={torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
"

# ── launch trajectory GRPO with DAgger rescue ────────────────────────────────
export PYTHONPATH=$PROJ:${PYTHONPATH}
cd $PROJ 2>/dev/null || echo "[warn] cd $PROJ failed — running from $(pwd), using PYTHONPATH"

RUN_NAME="${RUN_NAME:-v16_grpo_dagger}"
MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
OUT_DIR="$PROJ/models/$RUN_NAME"
mkdir -p "$OUT_DIR" && chown -R yguenn "$OUT_DIR" 2>/dev/null || true

echo "[grpo-dagger] run=$RUN_NAME model=$MODEL output=$OUT_DIR"
echo "[grpo-dagger] $(date)"

python -u -m pedantix_project.cli llm-grpo-trajectory \
  --pages           $PROJ/data/filtered_pages.jsonl \
  --model           "$MODEL" \
  --tiny-model      $PROJ/models/tiny_model.json \
  --output-dir      "$OUT_DIR" \
  --log             "$OUT_DIR/training_log.jsonl" \
  --plot            "$OUT_DIR/reward.png" \
  --max-steps       2000 \
  --batch-size      2 \
  --num-generations 4 \
  --micro-batch-size 1 \
  --max-completion-length 512 \
  --lora-rank       16 \
  --learning-rate   5e-6 \
  --seed-turns      8 \
  --dagger-rescue-threshold 3 \
  --beta            0.1 \
  --temperature     1.3 \
  --top-p           0.9 \
  --pool-size       32 \
  --game-max-steps  50 \
  --top-k-pages     10000 \
  --min-intro-chars 300 \
  --chat-format     qwen \
  --contrastive-weight 0.0 \
  --kl-alarm-threshold 50.0 \
  --kl-lr-cut-factor   0.5 \
  --save-steps      100 \
  --eval-every      9999 \
  --eval-corpus     $PROJ/data/filtered_pages.jsonl \
  --eval-pages      2 \
  --eval-max-game-steps 10 \
  --eval-chat-format qwen \
  --eval-num-generations 1 \
  --eval-batch-size 1 \
  --logging-steps   1 \
  --seed            42

echo "[grpo-dagger] done at $(date)"
