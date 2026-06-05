#!/usr/bin/env bash
# Generate SFT trajectories via DigitalOcean serverless inference.
#
# Uses llama3.3-70b-instruct (free tier serverless, OpenAI-compatible API).
# Produces a diverse dataset:
#   - 40% full games from scratch       (warm_start=0) — early-game topic discovery
#   - 40% mid-game starts at step 10    (warm_start=10) — topic narrowed, title zone
#   - 20% mid-game starts at step 20    (warm_start=20) — late-game title guessing
#
# Usage:
#   DO_INFERENCE_KEY=dop_v1_... bash scripts/run_do_traj_gen.sh
#   or: API_KEY=dop_v1_... N_PAGES=5000 bash scripts/run_do_traj_gen.sh
set -euo pipefail

cd /Data/yassine.guennoun/pedantix

PYTHON="${PYTHON:-/users/eleves-b/2022/yassine.guennoun/.conda/envs/py310/bin/python}"
N_PAGES="${N_PAGES:-1000}"
MAX_STEPS="${MAX_STEPS:-30}"
DO_MODEL="${DO_MODEL:-llama3.3-70b-instruct}"
OUTPUT="${OUTPUT:-data/do_trajectories_${N_PAGES}.jsonl}"
SEED="${SEED:-42}"
WARM_SCHEDULE="${WARM_SCHEDULE:-5,10,10,15,20}"

API_KEY="${API_KEY:-${DO_INFERENCE_KEY:-}}"
if [ -z "$API_KEY" ]; then
  echo "ERROR: set DO_INFERENCE_KEY or API_KEY"
  exit 1
fi

mkdir -p data

"$PYTHON" -m pedantix_project.cli do-traj-gen \
  --pages data/clean_pages.jsonl \
  --tiny-model models/tiny_model.json \
  --n-pages "$N_PAGES" \
  --max-steps "$MAX_STEPS" \
  --do-model "$DO_MODEL" \
  --output "$OUTPUT" \
  --seed "$SEED" \
  --warm-start-schedule "$WARM_SCHEDULE" \
  --api-key "$API_KEY"

echo "Done. Trajectories: $OUTPUT"
wc -l "$OUTPUT"
