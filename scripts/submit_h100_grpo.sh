#!/usr/bin/env bash
# Wrapper to submit h100_grpo_job.sbatch with the correct container image.
# The --container-image flag must be on the CLI (# is stripped in #SBATCH headers).
set -euo pipefail
cd /home/yguenn/pedantix
sbatch --container-image='nvcr.io#nvidia/pytorch:25.06-py3' scripts/h100_grpo_job.sbatch "$@"
