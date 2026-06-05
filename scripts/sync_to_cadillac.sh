#!/usr/bin/env bash
# Sync modified Python source files from harpie to cadillac.
# Run this after editing any .py file before launching jobs on cadillac.
set -euo pipefail
rsync -av \
  /Data/yassine.guennoun/pedantix/pedantix_project/*.py \
  /Data/yassine.guennoun/pedantix/scripts/*.sh \
  cadillac:/Data/yassine.guennoun/pedantix/pedantix_project/ 2>/dev/null || true
rsync -av \
  /Data/yassine.guennoun/pedantix/scripts/*.sh \
  cadillac:/Data/yassine.guennoun/pedantix/scripts/ 2>/dev/null || true
echo "[sync] done"
