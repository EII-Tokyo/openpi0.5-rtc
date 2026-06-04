#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-.venv-eii/bin/python}"
CONFIG="${CONFIG:-debug_training_time_rtc_lora}"
STEPS="${STEPS:-1}"

exec "$PYTHON_BIN" scripts/train.py "$CONFIG" \
  --num-train-steps "$STEPS" \
  --overwrite \
  --no-wandb-enabled
