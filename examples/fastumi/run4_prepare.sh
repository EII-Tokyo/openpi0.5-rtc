#!/usr/bin/env bash
set -euo pipefail
source /home/ubuntu/fastumi-run4-env.sh
trap 'code=$?; echo "$code" > /home/ubuntu/fastumi-run4-prepare.exit' EXIT
JAX_PLATFORMS=cpu .venv/bin/python examples/fastumi/prepare_mid_post_data.py \
  --root="$FASTUMI_TWO_PHASE_ROOT" \
  --fan-cache=/home/ubuntu/openpi-fastumi-run3/.cache/fastumi-run3-lerobot-verified
