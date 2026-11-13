#!/usr/bin/env bash
set -euo pipefail
source /home/ubuntu/fastumi-run4-env.sh
trap 'code=$?; echo "$code" > /home/ubuntu/fastumi-run4-upload-midtrain.exit' EXIT
.venv/bin/python examples/fastumi/upload_s3_checkpoints.py \
 --checkpoint-dir=checkpoints/pi05_fastumi_midtrain/umi_pi05_run4_midtrain \
 --destination=s3://openpi-tokyo/checkpoints/umi_pi05_run4_midtrain \
 --interval=50000 --final-step=199999
