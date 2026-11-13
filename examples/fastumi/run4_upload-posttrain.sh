#!/usr/bin/env bash
set -euo pipefail
source /home/ubuntu/fastumi-run4-env.sh
trap 'code=$?; echo "$code" > /home/ubuntu/fastumi-run4-upload-posttrain.exit' EXIT
.venv/bin/python examples/fastumi/upload_s3_checkpoints.py \
 --checkpoint-dir=checkpoints/pi05_fastumi_posttrain/umi_pi05_run4_posttrain \
 --destination=s3://openpi-tokyo/checkpoints/umi_pi05_run4_posttrain \
 --interval=25000 --final-step=99999
