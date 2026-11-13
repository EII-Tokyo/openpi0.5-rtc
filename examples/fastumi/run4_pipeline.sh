#!/usr/bin/env bash
set -euo pipefail
source /home/ubuntu/fastumi-run4-env.sh
trap 'code=$?; echo "$code" > /home/ubuntu/fastumi-run4-pipeline.exit' EXIT
while [[ ! -f /home/ubuntu/fastumi-run4-prepare.exit ]]; do sleep 30; done
[[ $(cat /home/ubuntu/fastumi-run4-prepare.exit) == 0 ]]
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q \
  examples/fastumi/mid_post_loader_test.py examples/fastumi/two_phase_test.py \
  examples/fastumi/mixture_test.py src/openpi/policies/fastumi_policy_test.py \
  examples/fastumi/upload_s3_checkpoints_test.py
JAX_PLATFORMS=cpu .venv/bin/python examples/fastumi/compute_mid_post_norm.py --root="$FASTUMI_TWO_PHASE_ROOT"
JAX_PLATFORMS=cpu .venv/bin/python examples/fastumi/verify_mid_post_ready.py --root="$FASTUMI_TWO_PHASE_ROOT"
.venv/bin/python - <<'PY'
import jax,wandb
assert len(jax.devices())==2
assert wandb.Api().viewer is not None
print('Two GPUs and W&B authentication verified', flush=True)
PY
aws s3api list-objects-v2 --bucket openpi-tokyo --prefix checkpoints/umi_pi05_run4_midtrain/ --max-keys 1 --no-cli-pager > /dev/null
for phase in midtrain posttrain; do
  tmux new-session -d -s "fastumi-run4-upload-$phase" "bash /home/ubuntu/fastumi-run4-upload-$phase.sh > /home/ubuntu/fastumi-run4-upload-$phase.log 2>&1"
done
.venv/bin/python scripts/train.py pi05_fastumi_midtrain \
  --exp-name=umi_pi05_run4_midtrain --fsdp-devices=2 2>&1 | tee /home/ubuntu/fastumi-run4-midtrain.log
printf '0\n' > /home/ubuntu/fastumi-run4-midtrain.exit
# WeightLoader loads the finalized midtraining EMA weights; posttraining initializes fresh optimizer state.
[[ -d checkpoints/pi05_fastumi_midtrain/umi_pi05_run4_midtrain/199999/params ]]
.venv/bin/python scripts/train.py pi05_fastumi_posttrain \
  --exp-name=umi_pi05_run4_posttrain --fsdp-devices=2 2>&1 | tee /home/ubuntu/fastumi-run4-posttrain.log
printf '0\n' > /home/ubuntu/fastumi-run4-posttrain.exit
for phase in midtrain posttrain; do
  while [[ ! -f "/home/ubuntu/fastumi-run4-upload-$phase.exit" ]]; do sleep 30; done
  [[ $(cat "/home/ubuntu/fastumi-run4-upload-$phase.exit") == 0 ]]
done
printf '%s\n' 'BOTH PHASES AND VERIFIED S3 BACKUPS COMPLETE'
