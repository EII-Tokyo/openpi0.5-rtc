# RLT Offline Key Region Workflow

This workflow keeps robot collection and robot testing on `192.168.1.103`, while local review, key-region cropping, rescoring, and actor/critic training run on this workstation.

## 1. Pull key-region data from 103

Dry-run first:

```bash
.venv/bin/python scripts/rlt_offline_transfer.py pull \
  --local-root local_rlt_data
```

Execute:

```bash
.venv/bin/python scripts/rlt_offline_transfer.py pull \
  --local-root local_rlt_data \
  --execute
```

This creates:

```text
local_rlt_data/raw_from_103/rollouts/key_regions/
local_rlt_data/raw_from_103/replay/rlt_key_regions/
local_rlt_data/raw_from_103/state/
```

## 2. Write local edits

Create an edits file with one JSON object per key region:

```jsonl
{"key_region_id":"abc123","start_sec":1.2,"end_sec":4.0,"reward":1}
{"key_region_id":"def456","start_sec":0.8,"end_sec":3.1,"reward":0}
{"key_region_id":"bad789","voided":true}
```

Cropping should start at the first frame where fine alignment starts to determine success or failure, and should end about `0.3-0.8s` after stable success or clear failure.

## 3. Build the edited replay dataset

```bash
.venv/bin/python scripts/rlt_prepare_offline_dataset.py \
  --raw-root local_rlt_data/raw_from_103 \
  --output-root local_rlt_data/edited \
  --edits-path local_rlt_data/edits.jsonl
```

The script rewrites:

- cropped `.npz` replay shards
- terminal `reward_seq`
- terminal `done`
- edited rollout `manifest.json`
- replay `manifest.jsonl`

The crop uses replay start-frame mapping when `num_frames`, `train_horizon`, and `chunk_stride` are available. This keeps video time and replay transitions aligned.

## 4. Train actor/critic locally

```bash
.venv/bin/python scripts/train_rlt_offline.py \
  --replay-dir local_rlt_data/edited/replay/rlt_key_regions \
  --recursive-scan true \
  --output-dir local_rlt_runs/rinse_insert_offline_v1 \
  --num-train-steps 10000 \
  --batch-size 64 \
  --critic-burn-in-steps 5000 \
  --beta 12 \
  --wandb-project openpi-rlt-offline \
  --wandb-run-name rinse_insert_offline_v1 \
  --overwrite
```

The output contains:

```text
local_rlt_runs/rinse_insert_offline_v1/inference_actor/LATEST
local_rlt_runs/rinse_insert_offline_v1/inference_actor/<step>/actor.msgpack
local_rlt_runs/rinse_insert_offline_v1/inference_actor/<step>/critic.msgpack
local_rlt_runs/rinse_insert_offline_v1/checkpoints/LATEST
local_rlt_runs/rinse_insert_offline_v1/training_summary.json
```

## 5. Deploy the trained actor/critic to 103

Read the local `LATEST` file:

```bash
cat local_rlt_runs/rinse_insert_offline_v1/inference_actor/LATEST
```

Dry-run deployment:

```bash
.venv/bin/python scripts/rlt_offline_transfer.py deploy \
  --local-checkpoint local_rlt_runs/rinse_insert_offline_v1/inference_actor/00010000 \
  --remote-checkpoint-dir /data/openpi0.5-rtc-reward-learning/rlt_offline_checkpoints/rinse_insert_offline_v1
```

Execute:

```bash
.venv/bin/python scripts/rlt_offline_transfer.py deploy \
  --local-checkpoint local_rlt_runs/rinse_insert_offline_v1/inference_actor/00010000 \
  --remote-checkpoint-dir /data/openpi0.5-rtc-reward-learning/rlt_offline_checkpoints/rinse_insert_offline_v1 \
  --execute
```

This script only copies files. It does not start, stop, or restart robot containers.
