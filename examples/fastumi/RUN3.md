# Run 3: original + rescued + new November 12 demonstrations

Branch: `train/pi05-fastumi-pick-fan-mixture`.
Config: `pi05_fastumi_pick_fan_relative_mix`.
Experiment: `umi_pi05_run3_relative_pose`.
Preparation only: do not launch until the new upload is complete and the user
asks to start. Run 2 continues in its existing checkout and cache.

## Sources and sampling

The exact source revisions are recorded in `run3_datasets.json`.

| Dataset | Episodes | Frames | Status |
| --- | ---: | ---: | --- |
| `michios/fastumi_pick_fan` | 105 | 9,076 | Pinned and audited |
| `michios/fastumi_pick_fan_1112_overlay_rescue` | 54 | 4,362 | Pinned and audited |
| `michios/fastumi_pick_fan_1112` | Pending | Pending | Upload pending; revision must be pinned |

The first two sources have 159 episodes / 13,438 frames in total. All their
pose/action entries are finite, `action[t] == state[t]`, and no exact pose
trajectory or pose frame in the rescue set duplicates the original set. This
exact-match check does not establish that there are no visually similar or
near-duplicate demonstrations. The third dataset still needs the same checks.

Sample uniformly over all frames across all three datasets, rather than giving
equal weight to datasets of different lengths. Dataset probability is its frame
count divided by the total frame count. No episode is held out for validation.
Recompute normalization over the complete mixture; never reuse run 2 stats.

Current observations keep their single-frame shapes. Action chunks remain
inside their source episode and use offsets +1 through +16; concatenate the
independent datasets after LeRobot resolves the per-episode action chunk.

## Settings carried forward

- Fresh pi05 base weights, full fine-tuning. This is a new run, not a resume
  from run 2; optimizer, normalization and W&B run are separate.
- Fixed prompt: `put the fan in the box`.
- Gripper-only input state; wrist RGB in the left-wrist slot; color jitter only.
- TCP-relative XYZ + rotation6D + absolute gripper opening fraction, horizon
  16 at 20 Hz, next-frame first target. Model action dimension remains padded 32.
- 30,000 steps, batch 32, two GPUs/FSDP=2, seed 42, four data workers.
- AdamW, 500-step warmup, peak LR 2.5e-5, cosine decay to 2.5e-6, EMA 0.999.
- Online W&B under project `openpi`.
- Local full checkpoints every 1,000 steps, approximately 1.3 TB if all kept.
- S3 inference backups at 10k/20k and full final checkpoint 29999, under
  `s3://openpi-tokyo/checkpoints/umi_pi05_run3_relative_pose/`.
- No Hugging Face uploads. HF is only the dataset source.

Final frame counts, dataset sampling proportions, and effective dataset passes
at 30k steps must be updated when the third upload is complete. The larger
mixture does not change the action representation or DROID wrapper contract.

## Preparation

Use the separate run 3 checkout, not the checkout running run 2. Install its
environment normally with `GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --python 3.11`.
Authenticate HF/W&B and configure temporary AWS credentials as for run 2.

```bash
export HF_LEROBOT_HOME="$PWD/.cache/fastumi-run3-lerobot"
export OPENPI_DATA_HOME="$PWD/.cache/openpi"
export CUDA_VISIBLE_DEVICES=0,1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE=online

# Optional now: prepare only the two pinned sources. This reports ready=false.
JAX_PLATFORMS=cpu uv run python examples/fastumi/prepare_mixture.py --allow-pending
```

After the upload is complete, inspect the final repository metadata/file list,
set its immutable SHA in `run3_datasets.json`, and run:

```bash
# Refuses missing revisions or incomplete uploads. Validates schema, counts,
# action alignment, and pose finiteness, and converts local v2.1 copies to v3.
JAX_PLATFORMS=cpu uv run python examples/fastumi/prepare_mixture.py

# Only after all three datasets are ready:
JAX_PLATFORMS=cpu uv run scripts/compute_norm_stats.py \
  --config-name pi05_fastumi_pick_fan_relative_mix
```

Statistics go to
`assets/pi05_fastumi_pick_fan_relative_mix/fastumi_pick_fan_run3/norm_stats.json`.
The preparation report lives at
`$HF_LEROBOT_HOME/umi_pi05_run3_relative_pose.preparation.json` and must show
`ready: true` with all three pinned datasets before training. A partial report
is deliberately not a ready-to-train mixture.

Available sources are already prepared locally under
`/mnt/data/fastumi_run3/lerobot`, including real-image and dataset-boundary smoke
checks. Use that same `HF_LEROBOT_HOME` to continue local preparation.

## Launch only after user approval

In separate persistent terminals, with the prepared environment above:

```bash
uv run scripts/train.py pi05_fastumi_pick_fan_relative_mix \
  --exp-name=umi_pi05_run3_relative_pose --fsdp-devices=2
```

```bash
uv run python examples/fastumi/upload_s3_checkpoints.py \
  --checkpoint-dir=checkpoints/pi05_fastumi_pick_fan_relative_mix/umi_pi05_run3_relative_pose \
  --destination=s3://openpi-tokyo/checkpoints/umi_pi05_run3_relative_pose
```

Use `--policy.config=pi05_fastumi_pick_fan_relative_mix` when serving a run 3
checkpoint. The same `buildup_fastumi_pi05.py` wrapper receives de-normalized
relative actions from the standard server.
