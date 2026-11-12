# Run 3: original + rescued + new November 12 demonstrations

Branch: `train/pi05-fastumi-pick-fan-mixture`.
Config: `pi05_fastumi_pick_fan_relative_mix`.
Experiment: `umi_pi05_run3_relative_pose`.
All uploads are now pinned and audited. Do not launch until the user confirms
the reviewed configuration. Run 2 continues in its existing checkout and cache.

## Sources and sampling

The exact source revisions are recorded in `run3_datasets.json`.

| Dataset | Episodes | Frames | Sampling share |
| --- | ---: | ---: | ---: |
| `michios/fastumi_pick_fan` | 105 | 9,076 | 27.2% |
| `michios/fastumi_pick_fan_1112_overlay_rescue` | 54 | 4,362 | 13.1% |
| `michios/fastumi_pick_fan_1112` | 244 | 19,930 | 59.7% |

The complete mixture has 403 episodes / 33,368 frames. All pose/action entries
are finite and `action[t] == state[t]`. No exact pose trajectories or pose frames
overlap across the three sources. This exact-match check does not rule out
visually similar or near-duplicate demonstrations. Numerical audit results and
source revisions are saved in `run3_data_audit.json`. Real-image smoke checks
passed at all dataset boundaries.

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

30k steps at batch 32 corresponds to approximately 28.8 dataset passes. Expected
training time on the same two H100s is about 7 hours, plus backup time. The larger
mixture keeps the action representation and DROID wrapper contract. The extra
closing rule (DROID closure >0.5 becomes 1.0) remains deployment postprocessing;
training targets retain measured continuous gripper widths.

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

```

All three source revisions are pinned in `run3_datasets.json`. Prepare and audit
them using the same manifest on every machine:

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

All three sources are prepared locally under `/mnt/data/fastumi_run3/lerobot`,
including real-image and dataset-boundary smoke checks. The isolated remote
checkout is `/home/ubuntu/openpi-fastumi-run3`; its environment script is
`/home/ubuntu/fastumi-run3-env.sh`. Preparation runs without launching training.

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
