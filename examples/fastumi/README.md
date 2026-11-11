# pi05 full fine-tuning on FastUMI pick fan

Config: `pi05_fastumi_pick_fan_full`. Dataset: only
[`michios/fastumi_pick_fan`](https://huggingface.co/datasets/michios/fastumi_pick_fan).
Initial weights: `gs://openpi-assets/checkpoints/pi05_base/params`.
Both language/vision and action models are dense and all parameters are trainable.

The dataset inspected on 2026-11-11 has 105 episodes, 9,076 frames, 20 Hz,
one `observation.images.front` camera, and seven-dimensional XYZ/RPY/gripper
state/actions. The task text is `fan`; it is read directly from each sample's
`task` field. Only XYZ actions are made relative to the current state; Euler
angles and gripper remain absolute. Missing wrist cameras are zeroed and masked.
State/actions are padded to 32 dimensions for pi05. Inference returns seven
absolute pose/gripper values, with XYZ restored using the input state.
The stored trajectory alignment is retained (the inspected episode has
`action[t] == state[t]`, so the first predicted target is the current pose).

Defaults: global batch 32, 16-frame horizon (0.8 seconds at 20 Hz), 10,000
steps, learning rate 2.5e-5 with 500 warmup steps and cosine decay to 2.5e-6,
EMA 0.999. This is about 35 passes through the data; evaluate intermediate
checkpoints for overfitting. Checkpoints are saved and retained every 1,000
steps, plus the final step (`9999`). Allow substantial disk space for full
model/optimizer checkpoints; adjust `--keep-period` if retention is too costly.

## Remote GPU commands

On the remote Linux machine, use an 80 GB GPU as the single-GPU starting point.
The repository estimates more than 70 GB for full fine-tuning; actual usage
depends on batch size and hardware. Use a persistent terminal such as tmux.
With `uv`, Git, FFmpeg (including AV1 decoding), and GPU drivers installed:

```bash
git clone --recurse-submodules --branch train/pi05-fastumi-pick-fan-full \
  git@github.com:EII-Tokyo/openpi0.5-rtc.git openpi-fastumi
cd openpi-fastumi
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e . --no-deps

# Login using an account that can read the dataset (skip if already logged in).
uv run huggingface-cli login

# Keep the converted dataset separate from any other LeRobot cache.
export HF_LEROBOT_HOME="$PWD/.cache/fastumi-lerobot"
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run python -c 'import jax; print(jax.devices()); assert jax.devices()[0].platform == "gpu"'

# Download main and convert the local v2.1 copy to v3.0. No Hub upload.
uv run python examples/fastumi/prepare_dataset.py

# Compute fresh quantile statistics from this dataset and these transforms.
JAX_PLATFORMS=cpu uv run scripts/compute_norm_stats.py \
  --config-name pi05_fastumi_pick_fan_full

# Base weights are downloaded automatically on the first run.
uv run scripts/train.py pi05_fastumi_pick_fan_full \
  --exp-name=fan_full_s42 --batch-size=32 --fsdp-devices=1 \
  --no-wandb-enabled
```

Normalization is written to
`assets/pi05_fastumi_pick_fan_full/michios/fastumi_pick_fan/norm_stats.json`.
The converter keeps the original v2.1 files in a sibling `fastumi_pick_fan_old`
directory. Rerunning preparation reuses an already converted v3.0 dataset.
It refuses to overwrite previous conversion directories after an interrupted
conversion. Keep the same `HF_LEROBOT_HOME` for preparation, statistics, and
training, including after reconnecting to the machine.

To use W&B, run `uv run wandb login` and omit `--no-wandb-enabled` from both
the initial and resumed commands. To resume with the same settings:

```bash
uv run scripts/train.py pi05_fastumi_pick_fan_full \
  --exp-name=fan_full_s42 --batch-size=32 --fsdp-devices=1 \
  --no-wandb-enabled --resume
```

For a new run on multiple GPUs in one machine, expose the desired GPUs and
set `--fsdp-devices` to their count (for example `CUDA_VISIBLE_DEVICES=0,1`
and `--fsdp-devices=2`). Batch size is global and must be divisible by the
number of visible GPUs. If memory is tight, lower the batch size; that also
reduces the number of data passes at the same step count. Do not use
`torchrun`: this recipe uses the JAX trainer.

Checkpoints are under `checkpoints/pi05_fastumi_pick_fan_full/fan_full_s42/`.
Serve the final checkpoint with:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_fastumi_pick_fan_full \
  --policy.dir=checkpoints/pi05_fastumi_pick_fan_full/fan_full_s42/9999
```

Inference requests need `observation/image` (RGB), `observation/state`
(seven XYZ/RPY/gripper values in the dataset's convention), and `prompt`
(use `fan` to match training). These are pose outputs, not DROID joint velocities.
