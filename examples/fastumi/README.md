# FastUMI pick fan: pi05 relative-pose full fine-tuning

Branch: `train/pi05-fastumi-pick-fan-relative`.
Config: `pi05_fastumi_pick_fan_relative`.
Run: `umi_pi05_run2_relative_pose`.

| Setting | Run 2 |
| --- | --- |
| Dataset | Only `michios/fastumi_pick_fan`, revision `fcca932d1000767e442ccbd0a7ca74a063a82766` |
| Training split | All 105 episodes / 9,076 frames; no validation holdout |
| Prompt | Always `put the fan in the box`, overriding dataset task `fan` |
| Initialization | Fresh `gs://openpi-assets/checkpoints/pi05_base/params`; all parameters trainable |
| Observation | One wrist GoPro RGB image + one gripper scalar; no absolute pose, joints, or history |
| Image preprocessing | Aspect-preserving resize/pad to 224x224, left wrist slot; unused cameras masked |
| Augmentation | Color jitter: brightness 0.3, contrast 0.4, saturation 0.5; no geometric crop/rotation for wrist |
| Action | 10 values: TCP-relative XYZ, rotation 6D, absolute gripper opening fraction |
| Chunk timing | Next 16 frames, t+1 through t+16 at 20 Hz (0.8 seconds) |
| Normalization | Fresh run-specific 1st/99th percentile statistics, computed after relative conversion |
| Model dimensions | 1 state and 10 action dimensions padded to 32; pi05 discrete state tokens |
| Optimization | 30,000 steps, global batch 32, AdamW beta1=0.9 / beta2=0.95 / eps=1e-8 / weight decay=1e-10 |
| Learning rate | 500-step warmup to 2.5e-5, cosine decay to 2.5e-6 at 30k |
| Other training settings | Gradient norm clip 1.0, EMA 0.999, seed 42, 4 data workers |
| Hardware plan | 2 GPUs with FSDP=2; bf16 compute, fp32 trainable parameters |
| W&B | Enabled, project `openpi`; loss, gradient/parameter norms, LR and throughput every 100 steps; initial images and prompts |
| Local checkpoints | Every 1,000 steps retained, plus final `29999`; approximately 1.3 TB total |
| S3 backups | Inference at 10000/20000; complete resumable final 29999; approximately 70 GB total |
| S3 destination | `s3://openpi-tokyo/checkpoints/umi_pi05_run2_relative_pose/<step>/` |
| HF uploads | None; Hugging Face is used only to read the dataset |

For each chunk anchored at observation time t:

```text
p_relative[k] = R_t.T @ (p_(t+k) - p_t)
R_relative[k] = R_t.T @ R_(t+k)
k = 1..16
rotation6D = [R00, R10, R20, R01, R11, R21]
```

Source Euler rotations use SciPy extrinsic `xyz`, matching the existing
FastUMI/DROID helper. The source pose is used to form targets, then discarded:
the model sees only the gripper scalar as state. At deployment, recover targets
with `T_target = T_observation_anchor @ T_relative`; all 16 targets share that
anchor. They are not deltas to accumulate frame by frame.

## Dataset review and limitations

The pinned dataset was checked in full. `action[t] == observation.state[t]`
for every frame, so the explicit +1 target offset is necessary. All 10 action
normalization ranges are nondegenerate. Tests verify full rigid-world-frame
invariance and the same rotation column convention used by the DROID client.

- 30k steps at batch 32 is about 106 dataset passes. There is no independent
  validation score; low training loss cannot establish rollout success. Compare
  the saved checkpoints on the robot.
- LeRobot repeats the final pose beyond episode boundaries. These targets are
  included in the loss, as in the existing pi05 pipeline: 9.83% of all chunk
  targets are repeated endpoints. This encourages stopping and weights short
  episodes more strongly. No padding mask was added.
- Demonstrations can move fast: one-frame translation p99 is 5.41 cm and max
  7.45 cm (1.49 m/s at 20 Hz); one-frame rotation max is 17.55 degrees. Chunk
  displacement reaches 75.3 cm over 0.8 seconds. No single-frame translation
  exceeds 10 cm, but this numerical check does not establish tracking quality.
  Keep the controller's motion limits and verify its ability to track the data.
- Observed gripper range is 0.126..1.0. The existing client assumes 1.0 means
  7 cm opening and converts to DROID's reversed 0=open/1=closed convention.
  Verify that calibration on the real gripper; there are no fully-zero targets.
- Gripper-only state omits robot configuration and absolute workspace pose.
  Wrist images must provide task context; matching camera placement and keeping
  the fan/box visible matters. Relative pose does not by itself solve camera or
  end-effector calibration differences.
- Loss includes the padded model dimensions. Use it to monitor optimization,
  not as a physical position/rotation error metric.

## Remote preparation (does not start training)

```bash
git clone --recurse-submodules --branch train/pi05-fastumi-pick-fan-relative \
  git@github.com:EII-Tokyo/openpi0.5-rtc.git openpi-fastumi
cd openpi-fastumi
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e . --no-deps

# Authenticate only if credentials are not already configured.
uv run huggingface-cli login
uv run wandb login
# AWS CLI v2 must have credentials for the destination bucket.
aws sts get-caller-identity

export HF_LEROBOT_HOME="$PWD/.cache/fastumi-lerobot"
export CUDA_VISIBLE_DEVICES=0,1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run python examples/fastumi/prepare_dataset.py
JAX_PLATFORMS=cpu uv run scripts/compute_norm_stats.py \
  --config-name pi05_fastumi_pick_fan_relative
```

Fresh stats go to
`assets/pi05_fastumi_pick_fan_relative/michios/fastumi_pick_fan/norm_stats.json`.
Never reuse run 1 statistics or checkpoint weights for this new representation.
Keep the same `HF_LEROBOT_HOME` in every terminal. Verify GPU collectives and
online W&B access before launching. Preparation is allowed; training awaits
the user's explicit OK.

## Training and S3 backup (only after approval)

In a persistent terminal, from the prepared environment:

```bash
uv run scripts/train.py pi05_fastumi_pick_fan_relative \
  --exp-name=umi_pi05_run2_relative_pose --fsdp-devices=2
```

In another persistent terminal:

```bash
uv run python examples/fastumi/upload_s3_checkpoints.py \
  --checkpoint-dir=checkpoints/pi05_fastumi_pick_fan_relative/umi_pi05_run2_relative_pose
```

The S3 watcher selects finalized numeric Orbax directories. It uses S3 upload
checksums, verifies object sizes and stored SHA256 metadata, writes a per-step
manifest last, retries failures, and records completed steps locally. Keep AWS
credentials valid through the final upload. It also backs up `wandb_id.txt` at
the run prefix for resuming the same W&B run. Training completion alone is not
confirmation that backup succeeded.

To resume training after preparing the same dataset and stats, restore the full
final checkpoint into the matching run directory, restore `wandb_id.txt` beside
its numeric step directory, and append `--resume` to the training command.
The 10k and 20k S3 copies contain inference parameters and assets only.

## Serving and DROID Docker

After downloading a run 2 checkpoint, run the policy server on the host:

```bash
uv run scripts/serve_policy.py --port=8000 policy:checkpoint \
  --policy.config=pi05_fastumi_pick_fan_relative \
  --policy.dir=/mnt/data/checkpoints/pi05_fastumi_pick_fan_relative/umi_pi05_run2_relative_pose/29999
```

The existing `buildup_fastumi.py` client uses the required wrist-image/gripper
request and relative 10D output format. When ready for a physical rollout:

```bash
docker exec -it droid-setup-1pc-robot_control_client-1 \
  /root/miniconda3/bin/conda run --no-capture-output -n robot \
  python /app/droid/scripts/buildup_fastumi.py \
  --host=127.0.0.1 --port=8000 --camera-device=0 \
  --open-loop-horizon=8 --translation-lead-gain=1.0 --gripper-lead=0
```

This executes eight actions before replanning, with RTC disabled. Gain 1 and
gripper lead 0 avoid the older client's default 2x translation extrapolation
and six-frame gripper anticipation. Its existing motion limits remain active.
No physical rollout is run by the setup procedure.

Run 1 remains accessible under config `pi05_fastumi_pick_fan_full`; its 7D
absolute output and normalization are incompatible with this relative client.
Its complete final backup is at `s3://openpi-tokyo/checkpoints/umi_pi05_run1/`.
