# openpi

## Installation

Install `uv` first: https://docs.astral.sh/uv/getting-started/installation/

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## ALOHA Real (Local Notes)

Commands:

```bash
# Start services
docker compose up

# Voice assistant
docker compose exec -it voice_assistant /bin/bash
uv run voice_assistant.py

# Runtime
docker compose exec -it runtime /bin/bash
python3 /app/examples/aloha_real/main.py --norm-stats-path /app/checkpoints/20260108/13000/assets/trossen/norm_stats.json

# Policy server
docker compose exec -it openpi_server /bin/bash
uv run scripts/serve_policy.py --env ALOHA

# Robot reset
uv run scripts/robot_reset_controller.py
```

### Gripper Data Flow

| Stage | State (gripper) | Action (gripper) | Key code path | Notes |
|---|---|---|---|---|
| Data collection | `FOLLOWER_GRIPPER_JOINT_NORMALIZE_FN(bot.gripper.get_gripper_position())` | `LEADER_GRIPPER_JOINT_NORMALIZE_FN(robot.gripper.get_gripper_position())` | `aloha-2.0/aloha/real_env.py` | `get_gripper_position()` returns joint angle (not linear position). |
| Training preprocess | `_decode_state(...): state[[6, 13]] = _gripper_to_angular(...)` when `adapt_to_pi=True` | `_encode_actions_inv(...): actions[:, [6, 13]] = _gripper_from_angular_inv(...)` when `adapt_to_pi=True` | `src/openpi/policies/aloha_policy.py` | Joint flip + gripper-space conversion are applied before model consumption. |
| Inference input | `PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6]/right_qpos_raw[6])` in `get_qpos()` | N/A | `examples/aloha_real/real_env.py` | Runtime state uses normalized position. |
| Inference output | N/A | `_encode_actions(...): actions[:, [6, 13]] = _gripper_from_angular(...)` then `PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(...)` in `set_gripper_pose()` | `src/openpi/policies/aloha_policy.py`, `examples/aloha_real/real_env.py` | Model output is converted to robot joint command before publish. |
| Gripper constants | Follower joint open/close: `1.6214 / 0.6197` | Leader joint open/close: `0.8298 / -0.0552` | `aloha-2.0/aloha/real_env.py` | These min/max values are critical for normalization consistency. |

## Training

### 1) Convert data to LeRobot dataset

Use the conversion script as a template (adjust for your dataset):

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/libero/data
```

### 2) Compute normalization statistics

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

### 3) Run training

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite
```

PI05 ALOHA subtask training example:

```bash
WANDB_API_KEY=d17182c8002546fcbd4f16cfb2977c71f2553df5 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run scripts/train.py pi05_aloha_pen_uncap \
    --exp-name pi05_subtask_twist_v1 \
    --batch-size 64 \
    --fsdp-devices 8 \
    --num-workers 16 \
    --num-train-steps 40000 \
    --save-interval 500 \
    --log-interval 10
```

### ALOHA examples

- ALOHA Real: `examples/aloha_real`
- ALOHA Sim: `examples/aloha_sim`

## Utility Commands

### Render subtasks from HDF5

```bash
uv run scripts/render_subtasks_video_from_hdf5.py --hdf5 tmp/episode_0.hdf5 --output tmp/episode_0_subtasks.mp4
```

### Deterministic compute loss from HDF5

```bash
XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python scripts/compute_loss_from_hdf5.py \
    --hdf5 /home/eii/learn/openpi0.5-rtc/tmp/episode_0_twist_one_bottle.hdf5 \
    --config pi05_aloha_pen_uncap \
    --checkpoint /home/eii/learn/openpi0.5-rtc/checkpoints/20260219/1500 \
    --prompt "twist off the bottle cap" \
    --subtask "pick up the bottle" \
    --frame-index 0 \
    --seed 0
```
