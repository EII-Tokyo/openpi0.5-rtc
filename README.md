# openpi

## Installation

Install `uv` first: https://docs.astral.sh/uv/getting-started/installation/

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## ALOHA Real (Local Notes)

### Safety

- `runtime` will talk to the real robot. Do not start it unless someone is physically present to watch the robot.
- It is safe to run offline HDF5 evaluation and policy-server startup checks without `runtime`.

### Voice Web Only

Start only the web stack without any model containers:

```bash
docker compose up -d redis voice_web_backend voice_web_frontend
```

Useful logs:

```bash
docker compose logs -f voice_web_backend
docker compose logs -f voice_web_frontend
```

### Training

Train `twist_off_the_bottle_cap_subtask_lora`:

```bash
PYTHONPATH=src uv run scripts/train.py \
  twist_off_the_bottle_cap_subtask_lora \
  --exp_name twist_off_the_bottle_cap_subtask_lora_$(date +%Y%m%d_%H%M%S) \
  --overwrite
```

Resume an existing run:

```bash
PYTHONPATH=src uv run scripts/train.py \
  twist_off_the_bottle_cap_subtask_lora \
  --exp_name <existing_exp_name> \
  --resume
```

### Policy Servers Only

Start both policy servers on the local machine without `runtime`:

```bash
SERVER_ARGS='--warmup-rtc --warmup-non-rtc --no-warmup-subtask policy:checkpoint --policy.config twist_off_the_bottle_cap_subtask_lora --policy.dir /app/checkpoints/twist_off_the_bottle_cap_subtask_lora/<exp_name>/<step>' \
HIGH_LEVEL_SERVER_ARGS='--no-warmup-rtc --no-warmup-non-rtc --warmup-subtask policy:checkpoint --policy.config twist_off_the_bottle_cap_subtask_lora --policy.dir /app/checkpoints/twist_off_the_bottle_cap_subtask_lora/<exp_name>/<step>' \
docker compose up -d openpi_server openpi_server_high_level
```

Expected ports:

- low-level websocket server: `127.0.0.1:8000`
- high-level websocket server: `127.0.0.1:8001`

This warmup split is intentional:

- low-level server warms up `infer(..., use_rtc=True)` and `infer(..., use_rtc=False)`
- high-level server warms up only `infer_subtask(...)`

### Split Deployment

The current recommended online setup is:

- local machine (`192.168.1.42`): `runtime`, `openpi_server_high_level`, `voice_web_backend`, `voice_web_frontend`
- remote machine (`192.168.1.40`): low-level `openpi_server`

The remote low-level repo is:

```bash
/home/eii/openpi0.5-rtc-lowlevel-run
```

Restart the remote low-level container:

```bash
ssh eii@192.168.1.40
cd /home/eii/openpi0.5-rtc-lowlevel-run
docker restart openpi05-rtc-lowlevel-run-openpi_server-1
docker logs -f openpi05-rtc-lowlevel-run-openpi_server-1
```

Wait for:

- `Creating server (...)`
- `server listening on 0.0.0.0:8000`

before reconnecting `runtime`.

### Runtime

When it is safe to move the robot, `examples/aloha_real/main.py` will now reset the robot to its initial pose on startup again, via `AlohaRealEnvironment.reset() -> RealEnv.reset()`.

Do not run this remotely unless someone is in front of the robot.

If low-level runs on `192.168.1.40`, start `runtime` locally with:

```bash
docker compose exec runtime bash -lc '
source /opt/ros/noetic/setup.bash &&
source /root/interbotix_ws/devel/setup.bash &&
cd /app &&
export PYTHONPATH=/app:/app/src:/app/packages/openpi-client/src:$PYTHONPATH &&
python3 -u examples/aloha_real/main.py \
  --model-dir /app/checkpoints/twist_off_the_bottle_cap_subtask_lora/<exp_name>/<step> \
  --low-level-host 192.168.1.40 \
  --low-level-port 8000 \
  --high-level-host 127.0.0.1 \
  --high-level-port 8001
'
```

Notes:

- `runtime` must be started from a shell that has both `/opt/ros/noetic/setup.bash` and `/root/interbotix_ws/devel/setup.bash` sourced.
- If the remote low-level server is still warming up, `runtime` may connect once and then exit on the first websocket request. In that case, wait for remote `server listening on 0.0.0.0:8000`, then restart `runtime`.
- `runtime` is considered healthy only after it reaches:
  - `Redis连接成功`
  - `Redis监听线程已启动`
  - `Starting episode...`
  - `Runtime 默认仅接受 Redis / voice web 任务`

### Measured VRAM (RTX 5090 32GB, offline HDF5)

- low-level `infer(..., use_rtc=False)`: about `8.8 GiB`
- low-level `infer(..., use_rtc=True)`: about `8.8 GiB`
- high-level `infer_subtask(...)`: about `16.6 GiB`
- low-level + high-level servers resident together: about `26.8 GiB`

Notes:

- Low-level warmup for RTC and non-RTC does not create two copies of the model weights in steady state.
- It can still increase startup compile time and transient peak memory.

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
