# openpi

## Installation

Install `uv` first:

- https://docs.astral.sh/uv/getting-started/installation/

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## ALOHA Real

### Startup

```bash
# 启动整个本地推理栈，包括 redis、voice_web_backend、voice_web_frontend、runtime、openpi_server 等容器
docker compose up

# voice web 后端服务，负责语音/文本命令、Redis 和机器人状态桥接
docker compose logs -f voice_web_backend

# voice web 前端页面，默认通过浏览器访问 http://localhost:3011
docker compose logs -f voice_web_frontend

# 进入 runtime 容器
docker compose exec -it runtime /bin/bash

# 在 runtime 容器内启动 ALOHA 实机控制主循环；根据 model_dir 自动读取
# checkpoint assets/<asset_id>/norm_stats.json。默认 adapt_to_pi=True，如需关闭请加 --no-adapt-to-p
python3 /app/examples/aloha_real/main.py \  --model-dir /app/checkpoints/eii_data_system_no_tear_cam3_lora/eii_no_tear_cam3_20260422/18000 
  --model-dir /app/checkpoints/20260205/39999
  --model-dir /app/checkpoints/two_direction_lora_from_20260205_39999/two_direction_lora_20260313/6000

python3 /app/examples/aloha_real/main.py \
    --model-dir /app/checkpoints/eii_rinse_cam4_fullft/h200_fullft_13repos_bs128_nw16_f6_s5_log10_20260430_085556/10000 \
    --video-memory-num-frames 6 \
    --video-memory-stride-seconds 5.0
  

# 进入 openpi_server 容器
docker compose exec -it openpi_server /bin/bash

# 在 openpi_server 容器内启动 policy server，负责加载模型并提供推理接口
uv run scripts/serve_policy.py --env ALOHA

# 在宿主机执行机器人复位脚本
uv run scripts/robot_reset_controller.py
```

### Redis Cleanup

```bash
sudo lsof -i :6379
sudo kill <pid>

docker ps | grep redis
docker stop <container_id>
```

### Gripper Data Flow

| Stage | State (gripper) | Action (gripper) | Key code path | Notes |
|---|---|---|---|---|
| Data collection | `FOLLOWER_GRIPPER_JOINT_NORMALIZE_FN(bot.gripper.get_gripper_position())` | `LEADER_GRIPPER_JOINT_NORMALIZE_FN(robot.gripper.get_gripper_position())` | `aloha-2.0/aloha/real_env.py` | `get_gripper_position()` returns joint angle, not linear position. |
| Training preprocess | `_decode_state(...): state[[6, 13]] = _gripper_to_angular(...)` when `adapt_to_pi=True` | `_encode_actions_inv(...): actions[:, [6, 13]] = _gripper_from_angular_inv(...)` when `adapt_to_pi=True` | `src/openpi/policies/aloha_policy.py` | Joint flip + gripper conversion before model input. |
| Inference input | `PUPPET_GRIPPER_POSITION_NORMALIZE_FN(...)` in `get_qpos()` | N/A | `examples/aloha_real/real_env.py` | Runtime state uses normalized position. |
| Inference output | N/A | `_encode_actions(...)` then `PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(...)` in `set_gripper_pose()` | `src/openpi/policies/aloha_policy.py`, `examples/aloha_real/real_env.py` | Model output is converted to robot joint command before publish. |

### Gripper Reference

Follower gripper positions:

- Open: `0.0579 m`
- Closed: `0.0440 m`
- Range: `0.0440 ~ 0.0579 m`

Leader gripper positions:

- Open: `0.0323 m`
- Closed: `0.0185 m`

### Parts Color Map

```text
件                 | 颜色
-------------------|------
爪子               | 白
爪子支架            | 黑
手腕旋转支架        | 白
手腕平移支架        | 白
臂腕连接            | 白
臂旋转支架          | 白
肘臂连接平移支架     | 白/黑
臂臂连接支架        | 白
臂肩连接支架        | 白
底座               | 黑
```

### SSH Notes

```bash
sudo apt install openssh-server
sudo apt install net-tools

ifconfig
systemctl start ssh
systemctl enable ssh
systemctl status ssh
```

## Training

### 1. Convert Data To LeRobot Dataset

Use the conversion script as a template and adjust it for your dataset:

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/libero/data
```

### 2. Compute Normalization Statistics

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

### 3. Run Training

Generic example:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite
```

PI05 ALOHA example:

```bash
WANDB_API_KEY=<your_wandb_key> \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/train.py twist_off_the_bottle_cap_lora \
  --exp_name lora_experiment \
  --overwrite
```

Large-scale example from the reference branch:

```bash
WANDB_API_KEY=<your_wandb_key> \
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

## Utility Commands

### Render Subtasks From HDF5

```bash
uv run scripts/render_subtasks_video_from_hdf5.py \
  --hdf5 tmp/episode_0.hdf5 \
  --output tmp/episode_0_subtasks.mp4
```

### Deterministic Compute Loss From HDF5

```bash
XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python scripts/compute_loss_from_hdf5.py \
  --hdf5 /home/eii/openpi0.5-rtc/tmp/episode_0_twist_one_bottle.hdf5 \
  --config pi05_aloha_pen_uncap \
  --checkpoint /home/eii/openpi0.5-rtc/checkpoints/20260219/1500 \
  --prompt "twist off the bottle cap" \
  --subtask "pick up the bottle" \
  --frame-index 0 \
  --seed 0
```

## Examples

- ALOHA Real: `examples/aloha_real`
- ALOHA Sim: `examples/aloha_sim`


XLA_PYTHON_CLIENT_PREALLOCATE=false   OPENBLAS_NUM_THREADS=4   OMP_NUM_THREADS=4   MKL_NUM_THREADS=4   NUMEXPR_NUM_THREADS=4   ./.venv/bin/python -W ignore scripts/profile_temporal_dataloader_batches.py     --repo-ids '[                                               "lyl472324464/2026-04-21_direction-lerobot-with-rinse",
      "lyl472324464/2026-04-21_direction_2-lerobot-with-rinse",
      "lyl472324464/2026-04-21_direction_havent_cap-lerobot-with-rinse",
      "lyl472324464/2026-04-21_direction_havent_cap_water-lerobot-with-rinse",
      "lyl472324464/2026-04-23_direction_have_cap_water-lerobot-with-rinse",
      "lyl472324464/2026-04-23_direction_havent_cap_water-lerobot-with-rinse",
      "lyl472324464/2026-04-27_direction_have_cap_water2-lerobot-with-rinse",
      "lyl472324464/2026-04-27direction_have_cap_water-lerobot-with-rinse",
      "lyl472324464/2026-04-28_direction_have_cap_water-lerobot-with-rinse",
      "lyl472324464/2026-04-28_direction_have_cap_water2-lerobot-with-rinse",
      "lyl472324464/2026-04-28_water1-lerobot-with-rinse",
      "lyl472324464/2026.03.18_twist-and-water_one_no_cap-with-rinse",
      "lyl472324464/2026.03.30_twist-and-water_two_have_cap-with-rinse"
    ]'     --batch-size 256     --num-workers 32     --prefetch-factor 2     --timed-batches 100     --warmup-batches 1     --video-memory-num-frames 6     --video-memory-stride-seconds 5.0     --assets-base-dir /workspace/openpi0.5-rtc/assets     --checkpoint-base-dir /workspace/openpi0.5-rtc/checkpoints