# openpi

## ALOHA Real (Local Notes)

Commands:

```bash
# 启动整个本地推理栈，包括 redis、voice_assistant、runtime、openpi_server 等容器
docker compose up

# 进入语音助手容器，用于手动启动语音识别与任务下发进程
docker compose exec -it voice_assistant /bin/bash
# 在 voice_assistant 容器内启动语音助手，负责把语音解析成任务并写入 Redis
# 方式1: 直接使用虚拟环境中的 Python（推荐，最快）
/.venv/bin/python3 voice_assistant.py

# 在 voice_assistant 容器内启动语音助手；和上面作用相同，只是通过 uv 管理环境
# 方式2: 使用 uv run（会自动检查依赖同步，可能较慢）
uv run voice_assistant.py

# 进入 runtime 容器，用于手动启动机器人侧主循环
docker compose exec -it runtime /bin/bash
# 在 runtime 容器内启动 ALOHA 实机控制主程序，负责读机器人观测并请求策略输出动作
python3 /app/examples/aloha_real/main.py --norm-stats-path /app/checkpoints/20260108/13000/assets/trossen/norm_stats.json

# 进入 openpi_server 容器，用于手动启动策略推理服务
docker compose exec -it openpi_server /bin/bash
# 在 openpi_server 容器内启动 policy server，负责加载模型并对 runtime 提供推理接口
uv run scripts/serve_policy.py --env ALOHA

# 在宿主机执行机器人复位控制脚本，用于把机器人回到初始位姿
uv run scripts/robot_reset_controller.py
```

清理redis：

sudo lsof -i :6379
sudo kill 2394

docker ps | grep redis
docker stop 7e


Follower gripper positions:
- Open: 0.0579 m
- Closed: 0.0440 m
- Range: 0.0440 ~ 0.0579 m

Leader gripper positions:
- Open: 0.0323 m
- Closed: 0.0185 m

Results:
- 15,000 steps: success 2, failure 3
- 10,000 steps: success 4, failure 6
- 5,000 steps: success 3, failure 7fdskglsdk;

Parts color map:

```
件              |1  |
——————————————————————————————————————————————————————————————
爪子            |白  |
爪子支架         |黑|
手腕旋转支架     |白|
手腕平移支架     |白|
臂腕连接         |白|
臂旋转支架       |白|
肘臂连接平移支架  |白/黑|
臂臂连接支架     |白|
臂肩连接支架     |白|
底座            |黑|
——————————————————————————————————————————————
```

SSH notes:

```bash
sudo apt install openssh-server
ifconfig
sudo apt install net-tools
ifconfig
systemctl start ssh
systemctl enable ssh
sudo systemctl enable ssh
systemctl stats ssh
systemctl status ssh
sudo systemctl start ssh
sudo systemctl start sshd
systemctl status ssh
ifconfig
history
```

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

### ALOHA examples

- ALOHA Real: `examples/aloha_real`
- ALOHA Sim: `examples/aloha_sim`
