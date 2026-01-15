# openpi

## ALOHA Real (Local Notes)

Commands:

```bash
docker compose up
docker compose exec -it voice_assistant /bin/bash
uv run voice_assistant.py

docker compose exec -it runtime /bin/bash
python3 /app/examples/aloha_real/main.py --norm-stats-path /app/checkpoints/20260108/13000/assets/trossen/norm_stats.json
docker compose exec -it openpi_server /bin/bash
uv run scripts/serve_policy.py --env ALOHA

uv run scripts/robot_reset_controller.py
```

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
