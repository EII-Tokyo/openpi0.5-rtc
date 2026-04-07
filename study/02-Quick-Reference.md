# OpenPI ALOHA 快速参考

## 常用命令速查表

### 训练

```bash
# 基础训练
PYTHONPATH=src uv run scripts/train.py twist_off_the_bottle_cap_subtask_full_finetune \
  --exp-name my_experiment_$(date +%Y%m%d_%H%M%S) \
  --overwrite

# 使用8个GPU + 64 batch size (需要大显存)
PYTHONPATH=src uv run scripts/train.py twist_off_the_bottle_cap_subtask_full_finetune \
  --exp-name my_experiment \
  --batch-size 64 \
  --fsdp-devices 8

# 恢复中断的训练
PYTHONPATH=src uv run scripts/train.py twist_off_the_bottle_cap_subtask_full_finetune \
  --exp-name <existing_exp_name> \
  --resume

# LORA微调（参数少）
PYTHONPATH=src uv run scripts/train.py twist_off_the_bottle_cap_subtask_lora \
  --exp-name my_lora_experiment \
  --batch-size 1 \
  --num-train-steps 20000
```

### Policy Server

```bash
# 启动服务
docker compose up -d openpi_server openpi_server_high_level

# 查看状态
docker compose ps

# 查看日志
docker compose logs -f openpi_server

# 停止服务
docker compose down

# 重启
docker compose restart openpi_server
```

### Runtime 控制

```bash
# 启动实时控制（确保server已启动！）
docker compose exec runtime bash -lc '
  source /opt/ros/noetic/setup.bash &&
  source /root/interbotix_ws/devel/setup.bash &&
  cd /app &&
  python3 examples/aloha_real/main.py \
    --model-dir /app/checkpoints/twist_off_the_bottle_cap_subtask_lora/20260301_120000 \
    --low-level-host 192.168.1.40 \
    --low-level-port 8000 \
    --use-rtc true \
    --policy-hz 50.0
'

# 保存episode为HDF5
docker compose exec runtime bash -lc '
  ... python3 examples/aloha_real/main.py \
    --if-save-hdf5 true \
    --dataset-dir /data/episodes
'
```

### 数据转换

```bash
# ALOHA → LeRobot
PYTHONPATH=src uv run examples/aloha_real/convert_aloha_data_to_lerobot.py \
  --data-dir /path/to/aloha/episodes \
  --output-dir /path/to/lerobot/dataset

# DROID → LeRobot (完整)
PYTHONPATH=src uv run examples/droid/convert_droid_data_to_lerobot.py \
  --data-dir /path/to/droid/data \
  --output-dir /path/to/lerobot/dataset

# DROID 故障 → LeRobot
PYTHONPATH=src uv run examples/droid/convert_droid_failures_to_lerobot.py \
  --failure-path /path/to/failures.txt
```

---

## 常见问题速解

### Q1: "Model loading failed" 或 "无法连接Policy Server"

**症状**：Training starts but crashes on model initialization

**解决**：
```bash
# 1. 检查base model权重是否可访问
gsutil ls gs://openpi-assets/checkpoints/pi05_base/

# 2. 如果离线，从HF下载预训练模型
huggingface-cli download physical-intelligence/pi-base --local-dir ./checkpoints/pi05_base

# 3. 修改config中的weight_loader指向本地
weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/pi05_base/params")
```

### Q2: "CUDA out of memory" 在训练时

**症状**：`torch.cuda.OutOfMemoryError` 或 `RuntimeError: Out of memory!`

**解决**：
```bash
# 降低batch size
--batch-size 32  # 从128降到32

# 减少num_workers
--num-workers 0

# 设置XLA内存限制
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run scripts/train.py ...

# 检查显存使用
nvidia-smi
# 或在代码中
jax.device_get(model).memory_usage()
```

### Q3: "Runtime timeout" 连接不到Policy Server

**症状**：`WebsocketClientPolicy: connection timeout`

**解决**：
```bash
# 1. 检查server是否在运行
docker compose ps | grep openpi_server
# 如果不在运行
docker compose up -d openpi_server

# 2. 检查warmup是否完成
docker compose logs openpi_server | grep "listening"

# 3. 检查网络连接
ping 127.0.0.1  # localhost
ping 192.168.1.40  # 远程server IP

# 4. 给server更多启动时间（有warmup）
sleep 60  # 等待server完全初始化后再启动runtime
```

### Q4: "关节抖动" 或 "动作执行不平滑"

**症状**：机器人在执行动作时频繁改变轨迹

**原因**：RTC引导强度设置不当或推理延迟高

**解决**：
```python
# 增加RTC保留步数（更多旧动作）
guided_inference(
    prev_action=prev_action,
    observation=observation,
    s=30,    # 从20增加到30，更长的连续段
    d=15,    # 从17减少到15，更弱的约束
    beta=4.0  # 从8.0降到4.0，更弱的引导
)

# 或降低推理频率
--high-level-hz 1.0  # 减少子任务解码频率
```

### Q5: "子任务识别不准确"

**症状**：输出的子任务文本与实际动作不符

**原因**：subtask_loss_weight太小或training不足

**解决**：
```python
# 增加子任务权重
pi0_config.Pi0Config(
    subtask_loss_weight=0.5,  # 从0.1增加到0.5
    subtask_max_token_len=250,
)

# 增加训练步数
num_train_steps=100_000  # 从40000增加

# 降低采样温度（更有信心）
policy.sample_subtask_tokens(temperature=0.0)  # 贪心解码
```

### Q6: "数据集转换缓慢"

**症状**：`convert_aloha_data_to_lerobot.py` 运行很慢

**解决**：
```bash
# 使用并行处理 (DROID专用)
PYTHONPATH=src uv run examples/droid/convert_droid_failures_to_lerobot_parallel.py \
  --failure-path /path/to/failures.txt \
  --num-workers 8

# 或手动并行
# 将episodes分成多份，并行处理每份
for dir in episodes/*; do
  python convert_aloha_data_to_lerobot.py --data-dir $dir &
done
wait
```

---

## 配置对比表

### 三个主要训练配置

| 指标 | LORA微调 | 全量微调 | 完整训练 |
|------|---------|---------|---------|
| **配置名** | twist_off_bottle_cap_subtask_lora | twist_off_bottle_cap_subtask_full_finetune | pi05_libero |
| **可训练参数** | 少（LORA） | 全部 | 全部 |
| **Batch Size** | 1 | 128 | 128 |
| **训练步数** | 20,000 | 40,000 | 100,000+ |
| **训练时间** | ~2h (1GPU) | ~8h (8GPU) | ~3d (8GPU) |
| **显存需求** | 8GB | 48GB+ | 48GB+ |
| **数据量** | <100k帧 | 100k-1M帧 | 1M+帧 |
| **最终精度** | 低（仅特定任务） | 中（特定任务优化） | 高（通用） |
| **适用场景** | 快速原型 | 任务优化 | 新robot/任务 |

### 部署配置

**本地推理（低延迟，需要GPU）**:
```python
# 适合实时控制，在runtime同一机器运行
policy = load_policy_jax("checkpoint_path")
for observation in stream:
    actions = policy.sample_actions(observation, num_steps=10)
    execute(actions[0])
```

**远程推理（高效，多robot共享）**:
```python
# Policy Server in Docker，多个runtime连接
# 一个GPU Server支持多个robot
client = WebsocketClientPolicy(host="192.168.1.40", port=8000)
for observation in stream:
    actions = client.infer(observation)
    execute(actions[0])
```

---

## 性能目标

### 推理延迟（从观测到动作）

```
                    目标时间      实际时间
sample_actions()    < 20ms       ~15-18ms (num_steps=10)
                    < 40ms       ~35-40ms (num_steps=20)
guided_inference()  < 20ms       ~15-18ms (RTC引导)
sample_subtask()    < 100ms      ~80-100ms (AR解码)
```

### 吞吐量

```
单GPU：
  Policy Server: ~50 req/s @ batch=1
  并行batch推理: ~200 req/s @ batch=32

多GPU（FSDP）：
  8×GPU: ~400 req/s 或更高
```

### 显存

```
单模型加载：
  PaliGemma (2B) + ActionExpert (300M): ~16GB
  + KV缓存: ~2GB
  + 数据: ~2GB
  总计: ~20GB per GPU (batch=32)

2个模型（low-level + high-level）:
  ~40GB (可用A100 80GB)
```

---

## 数据集对照表

| 数据集 | 大小 | Repo ID | 用途 |
|-------|------|---------|------|
| **LibEvo** | ~100k轨迹 | physical-intelligence/libero | 通用ALOHA训练 |
| **ALOHA Real** | 1-10k轨迹 | 内部 | 真实机器人微调 |
| **DROID Full** | 100k+轨迹 | droid | DROID机器人通用 |
| **DROID Custom** | <100k轨迹 | 内部 | 小数据集DROID微调 |
| **LeRobot Hub** | 多种 | lerobot/xxx | 众源任务数据 |

---

## Model Variant 选择

### 何时用哪个模型

**PI0 (基础)**:
- 低内存环境（<24GB）
- 单任务专项
- 离线推理
- 无子任务需求

**PI05 (推荐)**:
- 有GPU (>24GB)
- 多任务覆盖
- 子任务认知
- 实时控制（RTC）
- 生产部署

**PI0-FAST**:
- 极度内存约束
- DROID机器人
- 轻量部署
- 边缘设备推理

---

## 调试工作流

### 完整调试周期（新任务/机器人）

```
1. 数据准备 (1-2h)
   └─ 采集episode → 转换LeRobot格式 → 验证

2. 配置验证 (30m)
   ├─ 检查norm stats
   ├─ 检查prompt
   └─ 干跑一个epoch

3. 快速微调 (2-4h)
   ├─ LORA微调 (20k步)
   ├─ 离线评估 (compute_loss_from_hdf5)
   └─ 迭代调参

4. 推理测试 (30m)
   ├─ Policy Server启动
   ├─ WebSocket连接测试
   └─ 单step推理性能检测

5. 模拟验证 (1-2h, 可选)
   ├─ PyBullet模拟
   ├─ 动作轨迹检查
   └─ 子任务准确率

6. 真实部署 (1-4h)
   ├─ Runtime启动
   ├─ 人工监督第一个episode
   ├─ 自动化数据采集
   └─ 迭代改进

总耗时: ~1-2天 (新任务)
```

### 快速诊断命令集

```bash
# 0. 系统检查
nvidia-smi
python -c "import jax; print(f'JAX看到{jax.device_count()}个GPU')"

# 1. 数据检查
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('lyl472324464/your_dataset')
print(f'Episodes: {len(ds)}, Sample keys: {ds[0].keys()}')
"

# 2. 模型检查
PYTHONPATH=src python -c "
from openpi.training.config import get_config
cfg = get_config('twist_off_the_bottle_cap_subtask_full_finetune')
print(f'Model type: {cfg.model.model_type}')
print(f'Data config: {cfg.data}')
"

# 3. Server检查
curl http://localhost:8000/health  # 如果server暴露health endpoint

# 4. Performance检查
docker compose logs openpi_server | tail -100 | grep -i "inference_time\|latency"
```

---

## 常用超参数调整

### 加速训练

```python
TrainConfig(
    num_train_steps=10_000,      # 从40000降到10000
    batch_size=256,              # 从128增加到256
    log_interval=1000,           # 从10增加到1000（减少log开销）
    save_interval=10_000,        # 从1000增加（减少I/O）
    num_workers=4,               # 增加数据加载workers
)
```

### 提高精度

```python
TrainConfig(
    num_train_steps=100_000,     # 从40000增加
    batch_size=256,              # 从128增加
    lr_schedule=CosineDecaySchedule(
        warmup_steps=20_000,     # 从10000增加
        peak_lr=5e-5,            # 从2.5e-5增加
        decay_steps=80_000,      # 从40000增加
        decay_lr=5e-6,           # 从2.5e-6增加
    ),
    ema_decay=0.999,             # 从0.99增加（更强的EMA平滑）
)
```

### 节省显存

```python
TrainConfig(
    batch_size=32,               # 从128降到32
    num_workers=0,               # 禁用多进程data loading
    # 或启用gradient checkpointing (如果支持)
    # freeze_filter=更多frozen层
)
```

