# OpenPI ALOHA 完整技术指南

> 从数据格式 → 训练流程 → 推理系统 → 机器人控制

## 目录
1. [数据格式](#1-数据格式)
2. [训练流程](#2-训练流程)
3. [推理系统](#3-推理系统)
4. [ALOHA机器人控制](#4-aloha-机器人完整运行)
5. [核心代码位置](#5-核心代码位置)
6. [调试指南](#6-调试指南)

---

## 1. 数据格式

### 1.1 LeRobot 数据集结构

OpenPI 使用 **LeRobot 数据集格式**存储和处理数据：

```
LeRobot Dataset (HuggingFace Hub)
└── Episode
    ├── observations/
    │   ├── images/
    │   │   ├── cam_high        (224×224 RGB uint8)
    │   │   ├── cam_low         (224×224 RGB uint8)
    │   │   ├── cam_left_wrist  (224×224 RGB uint8)
    │   │   └── cam_right_wrist (224×224 RGB uint8)
    │   └── state [14]
    │       ├── [0:6]   = left_arm_joint_angles
    │       ├── [6]     = left_gripper_position
    │       ├── [7:13]  = right_arm_joint_angles
    │       └── [13]    = right_gripper_position
    └── actions [T, 14]          (与state结构相同)
```

### 1.2 ALOHA 坐标转换

**关键概念**：ALOHA 训练和运行时使用不同的关节角度约定

**转换逻辑**（src/openpi/policies/aloha_policy.py）:

```python
# 关节翻转掩码 - 某些关节需要反向
_joint_flip_mask = [1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1]

# 夹爪转换 (线性位置 ↔ 角度)
def _gripper_to_angular(value):
    """LeRobot线性位置 → PI0内部角度格式"""
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)
    # 线性→弧度 (Interbotix物理模型)
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)
    # 归一化到[0,1]
    return _normalize(value, min_val=0.5476, max_val=1.6296)

def _encode_actions(actions, adapt_to_pi=True):
    """模型输出动作 → ALOHA命令"""
    if adapt_to_pi:
        actions = _joint_flip_mask * actions  # 关节翻转
        # actions[:, [6,13]] = _gripper_from_angular(...)  # 夹爪转换
    return actions
```

### 1.3 图像处理

```python
# 预处理流程（train_step 前）
raw_image (uint8, HWC)
    ↓ 如需要则resize → 224×224
    ↓ 如训练则数据增强（crop, rotate, color jitter）
    ↓ 转换 uint8→float32 [-1, 1]
    ↓ 输入模型
```

---

## 2. 训练流程

### 2.1 配置示例：twist_off_the_bottle_cap_subtask_full_finetune

**位置**：`src/openpi/training/config.py:986`

```python
TrainConfig(
    name="twist_off_the_bottle_cap_subtask_full_finetune",
    
    # 模型配置
    model=pi0_config.Pi0Config(
        pi05=True,                        # 使用PI05（带子任务预测能力）
        max_token_len=80,                 # 提示词最多80 token
        subtask_loss_weight=0.1,          # 子任务损失权重（总损失的10%）
        subtask_max_token_len=250,        # 子任务输出最多250 token
        fast_tokenizer_path="physical-intelligence/fast",
    ),
    
    # 数据配置
    data=LeRobotAlohaDataConfig(
        image_size=(224, 224),
        force_prompt="process all bottles",       # 固定提示词
        repo_ids=["lyl472324464/wipe_the_desk_with_a_blue_cloth"],  # LeRobot数据集
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
            asset_id="trossen",  # ALOHA机器人assets
        ),
        base_config=DataConfig(prompt_from_task=True),
        video_memory_num_frames=1,    # 不使用视频堆叠
    ),
    
    # 优化器配置
    lr_schedule=CosineDecaySchedule(
        warmup_steps=10_000,
        peak_lr=2.5e-5,
        decay_steps=40_000,
        decay_lr=2.5e-6,
    ),
    ema_decay=0.99,                   # 指数移动平均
    
    # 训练超参
    num_train_steps=40_000,
    batch_size=128,
    save_interval=1_000,
    log_interval=10,
    num_workers=0,
    
    # 权重加载
    weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
)
```

### 2.2 完整训练流程

```
启动: python scripts/train.py twist_off_the_bottle_cap_subtask_full_finetune
│
├─ Phase 1: 初始化
│  ├─ 创建FSDP分布式网格（8个GPU）
│  ├─ 加载base model权重（从GCS）
│  └─ 初始化优化器和EMA
│
├─ Phase 2: 数据加载循环（scripts/train.py:250-254）
│  ├─ LeRobotDataset 读取HuggingFace数据集
│  ├─ TransformedDataset 应用转换：
│  │  ├─ Repack: 摄像头路径映射 (cam_high → base_0_rgb)
│  │  ├─ Tokenize: 提示词 "process all bottles" → token IDs
│  │  ├─ Normalize: 使用ALOHA asset的norm stats
│  │  ├─ Resize: 图像 → 224×224
│  │  └─ Augment: 随机crop/rotate/color jitter
│  └─ 批处理：[batch=128, img=224×224×3, state=14, actions=50×14]
│
├─ Phase 3: 前向传播（train_step，脚本:152-221）
│  │
│  └─ model.compute_loss_with_metrics(rng, observation, actions, train=True)
│     │
│     ├─ Vision编码 (SigLIP, src/openpi/models/siglip.py)
│     │  ├─ 4张224×224图 → 特征向量 [batch, tokens, 768]
│     │  └─ 处理缺失摄像头（动态4/3摄像头支持）
│     │
│     ├─ 提示词编码 (PaliGemma tokenizer)
│     │  └─ "process all bottles" → [batch, 80, vocab_size]
│     │
│     ├─ 状态编码 (if pi05)
│     │  └─ state [14] → discrete tokens
│     │
│     ├─ Flow Matching 损失 (pi0.py:200-300)
│     │  ├─ 采样时间步 t ~ U(0,1)
│     │  ├─ 添加噪声: x_t = sqrt(1-t)*x_0 + sqrt(t)*noise
│     │  ├─ 模型预测速度: v_pred = model(x_t, t, condition)
│     │  └─ 损失: ||v_pred - v_true||^2
│     │
│     └─ 子任务AR损失 (pi05特有)
│        ├─ 目标: 从观测预测当前/下一步的子任务文本
│        └─ 损失权重: 0.1 (总损失的10%)
│
├─ Phase 4: 反向传播 & 优化（train_step:184-196）
│  ├─ 梯度计算（只更新可训练参数）
│  ├─ Optax优化器更新
│  └─ EMA参数跟踪 (ema_params = 0.99*old + 0.01*new)
│
├─ Phase 5: 日志和检查点（train_step:300-311）
│  ├─ 每10步输出: loss, flow_loss, subtask_ar_loss, grad_norm, param_norm
│  ├─ 每1000步保存checkpoint (JAX Orbax)
│  └─ W&B记录训练曲线
│
└─ 40,000步后完成
   └─ 最终模型保存到 checkpoints/{exp_name}/final
```

### 2.3 数据加载细节

**LeRobotAlohaDataConfig** (config.py:265-337):

```python
@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    image_size: tuple[int, int] = (224, 224)
    force_prompt: str | None = None                    # 覆盖数据集prompt
    include_bottle_description: bool = True
    include_bottle_position: bool = True
    include_bottle_state: bool = True
    include_subtask: bool = True
    video_memory_num_frames: int = 1                   # 多帧堆叠
    video_memory_stride_seconds: float = 1.0           # 帧间隔
    repo_ids: list[str] | None = None                  # HF数据集列表
    
    # 关键: Repack转换 (config.py:1016-1034)
    repack_transforms=Group(
        inputs=[
            InjectDefaultField("subtask", None),
            RepackTransform({
                "images": {
                    "cam_high": "observation.images.cam_high",
                    "cam_low": "observation.images.cam_low",
                    "cam_left_wrist": "observation.images.cam_left_wrist",
                    "cam_right_wrist": "observation.images.cam_right_wrist",
                },
                "state": "observation.state",
                "actions": "action",
                "prompt": "prompt",
                "subtask": "subtask",
            })
        ]
    )
```

---

## 3. 推理系统

### 3.1 子任务识别：sample_subtask_tokens()

**位置**：`src/openpi/models/pi0.py:390-492`

**功能**：根据当前观测（摄像头+动作）预测ALOHA正在执行的子任务

```python
def sample_subtask_tokens(
    rng,
    observation: Observation,      # images, state, tokenized_prompt
    temperature: float = 0.0,       # 0=贪心, >0=采样
    eos_token_id: int | None,      # 停止token
    max_text_token_id: int = 240_000,  # 文本token范围
) -> Int[Array, "b max_steps"]
```

**执行流程**：

```
输入: (4张224×224图像, 14维state, 提示词"process all bottles")
│
├─ Vision编码（SigLIP）
│  └─ 4张图 → 特征向量 [batch, img_tokens, 768]
│
├─ 提示词编码（PaliGemma tokenizer）
│  └─ "process all bottles" → [batch, 80]
│
├─ 前缀前向（充满KV缓存）
│  ├─ 组装: 图像特征 + 提示词
│  ├─ 通过LLM transformer
│  └─ 输出: KV缓存 + 隐藏状态
│
├─ 自回归循环（逐token生成）
│  │
│  ├─ FOR step_i in [0, max_steps):
│  │  ├─ 计算logits: last_hidden @ embedding_table^T
│  │  ├─ 过滤不合法token:
│  │  │  ├─ 特殊token (vocab[-128:])  → -∞
│  │  │  ├─ 非文本token (>240000) → -∞
│  │  │  └─ PAD token (0) → -∞
│  │  ├─ 采样下一token:
│  │  │  ├─ if temperature==0: argmax(logits)
│  │  │  └─ else: categorical(logits/temperature)
│  │  ├─ 检查EOS: 如果token==eos_token_id则停止
│  │  ├─ Token embedding + KV缓存继续
│  │  └─ next_hidden = LLM(token, kv_cache)
│  │
│  └─ 直到EOS或max_steps
│
└─ 输出: 子任务token序列 [batch, t]
   └─ 解码回文本: "turn the bottle cap clockwise"
```

**实现细节**（pi0.py:441-475）：

```python
def step(carry):
    rng, last_hidden, output_tokens, finished, step_i, kv_cache = carry
    
    # 1. 计算logits (内积)
    logits = einsum("bd,vd->bv", last_hidden, embed_table, ...)
    
    # 2. 过滤非法token
    logits = where(special_token_mask[None,:], -inf, logits)
    logits = where(non_text_mask[None,:], -inf, logits)
    logits = logits.at[:, 0].set(-inf)  # PAD
    
    # 3. 采样
    if temperature <= 0.0:
        next_token = argmax(logits, axis=-1)
    else:
        rng, sample_rng = random.split(rng)
        next_token = random.categorical(sample_rng, logits/temperature)
    
    # 4. 检查EOS
    if eos_token_id is not None:
        next_token = where(finished, eos_token_id, next_token)
        finished = or_(finished, next_token == eos_token_id)
    
    # 5. 保存token并继续
    output_tokens = put_along_last_axis(output_tokens, step_i, next_token)
    token_emb = LLM(next_token[:, None], method="embed")
    generated_valid = (arange(max_steps)[None,:] <= step_i)
    key_mask = concatenate([prefix_mask, generated_valid], axis=1)
    (full_out, _), kv_cache = LLM([token_emb, None], mask=key_mask, kv_cache=kv_cache)
    last_hidden = full_out[:, 0, :]
    
    return rng, last_hidden, output_tokens, finished, step_i+1, kv_cache
```

### 3.2 动作预测：sample_actions()

**位置**：`src/openpi/models/pi0.py:322-387`

**功能**：从观测预测下50个时间步的机器人动作

```python
def sample_actions(
    rng: KeyArrayLike,
    observation: Observation,     # 当前摄像头+状态+提示词
    num_steps: int = 10,           # ODE求解步数（扩散步数）
    noise: Array | None = None,    # 初始高斯噪声
) -> Actions  # [batch, action_horizon=50, action_dim=14]
```

**Flow Matching扩散过程**：

```
初始化: x_1 ~ N(0, I) [batch, 50, 14]  (纯噪声)
│
├─ 前缀前向（一次，充满KV缓存）
│  ├─ Vision: 4张224×224图 → 特征
│  ├─ Prompt: "process all bottles" → tokens
│  ├─ 组装前缀序列
│  └─ LLM前向 → KV缓存
│
├─ ODE循环: FOR t in [1, 1-dt, 1-2dt, ..., dt]
│  （dt = -1/num_steps，例如num_steps=10则dt=-0.1）
│  │
│  ├─ 后缀编码: x_t (当前noisy action)
│  │  ├─ 时间步embedding (sine-cosine pos emb)
│  │  └─ x_t → token
│  │
│  ├─ 注意力计算
│  │  ├─ 后缀tokens对前缀的注意 (跨KV缓存)
│  │  └─ 后缀tokens之间的注意 (causal)
│  │
│  ├─ LLM前向（使用KV缓存）
│  │  ├─ 输入: [None, suffix_tokens]
│  │  ├─ KV缓存: 前缀KV
│  │  └─ 输出: suffix_out [batch, suffix_len, 768]
│  │
│  ├─ 动作投影
│  │  ├─ v_t = action_out_proj(suffix_out[:, -50:])
│  │  └─ v_t [batch, 50, 14]  (速度预测)
│  │
│  └─ ODE更新
│     └─ x_t ← x_t - dt * v_t
│        (相当于x_t ← x_t + 0.1*v_t 当dt=-0.1)
│
└─ 输出: x_0 [batch, 50, 14]  (清晰的动作轨迹)
   └─ 归一化到[-1,1]范围
```

**ODE求解循环**（pi0.py:347-387）：

```python
def step(carry):
    x_t, time = carry
    
    # 1. 后缀嵌入
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = embed_suffix(
        observation, x_t, broadcast(time, batch_size)
    )
    
    # 2. 构建注意力掩码
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn_mask = repeat(prefix_mask, "b p -> b s p", s=suffix_len)
    full_attn_mask = concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    
    # 3. LLM前向
    positions = sum(prefix_mask, axis=-1)[:, None] + cumsum(suffix_mask, axis=-1) - 1
    (prefix_out, suffix_out), _ = LLM(
        [None, suffix_tokens],
        mask=full_attn_mask,
        positions=positions,
        kv_cache=kv_cache,
        adarms_cond=[None, adarms_cond],
    )
    
    # 4. 速度预测和更新
    v_t = action_out_proj(suffix_out[:, -50:])  # 取最后50个token
    return x_t + dt*v_t, time + dt

def cond(carry):
    x_t, time = carry
    return time >= -dt/2  # 停止条件

x_0, _ = while_loop(cond, step, (noise, 1.0))
return x_0
```

**扩散步数影响**：
- `num_steps=10`：快速但质量一般，用于实时控制
- `num_steps=50`：高质量但慢，用于离线评估

### 3.3 RTC实时控制：guided_inference()

**位置**：`src/openpi/models/pi0.py:494-550+`

**功能**：使用前一步的动作预测**引导**当前预测，保证轨迹平滑

```python
def guided_inference(
    rng: KeyArrayLike,
    prev_action: Actions,          # 上一步模型输出 [batch, 50, 14]
    observation: Observation,      # 当前观测
    num_steps: int = 10,           # ODE步数
    s: int = 20,                   # 前s步保持旧动作
    d: int = 17,                   # 最后d步用旧动作引导
    beta: float = 8.0,             # 引导强度
) -> Actions  # [batch, 50, 14]
```

**RTC引导策略**：

```
prev_action (上一步输出，50步)
│  ├─ [0:20]   → 直接用（保证连续性）
│  ├─ [20:33]  → 新模型预测（自由探索）
│  └─ [33:50]  → 新模型预测 + 旧动作引导（约束）
│
实际执行:
│  ├─ 当前帧: 执行prev_action[0]
│  ├─ +500ms: 执行prev_action[1:10]（保持旧轨迹前10步）
│  ├─ +500ms: 请求新预测（获取guided_inference结果）
│  ├─ +1000ms: 执行新预测[0:5]（中间10步）
│  └─ +1000ms: 执行新预测[5:20]（最后20步，受旧预测约束）
│
结果: 实时控制频率 ~2Hz (每25步重新计划)
      执行频率 50Hz (逐步执行)
```

**实现**（pi0.py:515-550）：

```python
# 提取prev_action的有用部分
prev_action_slice = prev_action[:, s:, :]  # [batch, 50-s, 14]
zero_actions = zeros((batch_size, s, 14))
prev_action_slice = concatenate([prev_action_slice, zero_actions], axis=1)

# 在ODE循环中使用beta加权引导
# guided_step ∝ (1-beta_t)*x_t_sampled + beta_t*prev_action_slice
```

---

## 4. ALOHA 机器人完整运行

### 4.1 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    实时控制系统架构                          │
└─────────────────────────────────────────────────────────────┘

本地机 (192.168.1.42)
┌──────────────────────────────┐
│   Runtime (main.py)          │  50Hz控制循环
│  ┌──────────────────────────┐│
│  │ AlohaRealEnvironment     ││  读摄像头+获取state
│  │ ├─ RealSense x4          ││
│  │ └─ Interbotix状态        ││
│  └──────────────────────────┘│
│  ┌──────────────────────────┐│
│  │ PolicyAgent              ││  处理动作
│  │ └─ ActionChunkBroker     ││  RTC buffer管理
│  └──────────────────────────┘│
└──────────────────────────────┘
         ↕ WebSocket 8000,8001
┌──────────────────────────────┐
│  Policy Servers (compose)    │
│  ├─ openpi_server            │  低级: infer()
│  │  ├─ sample_actions()      │  50步×14维
│  │  └─ guided_inference()    │  RTC引导
│  │                           │
│  └─ openpi_server_high_level │  高级: infer_subtask()
│     └─ sample_subtask_tokens │  当前子任务
└──────────────────────────────┘

远程机 (192.168.1.40, 可选)
└─ openpi_server (低级, RTC=True)
```

### 4.2 Policy Server启动

**docker-compose.yml** (主要服务):

```yaml
services:
  openpi_server:
    build: .
    image: openpi:latest
    environment:
      CUDA_VISIBLE_DEVICES: "0,1"
    command: >
      python scripts/serve_policy.py
      --config pi05_aloha_pen_uncap
      --checkpoint /app/checkpoints/pi05_base
      --use-rtc
      --warmup-rtc
      --warmup-non-rtc
      --no-warmup-subtask
    ports:
      - "8000:8000"
    volumes:
      - ./checkpoints:/app/checkpoints

  openpi_server_high_level:
    image: openpi:latest
    environment:
      CUDA_VISIBLE_DEVICES: "2,3"
    command: >
      python scripts/serve_policy.py
      --config pi05_aloha_pen_uncap
      --checkpoint /app/checkpoints/pi05_base
      --warmup-subtask
      --no-warmup-rtc
      --no-warmup-non-rtc
    ports:
      - "8001:8001"
    volumes:
      - ./checkpoints:/app/checkpoints
```

**服务器内部** (scripts/serve_policy.py):

```python
while True:
    # WebSocket接收观测
    observation = await websocket.receive_observation()
    
    # 推理
    if use_rtc:
        actions = policy.guided_inference(
            prev_action=prev_actions,
            observation=observation,
            s=20, d=17, beta=8.0
        )
    else:
        actions = policy.sample_actions(observation)
    
    prev_actions = actions
    
    # 可选：推理子任务
    if warmup_subtask:
        subtask_tokens = policy.sample_subtask_tokens(observation)
        subtask_text = tokenizer.decode(subtask_tokens)
    
    # 返回ActionChunk
    await websocket.send_response({
        "actions": actions[:25],  # 执行前25步
        "subtask": subtask_text if warmup_subtask else None
    })
```

### 4.3 Runtime 完整执行流程

**启动命令** (examples/aloha_real/main.py):

```bash
docker compose exec runtime bash -lc '
  source /opt/ros/noetic/setup.bash &&
  source /root/interbotix_ws/devel/setup.bash &&
  cd /app &&
  python3 examples/aloha_real/main.py \
    --model-dir /app/checkpoints/twist_off_the_bottle_cap_subtask_lora/20260301_120000 \
    --low-level-host 192.168.1.40 \
    --low-level-port 8000 \
    --high-level-host 127.0.0.1 \
    --high-level-port 8001 \
    --use-rtc true \
    --policy-hz 50.0 \
    --action-horizon 50 \
    --adapt-to-pi true \
    --if-save-hdf5 true \
    --dataset-dir /data/episodes
'
```

**Runtime执行流程**：

```
初始化 (main.py:56-100)
├─ 连接Policy Servers (WebSocket)
│  ├─ low_level_policy  @ 192.168.1.40:8000
│  └─ high_level_policy @ 127.0.0.1:8001
├─ 初始化环境 (AlohaRealEnvironment)
│  ├─ 连接4个RealSense摄像头
│  ├─ 连接Interbotix ARM控制器
│  └─ 重置到初始位置 (reset_position)
└─ 创建Runtime控制器

┌─────────────────────────────────────────────────────┐
│             主循环 (50Hz)                           │
└─────────────────────────────────────────────────────┘

FOR each timestep (max 10,000):
  │
  ├─ 步骤1: 观测获取 (AlohaRealEnvironment.step)
  │  ├─ 摄像头读取 (RealSense ×4)
  │  │  └─ 实际分辨率: 1280×720 → 裁切+缩放到224×224
  │  ├─ 机器人状态读取
  │  │  ├─ 左臂关节角度 [6]
  │  │  ├─ 左夹爪位置 [1]
  │  │  ├─ 右臂关节角度 [6]
  │  │  └─ 右夹爪位置 [1]
  │  └─ 坐标转换 (adapt_to_pi=True)
  │     └─ 关节翻转 + 夹爪角度转换
  │
  ├─ 步骤2: 策略推理 (ActionChunkBroker)
  │  │
  │  ├─ 缓冲检查
  │  │  ├─ 如果action_buffer有剩余 → 返回buffer[0]
  │  │  └─ 否则 → 请求新动作
  │  │
  │  ├─ WebSocket请求 (每25步一次)
  │  │  └─ 发送: {images, state, prompt}
  │  │
  │  ├─ Policy处理
  │  │  ├─ 低级Server:
  │  │  │  ├─ 如果use_rtc: guided_inference(prev_action)
  │  │  │  │  └─ 返回 [batch=1, 50, 14]
  │  │  │  └─ 否则: sample_actions()
  │  │  │
  │  │  └─ 高级Server (异步):
  │  │     └─ sample_subtask_tokens()
  │  │        └─ 返回当前子任务文本
  │  │
  │  └─ ActionChunkBroker管理
  │     ├─ 收到50步预测
  │     ├─ 保存当前prev_action (用于RTC)
  │     ├─ 缓冲25步执行
  │     └─ 返回当前要执行的动作
  │
  ├─ 步骤3: 动作执行 (AlohaRealEnvironment.send_action)
  │  ├─ 动作转换
  │  │  ├─ 模型输出[14] → 关节+夹爪
  │  │  ├─ 反向apply adapt_to_pi (关节翻转+夹爪转换)
  │  │  └─ 归一化到ALOHA硬件范围
  │  ├─ 发送指令
  │  │  ├─ 左臂IK → Interbotix硬件
  │  │  ├─ 右臂IK → Interbotix硬件
  │  │  └─ 夹爪PWM → 对应执行器
  │  └─ 硬件延迟: ~50ms (反馈闭环)
  │
  ├─ 步骤4: 数据记录 (可选HDF5)
  │  └─ h5df_saver.save({
  │       images, state, action, timestamp
  │     })
  │
  └─ 等待下一个50Hz周期

完成: episode结束或max_steps
├─ 安全关机
│  └─ sleep_arms() → 回到安全位置
└─ 保存episode数据
```

### 4.4 数据流速对照表

| 组件 | 频率 | 说明 |
|------|------|------|
| **摄像头采集** | 50Hz | RealSense FPS=50 |
| **状态读取** | 50Hz | Interbotix反馈 |
| **Policy推理** | 50Hz | 每帧1次推理 |
| **Policy请求** | 2Hz | 每25帧一次 (RTC频率) |
| **动作执行** | 50Hz | ARM控制频率 |
| **模型输出** | 50步 | action_horizon=50 |
| **执行缓冲** | 25步 | ActionChunkBroker |

---

## 5. 核心代码位置

### 5.1 数据处理

| 文件 | 行号 | 功能 |
|------|------|------|
| `src/openpi/policies/aloha_policy.py` | 10-21 | 示例数据结构 |
| `src/openpi/policies/aloha_policy.py` | 25-88 | AlohaInputs 转换 |
| `src/openpi/policies/aloha_policy.py` | 92-106 | AlohaOutputs 转换 |
| `src/openpi/policies/aloha_policy.py` | 109-215 | 坐标系转换函数 |
| `src/openpi/models/model.py` | 105-216 | Observation 数据类 |
| `src/openpi/models/model.py` | 219-302 | 图像预处理 |

### 5.2 训练

| 文件 | 行号 | 功能 |
|------|------|------|
| `src/openpi/training/config.py` | 1 | 配置列表 |
| `src/openpi/training/config.py` | 612 | _CONFIGS 定义 |
| `src/openpi/training/config.py` | 986 | twist_off_bottle_cap配置 |
| `scripts/train.py` | 100-148 | init_train_state |
| `scripts/train.py` | 152-221 | train_step 循环 |
| `scripts/train.py` | 224-315 | main 训练管道 |
| `src/openpi/training/data_loader.py` | 55-84 | TransformedDataset |

### 5.3 推理

| 文件 | 行号 | 功能 |
|------|------|------|
| `src/openpi/models/pi0.py` | 322-387 | sample_actions() |
| `src/openpi/models/pi0.py` | 390-492 | sample_subtask_tokens() |
| `src/openpi/models/pi0.py` | 494-550+ | guided_inference() RTC |
| `src/openpi/models/pi0.py` | 19-44 | make_attn_mask() |
| `src/openpi/models/pi0.py` | 66-82 | posemb_sincos() |

### 5.4 机器人控制

| 文件 | 行号 | 功能 |
|------|------|------|
| `examples/aloha_real/main.py` | 19-54 | Args 配置 |
| `examples/aloha_real/main.py` | 56-128 | main() 流程 |
| `examples/aloha_real/env.py` | - | AlohaRealEnvironment |
| `examples/aloha_real/real_env.py` | - | 底层环境接口 |
| `src/openpi/serving/websocket_policy_server.py` | 1-124 | Policy Server |

### 5.5 模型架构

| 文件 | 行号 | 功能 |
|------|------|------|
| `src/openpi/models/pi0_config.py` | 18-89 | Pi0Config 配置 |
| `src/openpi/models/pi0.py` | 85-320 | Pi0 完整模型 |
| `src/openpi/models/gemma.py` | - | PaliGemma + ActionExpert |
| `src/openpi/models/siglip.py` | - | SigLIP视觉编码器 |
| `src/openpi/models/tokenizer.py` | - | PaliGemma & FAST tokenizer |

---

## 6. 调试指南

### 6.1 子任务识别调试

```python
from openpi.policies import aloha_policy
from openpi.models import model as _model
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 加载数据集
ds = LeRobotDataset('lyl472324464/wipe_the_desk_with_a_blue_cloth')
sample = ds[0]

# 转换数据
aloha_input = aloha_policy.AlohaInputs(adapt_to_pi=True)
observation_dict = aloha_input(sample)
observation = _model.Observation.from_dict(observation_dict)

# 推理子任务
rng = jax.random.key(0)
subtask_tokens = policy.sample_subtask_tokens(rng, observation)

# 解码
from openpi.models.tokenizer import PaligemmaTokenizer
tokenizer = PaligemmaTokenizer(max_token_len=250)
subtask_text = tokenizer.decode(subtask_tokens[0])
print(f"当前子任务: {subtask_text}")
```

### 6.2 动作预测调试

```python
# 单步推理
actions = policy.sample_actions(rng, observation, num_steps=10)
print(f"预测50步动作: shape={actions.shape}")  # [1, 50, 14]

# 检查action范围
import numpy as np
print(f"动作范围: [{actions.min():.3f}, {actions.max():.3f}]")
# 应该在[-1, 1]范围内

# RTC推理
prev_actions = actions.copy()
rtc_actions = policy.guided_inference(
    rng, 
    prev_actions, 
    observation,
    s=20, d=17, beta=8.0
)
print(f"RTC动作前20步偏差: {np.mean(np.abs(rtc_actions[:, :20] - prev_actions[:, :20]))}")
# 应该很小（保持连续性）
```

### 6.3 数据集检查

```bash
# 检查LeRobot数据集
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

ds = LeRobotDataset('lyl472324464/wipe_the_desk_with_a_blue_cloth')
print(f'总episodes: {len(ds)}')

sample = ds[0]
print(f'样本keys: {sample.keys()}')
print(f'观测keys: {sample[\"observation\"].keys()}')
print(f'图像shape: {sample[\"observation\"][\"images\"][\"cam_high\"].shape}')
print(f'状态shape: {sample[\"observation\"][\"state\"].shape}')
print(f'动作shape: {sample[\"action\"].shape}')
print(f'提示词: {sample.get(\"task\", \"N/A\")}')
"
```

### 6.4 Policy Server日志

```bash
# 查看低级server日志
docker compose logs -f openpi_server

# 查看高级server日志
docker compose logs -f openpi_server_high_level

# 常见日志信息
# "Creating server (..."
# "server listening on 0.0.0.0:8000"  ← 就绪
# WebSocket连接时显示 "new websocket client"
```

### 6.5 Runtime 诊断

```bash
# 实时查看日志
docker compose exec runtime tail -f /tmp/runtime.log

# 重启runtime（在policy server准备好后）
docker compose restart runtime

# 检查机器人连接
docker compose exec runtime bash -c "
  source /opt/ros/noetic/setup.bash &&
  source /root/interbotix_ws/devel/setup.bash &&
  rostopic list  # 应显示arm和gripper话题
"
```

### 6.6 性能诊断

```python
# 在policy server中添加计时
import time

start = time.time()
actions = policy.sample_actions(rng, observation, num_steps=10)
inference_time = time.time() - start
print(f"推理时间: {inference_time*1000:.1f}ms")
# 应该 < 20ms (50Hz = 20ms per frame)

# 内存使用
import jax
print(f"已用GPU内存: {jax.device_get(model.parameters).size / 1e9:.1f}GB")
```

---

## 附录：关键术语

| 术语 | 含义 | 重要性 |
|------|------|--------|
| **Flow Matching** | 扩散模型的新一代训练方式，效率更高 | ★★★★★ |
| **RTC** (Real-Time Control) | 使用前一步动作引导当前预测 | ★★★★★ |
| **ActionChunkBroker** | 管理50步动作缓冲，25步重新计划 | ★★★★☆ |
| **KV缓存** | Transformer推理优化，减少重复计算 | ★★★★☆ |
| **EMA** (Exponential Moving Average) | 训练中跟踪参数的指数移动平均 | ★★★☆☆ |
| **FSDP** (Fully Sharded Data Parallel) | JAX分布式训练，8个GPU并行 | ★★★★☆ |
| **Norm Stats** | 数据归一化统计，每个robot platform不同 | ★★★★★ |
| **adapt_to_pi** | ALOHA坐标系 → PI0内部坐标系转换 | ★★★★★ |

