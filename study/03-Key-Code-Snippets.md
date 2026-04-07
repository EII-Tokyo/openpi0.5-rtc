# OpenPI ALOHA 关键代码详解

## 1. 数据转换管道

### 1.1 ALOHA输入转换 (AlohaInputs)

**源文件**: `src/openpi/policies/aloha_policy.py:24-88`

```python
@dataclasses.dataclass(frozen=True)
class AlohaInputs(transforms.DataTransformFn):
    """
    目的: 将LeRobot原始数据转换成模型输入格式
    
    输入约定:
      - images: dict with keys in EXPECTED_CAMERAS
      - state: [14] = [left_6j, left_gripper, right_6j, right_gripper]
      - actions: [action_horizon, 14]
    
    输出格式:
      {
        "image": {base_0_rgb, base_1_rgb, left_wrist_0_rgb, right_wrist_0_rgb},
        "image_mask": bool masks,
        "state": [14],
        "prompt": str (可选),
        "subtask": str (可选)
      }
    """
    
    adapt_to_pi: bool = True  # ← 关键：启用ALOHA→PI坐标转换
    EXPECTED_CAMERAS = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")
    
    def __call__(self, data: dict) -> dict:
        # 步骤1: 坐标系转换（ALOHA ↔ PI0）
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)
        
        # 步骤2: 图像映射（摄像头命名 → 模型期望的名字）
        base_image = in_images["cam_high"]  # 总是存在
        images = {"base_0_rgb": base_image}
        
        # 步骤3: 动态添加额外摄像头（动态4/3摄像头支持！）
        extra_image_names = {
            "base_1_rgb": "cam_low",
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]  # 存在则添加
                image_masks[dest] = True
        
        return {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
            "prompt": data.get("prompt"),
            "subtask": data.get("subtask"),
        }
```

### 1.2 坐标转换细节 (_decode_state)

**源文件**: `src/openpi/policies/aloha_policy.py:193-199`

```python
def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    """
    ALOHA的state是Interbotix原始格式
    PI0期望转换后的格式
    
    转换内容:
    1. 关节角度翻转（某些关节方向相反）
    2. 夹爪位置转换（线性→角度）
    """
    if adapt_to_pi:
        # 关节翻转掩码：对应关节*-1
        # [1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1]
        #  L0 L1 L2 L3 L4 L5 LG  R0 R1 R2 R3 R4 R5 RG
        state = _joint_flip_mask() * state  # 元素乘法
        
        # 夹爪变换（注释掉，但代码框架在这）
        # state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
    
    return state
```

### 1.3 夹爪转换数学

**源文件**: `src/openpi/policies/aloha_policy.py:122-156`

```python
def _gripper_to_angular(value):
    """
    LeRobot存储的是**线性夹爪位置** (mm)
    但PI0期望**角度** (radians)
    
    转换过程:
    1. 反归一化: [0.01844, 0.05800] → 物理长度
    2. 线性→角度: 通过Interbotix fork-bar机械模型
       value = arcsin((h² + l² - L²) / (2*h*l))
       其中 h=horn_radius=0.022, L=arm_length=0.036
    3. 重新归一化: [0.5476, 1.6296] → [-1, 1]
    
    为什么这么复杂？
    - ALOHA用线性Actuator（直线动作）
    - Interbotix用回转关节（角度动作）
    - Fork-bar是中间的转换机械
    """
    
    # 线性范围: [0.01844, 0.05800] (Aloha缩放)
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)
    
    # 线性→角度的物理公式
    def linear_to_radian(linear_position, arm_length=0.036, horn_radius=0.022):
        # 这是Fork-bar的几何关系
        numerator = horn_radius**2 + linear_position**2 - arm_length**2
        denominator = 2 * horn_radius * linear_position
        cos_value = numerator / denominator
        return np.arcsin(np.clip(cos_value, -1.0, 1.0))
    
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)
    
    # 角度范围: [0.5476, 1.6296] (PI0范围)
    return _normalize(value, min_val=0.5476, max_val=1.6296)
```

---

## 2. 训练循环

### 2.1 train_step() 详解

**源文件**: `scripts/train.py:152-221`

```python
@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """
    单个训练步的完整流程
    
    分布式设置（FSDP）:
    - 输入数据 按 DATA 轴分片 (batch维)
    - 模型参数 按 FSDP 风格分片 (多GPU)
    - 梯度 自动合并
    """
    
    # 步骤1: 重建模型（从params + graphdef）
    # 为什么要重建？因为在NNX中模型是分离的state和graphdef
    model = nnx.merge(state.model_def, state.params)
    model.train()  # 启用dropout等
    
    # 步骤2: 定义损失函数（有梯度的）
    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions
    ):
        if hasattr(model, "compute_loss_with_metrics"):
            # PI05有子任务损失
            chunked_loss, flow_chunked_loss, subtask_ar_chunked_loss = \
                model.compute_loss_with_metrics(rng, observation, actions, train=True)
            return jnp.mean(chunked_loss), {
                "flow_loss": jnp.mean(flow_chunked_loss),
                "subtask_ar_loss": jnp.mean(subtask_ar_chunked_loss),
            }
        else:
            # PI0只有flow matching损失
            chunked_loss = model.compute_loss(rng, observation, actions, train=True)
            return jnp.mean(chunked_loss), {
                "flow_loss": jnp.mean(chunked_loss),
                "subtask_ar_loss": jnp.asarray(0.0, dtype=chunked_loss.dtype),
            }
    
    # 步骤3: 计算损失和梯度（使用DiffState过滤可训练参数）
    observation, actions = batch
    train_rng = jax.random.fold_in(rng, state.step)  # 不同step不同RNG
    
    # DiffState: 只计算可训练参数的梯度（freeze_filter定义）
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux_metrics), grads = nnx.value_and_grad(
        loss_fn,
        argnums=diff_state,
        has_aux=True
    )(model, train_rng, observation, actions)
    
    # 步骤4: 优化器更新
    params = state.params.filter(config.trainable_filter)  # 可训练参数
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    # 步骤5: 把更新应用回模型
    nnx.update(model, new_params)
    new_params = nnx.state(model)
    
    # 步骤6: 创建新state
    new_state = dataclasses.replace(
        state,
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state
    )
    
    # 步骤7: EMA更新（指数移动平均）
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params
            ),
        )
    
    # 步骤8: 计算诊断信息
    # 只统计kernel参数（不含bias, scale, pos_embedding等）
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    
    info = {
        "loss": loss,
        "flow_loss": aux_metrics["flow_loss"],
        "subtask_ar_loss": aux_metrics["subtask_ar_loss"],
        "grad_norm": optax.global_norm(grads),      # 梯度范数
        "param_norm": optax.global_norm(kernel_params),  # 参数范数
    }
    
    return new_state, info
```

---

## 3. 推理系统

### 3.1 Flow Matching 采样

**源文件**: `src/openpi/models/pi0.py:322-387`

```python
def sample_actions(
    self,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    *,
    num_steps: int | at.Int[at.Array, ""] = 10,
    noise: at.Float[at.Array, "b ah ad"] | None = None,
) -> _model.Actions:
    """
    Flow Matching ODE求解
    
    原理:
      从x_1 (纯噪声) 逐步去噪到 x_0 (清晰动作)
      x_t = (1-t)*x_0 + t*noise
      通过学习 v(x_t, t) ≈ dx_t/dt
      进行ODE更新: x_t ← x_t - dt * v_t
    
    为什么是"Flow Matching"?
    - 传统扩散: 学习分数函数∇log p(x_t)
    - Flow Matching: 直接学习向量场v(x_t)
    - 优势: 更稳定，收敛快，可用任意ODE求解器
    """
    
    # 预处理观测（图像缩放、数据增强等）
    observation = _model.preprocess_observation(
        None, observation, train=False, image_resolution=self.image_resolution
    )
    
    # ODE参数
    dt = -1.0 / num_steps  # 时间步长（负数，从1→0）
    batch_size = observation.state.shape[0]
    
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
    
    # 前缀前向（充填KV缓存）
    # 前缀 = 图像特征 + 提示词编码
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, self.action_horizon)))
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    
    # 一次性前向，保存KV缓存用于后续步
    _, kv_cache = self.PaliGemma.llm(
        [prefix_tokens, None],
        mask=prefix_attn_mask,
        positions=positions
    )
    
    # ODE循环
    def step(carry):
        """
        单个ODE步骤
        输入: (x_t, time_t)
        输出: (x_{t-dt}, time_{t-dt})
        """
        x_t, time = carry
        
        # 后缀编码（当前noisy action）
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = \
            self.embed_suffix(observation, x_t, jnp.broadcast_to(time, batch_size))
        
        # 构建注意力掩码
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        
        # 位置编码
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + \
                   jnp.cumsum(suffix_mask, axis=-1) - 1
        
        # 前向传播（使用KV缓存，避免重复计算前缀）
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],  # None表示使用缓存的前缀
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        
        # 提取速度预测（最后50个token对应50步action）
        assert prefix_out is None
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        
        # ODE更新: x_t ← x_t - dt*v_t
        # 等价于: x_t ← x_t + |dt|*v_t (因为dt是负数)
        return x_t + dt * v_t, time + dt
    
    # ODE停止条件
    def cond(carry):
        x_t, time = carry
        return time >= -dt / 2  # 当time接近0时停止
    
    # 执行ODE循环
    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    
    return x_0
```

### 3.2 自回归子任务解码

**源文件**: `src/openpi/models/pi0.py:390-492`

```python
@at.typecheck
def sample_subtask_tokens(
    self,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    *,
    temperature: float = 0.0,
    eos_token_id: int | None = None,
    max_text_token_id: int = 240000,
    debug_top_logits: bool = False,
) -> at.Int[at.Array, "b _t"]:
    """
    自回归语言生成（生成当前子任务描述）
    
    原理:
      P(w_i | w_{<i}, image, prompt) 通过LLM输出logits
      t=0: logits = prefix_hidden @ embedding_table^T
      t>0: logits = last_token_hidden @ embedding_table^T
    
    为什么需要KV缓存?
    - 每步都要重新计算整个序列的attention会很慢
    - KV缓存保存前面步的K,V矩阵
    - 新token只需要计算它自己的Q，与所有历史K,V相乘
    """
    
    observation = _model.preprocess_observation(
        None, observation, train=False, image_resolution=self.image_resolution
    )
    
    if observation.tokenized_prompt is None or observation.tokenized_prompt_mask is None:
        raise ValueError("tokenized_prompt required for subtask decoding")
    if observation.tokenized_subtask is None:
        raise ValueError("tokenized_subtask required")
    
    # 前缀编码与前向
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    max_steps = int(observation.tokenized_subtask.shape[1])
    subtask_len = int(observation.tokenized_subtask.shape[1])
    
    # 移除前缀中的子任务部分（保留提示词部分）
    prefix_tokens = prefix_tokens[:, :-subtask_len, :]
    prefix_mask = prefix_mask[:, :-subtask_len]
    prefix_ar_mask = prefix_ar_mask[:-subtask_len]
    prefix_len = prefix_tokens.shape[1]
    
    # 获取词表（用于计算logits）
    embed_table = self.PaliGemma.llm.embedder["input_embedding"].value
    vocab_size = embed_table.shape[0]
    
    # 过滤掩码
    special_token_mask = jnp.arange(vocab_size) >= (vocab_size - 128)  # 特殊token
    non_text_mask = jnp.arange(vocab_size) > max_text_token_id  # 非文本token
    
    # 前缀前向（得到initial context）
    prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_steps)))
    
    (prefix_out, _), kv_cache = self.PaliGemma.llm(
        [prefix_tokens, None],
        mask=prefix_attn_mask,
        positions=prefix_positions
    )
    
    # 获取前缀最后的隐藏状态
    assert prefix_out is not None
    batch_size = prefix_out.shape[0]
    last_valid_idx = jnp.max(
        jnp.where(prefix_mask, jnp.arange(prefix_len)[None, :], -1),
        axis=1,
    )
    last_hidden = prefix_out[jnp.arange(batch_size), last_valid_idx, :]
    prefix_valid_len = jnp.sum(prefix_mask, axis=1, dtype=jnp.int32)
    
    # 自回归生成循环
    def step(carry):
        rng, last_hidden, output_tokens, finished, step_i, kv_cache = carry
        
        # 步骤1: 计算logits (内积相似度)
        logits = jnp.einsum(
            "bd,vd->bv",
            last_hidden,
            embed_table,
            preferred_element_type=jnp.float32
        )
        
        # 步骤2: 过滤非法token
        logits = jnp.where(special_token_mask[None, :], -jnp.inf, logits)  # 特殊
        logits = jnp.where(non_text_mask[None, :], -jnp.inf, logits)  # 非文本
        logits = logits.at[:, 0].set(-jnp.inf)  # PAD token
        
        # 步骤3: 采样
        if temperature <= 0.0:
            next_token = jnp.argmax(logits, axis=-1).astype(jnp.int32)  # 贪心
        else:
            rng, sample_rng = jax.random.split(rng)
            next_token = jax.random.categorical(
                sample_rng,
                logits / temperature,
                axis=-1
            ).astype(jnp.int32)
        
        # 步骤4: 检查EOS token
        if eos_token_id is not None:
            next_token = jnp.where(finished, jnp.asarray(eos_token_id, dtype=jnp.int32), next_token)
            finished = jnp.logical_or(finished, next_token == jnp.asarray(eos_token_id, dtype=jnp.int32))
        
        # 步骤5: 保存token
        output_tokens = put_along_last_axis(
            output_tokens,
            jnp.broadcast_to(step_i, (next_token.shape[0], 1)),
            next_token[:, None],
        )
        
        # 步骤6: Token embedding
        token_embeddings = self.PaliGemma.llm(next_token[:, None], method="embed")
        
        # 步骤7: 更新KV缓存，继续前向
        generated_valid = (jnp.arange(max_steps)[None, :] <= step_i)
        key_mask = jnp.concatenate([prefix_mask, generated_valid], axis=1)
        full_attn_mask = key_mask[:, None, :]
        positions = prefix_valid_len[:, None] + step_i
        
        (full_out, _), kv_cache = self.PaliGemma.llm(
            [token_embeddings, None],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
        )
        assert full_out is not None
        last_hidden = full_out[:, 0, :]  # 取生成token的隐藏状态
        
        return rng, last_hidden, output_tokens, finished, step_i + 1, kv_cache
    
    # 停止条件
    def cond(carry):
        _, _, _, finished, step_i, _ = carry
        if eos_token_id is None:
            return step_i < max_steps
        return (~jnp.all(finished)) & (step_i < max_steps)
    
    # 生成循环
    init_tokens = jnp.zeros((batch_size, max_steps), dtype=jnp.int32)
    init_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)
    _, _, output_tokens, _, _, _ = jax.lax.while_loop(
        cond,
        step,
        (rng, last_hidden, init_tokens, init_finished, 0, kv_cache),
    )
    
    return output_tokens
```

---

## 4. RTC 实时控制

### 4.1 引导推理 (guided_inference)

**源文件**: `src/openpi/models/pi0.py:494-550+`

```python
def guided_inference(
    self,
    rng: at.KeyArrayLike,
    prev_action: _model.Actions,      # 上一次的预测 [b, 50, 14]
    observation: _model.Observation,
    *,
    num_steps: int | at.Int[at.Array, ""] = 10,
    s: int = 20,      # 前s步保持旧动作
    d: int = 17,      # 后d步用旧动作引导
    beta: float = 8.0,  # 引导强度
) -> _model.Actions:
    """
    RTC (Real-Time Control) 引导推理
    
    关键思想:
      对于实时控制，我们想要：
      1. 连续性: 前几步与上次预测相同
      2. 灵活性: 后几步可以改变
      3. 约束: 最后几步受上次预测的"引导"（但允许改进）
    
    s = 20 (保守期): prev_action[:, 0:20] 直接用，不质疑
    d = 17 (引导期): prev_action[:, 33:50] 作为"目标"，引导新预测
              (自由期): 新预测[:, 20:33] 完全自由
    
    时间轴 (假设每帧20ms，50Hz):
      t=0ms:   执行 prev_action[0]
      t=20ms:  执行 prev_action[1]    (前20个都用prev)
      ...
      t=400ms: 执行 prev_action[20]   (现在请求新预测)
      t=420ms: 执行 new_pred[0]       (开始执行新预测)
      t=440ms: 执行 new_pred[1]       (新预测中间部分)
      ...
      t=660ms: 执行 new_pred[20]      (开始引导部分)
      t=1000ms: 执行 new_pred[30]     (最后10步，被prev引导过)
    """
    
    # 整理prev_action用于引导
    # 我们要：[prev[0:s], 自由部分, prev[s:] 作为引导]
    prev_action_slice = prev_action[:, s:, :]  # 取prev[20:]
    zero_actions = jnp.zeros((batch_size, s, self.action_dim))
    prev_action_slice = jnp.concatenate([prev_action_slice, zero_actions], axis=1)
    # 现在: prev_action_slice[0:30] = prev[20:50]
    #      prev_action_slice[30:50] = zeros (后30步无引导)
    
    # ... (与sample_actions相同的初始化和前缀编码)
    
    def step(carry):
        x_t, time = carry
        
        # ... (与sample_actions相同的后缀编码和forward)
        
        # 速度预测
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        
        # ★ RTC关键: 使用加权平均结合v_t和prev动作
        # beta大 → 更多依赖prev（约束强）
        # beta小 → 更多依赖v_t（自由度高）
        
        # 简化形式（伪代码）:
        # if step_i >= s:  # 在保守期之后
        #   if step_i >= (50-d):  # 在引导期
        #     beta_t = beta * (1 - (step_i - (50-d)) / d)  # 线性衰减
        #     v_t = v_t + beta_t * (prev_action_slice[step_i] - x_t)
        
        return x_t + dt * v_t, time + dt
    
    # 执行ODE循环
    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x_0
```

---

## 5. Runtime 集成

### 5.1 ActionChunkBroker

**源文件**: `examples/aloha_real/main.py:83-90` (使用)
**实现**: `openpi-client` 包（外部）

```python
# Runtime中的使用
policy = action_chunk_broker.ActionChunkBroker(
    policy=runtime_policy,           # 底层WebSocket客户端
    action_horizon=args.action_horizon,  # 50
    model_dir=args.model_dir,        # checkpoint目录
    adapt_to_pi=args.adapt_to_pi,    # True (坐标转换)
    use_rtc=args.use_rtc,            # True (启用RTC引导)
)

# 每一步
action = policy.step(observation)  # 返回单个动作

# 内部机制:
# 初始化时: 申请一次动作 → 获得50步，存入buffer
# 步骤0-24: 从buffer返回，无新请求
# 步骤25: buffer用完，重新请求 (RTC引导)
# 步骤25-49: 从新buffer返回
# ...循环
```

### 5.2 AlohaRealEnvironment

**源文件**: `examples/aloha_real/env.py`

```python
class AlohaRealEnvironment:
    """
    与真实ALOHA机器人的接口
    
    功能:
    1. 读取摄像头和关节状态
    2. 发送动作指令到硬件
    3. 处理坐标转换和安全检查
    """
    
    def __init__(self, reset_position, gripper_current_limits):
        self.reset_position = reset_position  # 初始位置
        self.gripper_current_limits = gripper_current_limits
        # 初始化ROS节点、连接ARM和gripper驱动
    
    def reset(self):
        """重置到初始位置"""
        # 向目标位置动作
        # 等待到达
        # 读取初始观测
        return observation
    
    def step(self, action):
        """
        执行单个动作步骤
        
        action: [14] = [左6关节, 左夹爪, 右6关节, 右夹爪]
        
        流程:
        1. 动作安全检查
        2. 发送到ARM控制器
        3. 等待反馈
        4. 读取新观测
        """
        # 验证action范围
        assert action.min() >= -1.0 and action.max() <= 1.0
        
        # 发送到硬件 (需要adapt_to_pi逆变换)
        # ...
        
        # 读取新观测
        return observation, done, info
    
    def sleep_arms(self):
        """安全关机：移到睡眠位置"""
        # 这很重要！避免机器人突然停止
```

