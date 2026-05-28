# RLT Actor-Critic 第二阶段设计规划

本文档规划 RLT 论文第二阶段：在已经训练好的 RL Token 网络基础上，冻结 VLA 和 RL Token extractor，训练轻量 actor 和 Q critic 网络，用于改善双臂 ALOHA 拧瓶盖任务中“右臂靠近瓶盖后第一次夹取 miss”的问题。

## 目标

当前问题不是完整任务不会做，而是精细阶段不稳定：

- 左臂通常能固定瓶子。
- 右臂能靠近瓶盖。
- 但右夹爪第一次闭合时约 30% miss，夹不住瓶盖。

因此 RLT 第二阶段不应该重新学习完整拧瓶盖流程，而应该重点优化关键阶段：

```text
右臂接近瓶盖 -> 右夹爪对准 -> 第一次闭合 -> 是否夹住瓶盖 -> 夹住后开始旋转
```

第一版训练目标建议聚焦：

```text
提高第一次夹住瓶盖的概率
```

而不是一开始就把“完全拧开瓶盖”作为唯一 reward。

## RLT 论文对应关系

论文第二阶段算法里：

$$
x_t = \left(z_{\mathrm{rl}}(s_t), s_t^p\right)
$$

其中：

- `z_rl(s_t)`：论文里的高层写法，表示“从当前观测 `s_t` 中提取 RL Token”。实际实现时，不是把 proprio 或原始状态向量直接输入 RL Token 网络，而是先把 `s_t` 里的图像观测送入冻结 VLA，得到有序的 VLA image-token hidden sequence，然后由冻结 RL Token encoder 压缩成 `z_rl`。
- `s_t^p`：机器人 proprioception，例如双臂关节、夹爪状态。
- `x_t`：给 actor 和 critic 的 RL 状态。

也就是说，第二阶段 actor/critic 的状态可以理解为：

$$
x_t =
\left(
z_{\mathrm{rl}}(\bar{z}_{1:M}(s_t)),
s_t^p
\right)
$$

其中：

- `\bar{z}_{1:M}(s_t)`：冻结 VLA 从当前图像观测中产生的 image-token embedding 序列。
- `z_rl(...)`：冻结 RL Token 网络对这串 image-token embedding 的压缩结果。
- `s_t^p`：单独拼接的 proprio，不参与 RL Token 压缩。

如果任务语言指令固定，按照论文说明，这一步可以 drop language embeddings；当前 RL Token 训练代码也是按这个方向实现的，只保留 image-token hidden sequence 来训练/提取 `z_rl`。

动作部分：

$$
\tilde{a}_{t:t+C-1} \sim \pi_{\mathrm{vla}}(s_t)
$$

$$
a_{t:t+C-1} \sim \pi_{\theta}\left(\cdot \mid x_t, \tilde{a}_{t:t+C-1}\right)
$$

其中：

- `ã` 是 VLA 参考动作 chunk。
- `a` 是 actor 输出的最终动作 chunk。
- actor 不是从零生成动作，而是在 VLA 动作附近做修正。

critic：

$$
Q_{\psi}(x, a)
$$

policy loss：

$$
\mathcal{L}_{\pi}(\theta)
=
\mathbb{E}_{b}
\left[
-Q_{\psi}(x, a)
+
\beta \left\|a - \tilde{a}\right\|_2^2
\right]
$$

这表示：actor 想找 Q 更高的动作，但不能离 VLA 参考动作太远。

## 状态和动作维度

当前 pi0 内部训练空间：

```text
z_rl: [2048]
proprio / state: [32]
action chunk: [50, 32]
```

实际 ALOHA 机器人执行空间：

```text
raw action: [14]
双臂各 6 个关节 + 1 个夹爪
```

RLT actor 和 critic 建议在 pi0 内部空间训练：

```text
x = concat(z_rl, proprio)      # [2080]
ã = VLA action chunk           # [50, 32]
a = actor action chunk         # [50, 32]
```

但 rollout 数据中必须同时保存 raw action `[14]`，用于回放、调试、人工分析和真实执行。

## Actor 网络设计

第一版 actor 用 residual action 形式：

```text
input:  x, ã
output: Δa
a = ã + Δa
```

网络：

```text
input_dim = 2048 + 32 + 50 * 32
hidden = 1024
layers = 3
activation = gelu 或 silu
output = mean_delta, log_std_delta
```

训练时采样：

$$
\Delta a \sim \mathcal{N}(\mu_{\Delta a}, \sigma_{\Delta a})
$$

$$
a = \tilde{a} + \Delta a
$$

需要 reference-action dropout：

```text
训练时以一定概率把 ã 置零或 mask
```

原因：如果 actor 总是看到 VLA 参考动作，它可能只学会复制 VLA，不学会独立修正。

## Twin Q Critic 网络设计

critic 输入：

```text
x, a
```

输出：

```text
Q1(x, a) -> scalar
Q2(x, a) -> scalar
```

网络：

```text
input_dim = 2048 + 32 + 50 * 32
hidden = 1024
layers = 3
activation = gelu 或 silu
output = 1
```

这里应该严格按论文的 TD3 风格实现 twin Q，而不是单 Q：

```text
Q_ψ = {Q_ψ1, Q_ψ2}
Q_ψ' = {Q_ψ1', Q_ψ2'}   # target critics
```

原因：

- actor 会主动寻找 critic 认为高价值的动作。
- 单 Q 容易高估某些动作，actor 会 exploit 这个高估。
- TD3 的 clipped double-Q 用两个 critic 的较小值做 target，降低 Q 过估计风险。

target value 使用：

$$
\min_{i=1,2} Q_{\psi_i'}(x', a')
$$

actor 更新时可以使用 `Q1`，也可以使用两个 Q 的较小值。第一版建议：

```text
critic target: min(Q1_target, Q2_target)
actor loss:    -Q1(x, actor_action) + β ||a - ã||²
```

这是 TD3 常见实现方式，计算简单，也符合论文“类似 TD3”的要求。

## Replay Buffer 数据结构

硬约束：

```text
RLT replay buffer 只保存关键区域 transition。
整条 episode 可以继续保存为普通 rollout 日志、视频、HDF5 或调试数据，
但非关键区域 transition 不进入 actor/critic 的 replay buffer。
```

原因：

- 论文的 RLT 第二阶段是 targeted improvement of critical phases。
- base VLA 已经能完成非关键区域，RLT 不应该重新学习整条轨迹。
- 非关键区域 transition 会稀释关键抓取阶段的数据，增加 credit assignment 难度。
- 当前任务失败点集中在“右臂靠近瓶盖后的第一次夹取”，因此 replay buffer 应该集中保存这一段。

建议每个 chunk 存一条 transition：

```python
{
    "z_rl": float32[2048],
    "proprio": float32[32],
    "vla_action": float32[50, 32],      # ã
    "actor_action": float32[50, 32],    # a
    "reward_seq": float32[50],
    "reward": float32,
    "next_z_rl": float32[2048],
    "next_proprio": float32[32],
    "done": bool,
    "metadata": {
        "episode_id": str,
        "chunk_idx": int,
        "raw_vla_action": float32[50, 14],
        "raw_actor_action": float32[50, 14],
        "is_key_region": bool,
        "key_region_phase": str,
        "human_grasp_success": bool | None,
        "human_miss": bool | None,
        "auto_gripper_closed": bool | None,
        "auto_grasp_likely": bool | None,
    },
}
```

## Reward 设计原则

你当前确定的 reward 与关键区域控制方式是：

```text
人通过实时四路摄像头观察 ALOHA，尤其是 right wrist 视角。
S: 关键区域开始。
E: 关键区域结束。
关键区域结束后，人工输入评分：
    1: 成功，例如右夹爪夹住瓶盖，当前 RL 子任务完成。
    0: 失败，例如 miss，没夹住瓶盖，或动作无效。
如果 E 后 10 秒内没有评分，默认失败，reward=0。
```

这是第一版最适合当前场景的方案。原因是：

- 你有实时四个摄像头，尤其 right wrist 能直接观察是否夹住瓶盖。
- 当前失败点很细，自动判断“夹住瓶盖”容易误判。
- 人工控制开始/结束能保证 replay buffer 只包含真正关键区域。
- 这与 RLT 论文中“critical phase 由 human operator 选择，terminal signal 由 human operator 给出”的流程一致。

所以第一版 reward 必须是 human-in-the-loop terminal reward：

```text
关键区域结束后：
    人工评分 1 -> terminal reward = 1
    人工评分 0 -> terminal reward = 0
    10 秒内未评分 -> terminal reward = 0
```

程序自动信号不作为主 reward，也不作为第一版关键区域开始/结束依据，只作为辅助字段：

```text
auto_gripper_closed
auto_grasp_likely
right_wrist_near_cap
auto_close_detected
```

这些字段可以用于 debug、后续自动化、检查人工标签质量，但第一版训练数据的边界和 reward 以人工输入为准。

## 第一版 Reward 方案

### 主 reward

第一版只优化“第一次夹住瓶盖”：

```text
关键区域内：
    S 开始记录 RLT transition
    E 停止记录 RLT transition
    E 后人工输入 1 -> 本关键区域 terminal reward = 1
    E 后人工输入 0 -> 本关键区域 terminal reward = 0
    E 后 10 秒内没有输入 -> 本关键区域 terminal reward = 0
```

这个 reward 直接对应你的问题：

```text
miss 的样本 reward=0
夹住的样本 reward=1
```

### 为什么不是 miss 给 -1

第一版建议先不用 `-1`，而是 miss 给 `0`。原因：

- 操作机器人时 reward 过强负值可能让策略变得保守。
- 当前 actor 还有 `β ||a - ã||²` 约束，已经限制偏离 VLA。
- `0/1` reward 更稳定，便于确认训练闭环是否正确。

后续如果发现 learning signal 不够强，可以改成：

```text
success: +1
miss: -0.5
```

### 可选辅助 reward

等第一版跑通后，再加入 shaping：

```text
+0.2  右 wrist 更靠近瓶盖中心
+0.2  夹爪姿态更对准瓶盖轴线
+0.5  夹住后有有效旋转趋势
-0.01 每个 chunk 时间惩罚
-1.0  撞倒瓶子或左手丢瓶
```

但这些都应该是第二阶段增强，不是第一版必须项。

## 人工按键与关键区域交互

第一版按键协议：

```text
S: start，人工标记关键区域开始，从这一刻开始写入 RLT replay buffer。
E: end，人工标记关键区域结束，从这一刻停止写入 RLT replay buffer。
1: success，在 E 后输入，表示本次关键区域成功。
0: fail，在 E 后输入，表示本次关键区域失败。
```

超时规则：

```text
E 后 10 秒内没有输入 1/0，则默认 reward=0。
```

不再使用 `S=success`、`F=fail` 的旧协议，避免和 `S=start` 混淆。

建议标注逻辑：

```text
默认状态:
    mode = "base_vla"
    不写入 RLT replay buffer

如果操作者按 S:
    mode = "key_region"
    key_region_start_time = now
    开始保存关键区域 transition
    actor 是否接管由当前阶段决定：
        warmup 阶段: actor 不接管，执行 VLA
        RL 阶段: actor 在关键区域接管或修正 VLA

如果操作者按 E:
    mode = "await_score"
    key_region_end_time = now
    停止保存新的关键区域 transition
    等待人工输入 1/0

如果 E 后操作者输入 1:
    terminal_reward = 1
    human_grasp_success = True
    key_region_phase = "success"

如果 E 后操作者输入 0:
    terminal_reward = 0
    human_grasp_success = False
    key_region_phase = "failure"

如果 E 后 10 秒内没有输入:
    terminal_reward = 0
    human_grasp_success = False
    key_region_phase = "score_timeout_default_failure"
```

注意：`S` 和 `E` 控制的是 replay buffer 的时间边界；`1/0` 控制的是 reward。

## Reward 分配到哪个 transition

因为 actor 执行的是 action chunk，不是单个低层动作，所以 reward 也建议按 chunk 记。

硬约束：

```text
非关键区域 chunk 不进入 RLT replay buffer。
非关键区域可以保存到普通 rollout 文件中用于回看和 debug，
但不能作为 actor/critic 训练样本。
```

关键区域内的最简单做法：

```text
一个关键区域 chunk 成功夹住 -> 该 chunk reward = 1
一个关键区域 chunk 没夹住 -> 该 chunk reward = 0
```

更稳定的做法：

```text
关键区域开始后的若干 chunk 都存入 replay。
如果最终 success，则最后导致 success 的 chunk reward=1。
之前 chunk reward=0。
```

第一版推荐：

```text
只训练关键区域内的 chunk。
只给触发 success 的 chunk reward=1。
其他关键区域 chunk reward=0。
非关键区域 chunk 不写入 RLT replay buffer。
```

这样 credit assignment 更清楚。

## 关键区域是什么

关键区域定义：

```text
右臂已经完成大范围移动，进入瓶盖附近，准备对准并闭合夹爪。
```

它不是整个 episode，而是短窗口。

RLT 应该只在这个窗口里接管或修正动作。原因：

- 前面靠近瓶子的粗动作 VLA 已经能做。
- 失败集中在第一次夹瓶盖。
- 在非关键区域训练 RL 可能浪费样本，还可能破坏原本能做好的行为。

## 关键区域如何自动判断

第一版不做自动关键区域判断。关键区域开始和结束由人工按键决定。

硬约束：

```text
S: 关键区域开始。
E: 关键区域结束。
只有 S 到 E 之间的 transition 进入 RLT replay buffer。
```

原因：

- 当前有实时四个摄像头，人工能可靠判断 right wrist 是否进入抓瓶盖阶段。
- 自动判断需要瓶盖定位、右腕位姿、夹爪闭合趋势等额外逻辑，第一版会增加不必要风险。
- RLT 论文允许 human operator 选择何时把控制交给 RL policy，这与人工按 `S` 开始关键区域一致。

下面的自动方案只作为第二版可选自动化方向，不作为第一版实现要求。

### 方案 A：基于右夹爪位置和瓶盖固定位置

如果瓶子和瓶盖在工作台上的位置相对固定，可以人工标定一个瓶盖中心：

```text
cap_position_base = [x, y, z]
```

运行时读取右末端位姿：

```text
right_ee_position
```

当：

```text
distance(right_ee_position, cap_position_base) < d_start
```

进入关键区域。

建议阈值：

```text
d_start = 8 cm 到 12 cm
d_end_success = 人按 S
d_end_timeout = 3 到 5 秒未成功
d_end_far = distance > 15 cm
```

优点：

- 简单稳定。
- 不需要视觉检测。
- 适合瓶子位置固定或左臂固定瓶子位置稳定的场景。

缺点：

- 如果瓶子位置变化大，需要重新标定或估计瓶盖位置。

### 方案 B：基于 VLA 动作中的右夹爪闭合趋势

关键区域通常发生在右夹爪准备闭合前后。可以检测 VLA 或 actor action chunk 中右夹爪维度：

```text
right_gripper_command
```

当未来 action chunk 中出现：

```text
右夹爪从 open -> close
```

就认为即将进入关键区域。

触发条件示例：

```text
max_close_delta_in_next_chunk > threshold
```

优点：

- 不依赖瓶盖位置。
- 直接对应“第一次 close”这个失败点。

缺点：

- 如果 VLA 过早或过晚发 close，关键区域会偏移。
- 只能知道准备夹，不能保证已经靠近瓶盖。

### 方案 C：位姿 + 夹爪闭合组合

第一版最推荐：

```text
right_ee_near_cap == True
AND
right_gripper_will_close == True
```

即：

```text
distance(right_ee, cap_position) < d_start
并且
未来 chunk 里右夹爪有 close 动作
```

这样可以避免：

- 右臂经过瓶盖附近但不是抓取。
- 夹爪闭合但还没到瓶盖。

### 推荐第一版

第一版采用完全人工关键区域边界：

```text
S: 人工开始关键区域
E: 人工结束关键区域
1/0: 关键区域结束后的人工评分
```

自动检测只记录辅助信息，不触发 replay buffer 开始/结束。

## 关键区域结束条件

关键区域开始后，需要自动结束，否则 replay 会混入很多无关动作。

结束条件建议：

```text
1. 人按 E -> 结束关键区域，停止写入 RLT replay buffer，进入等待评分状态。
2. episode done / safety stop -> 强制结束关键区域，reward=0，标记异常结束。
3. 可选 key-region 最大时长超时 -> 强制结束，reward=0，防止误按 S 后无限记录。
```

建议参数：

```text
score_timeout = 10 秒
max_key_region_duration = 10 到 20 秒
```

## 自动判断“夹住瓶盖”的辅助信号

虽然第一版主 reward 用人工按键，但仍建议记录自动信号：

### 夹爪闭合残差

如果命令右夹爪 close，但实际夹爪没有完全闭合，说明中间夹住了东西：

```text
command_close = True
actual_gripper_opening > fully_closed_threshold
```

这可能表示夹住瓶盖。

风险：

- 也可能夹到瓶身、桌面或误碰。

### 右腕旋转带动瓶盖

如果 close 后右腕旋转，瓶盖区域视觉或关节阻力有变化：

```text
right_wrist_rotation_commanded = True
cap_visual_motion_detected = True
```

更可信。

风险：

- 需要视觉检测或额外传感。

### 夹爪接近瓶盖中心

如果右夹爪位置接近瓶盖中心，姿态也对齐：

```text
distance(right_ee, cap_position) < threshold
angle(right_gripper_axis, cap_axis) < threshold
```

可作为 shaping，但不建议直接判 success。

## 为什么第一版不用全自动 reward

全自动 reward 会遇到三个风险：

```text
1. 夹爪状态误判：夹到瓶身也可能看起来像夹住。
2. 视觉误判：腕部遮挡、反光、瓶盖小，检测不稳定。
3. credit 错误：程序以为 success，实际没有夹住，会把错误动作强化。
```

你的问题是 30% miss，信号比较细。第一版应该优先让 reward 准确，而不是自动化。

因此：

```text
人工按键 reward 是第一版主方案。
人工 S/E 负责切出关键区域。
自动检测只记录辅助信息，不决定 reward，也不决定 replay buffer 边界。
```

## 训练流程

### 阶段 1：Warmup

先执行 VLA 动作，不让 actor 接管：

```text
a = ã
```

收集关键区域 transition：

```text
x, ã, a=ã, reward, x'
```

建议：

```text
N_warm = warmup_target 条关键区域样本
默认 warmup_target = 100，但必须可配置；例如你想 90 条后开始 actor 训练，也应该允许。
```

硬约束：

```text
warmup 阶段只执行 VLA 或 human intervention。
actor 不允许接管，不允许输出动作到机器人。
warmup 只负责建立初始 replay buffer，让 critic 先有学习信号。
```

### 阶段 2：Actor 小幅接管

actor 输出 residual：

```text
a = ã + Δa
```

但 `β` 设大，让动作不要偏太远：

```text
β = 5 到 10
```

### 阶段 3：逐步放开

如果 success rate 提高，再降低 `β`：

```text
β = 1 到 5
```

让 actor 更自由地修正 VLA。

## 训练更新公式

TD3-style TD target：

$$
\hat{Q}
=
\sum_{t'=1}^{C} \gamma^{t'-1} r_{t'}
+
\gamma^C
\mathbb{E}_{a' \sim \pi_{\theta}}
\left[
\min_{i=1,2} Q_{\psi_i'}(x', a')
\right]
$$

如果第一版只有 chunk-level reward：

```text
reward_seq = [0, 0, ..., reward]
```

critic loss：

$$
\mathcal{L}_Q(\psi_1,\psi_2)
=
\mathbb{E}_{b}
\left[
\left(
\hat{Q} - Q_{\psi_1}(x, a)
\right)^2
+
\left(
\hat{Q} - Q_{\psi_2}(x, a)
\right)^2
\right]
$$

actor loss：

$$
\mathcal{L}_{\pi}(\theta)
=
\mathbb{E}_{b}
\left[
-Q_{\psi_1}(x, a)
+
\beta \left\|a - \tilde{a}\right\|_2^2
\right]
$$

其中 `z_rl`、VLA、RL Token extractor 全部冻结。`Q_{\psi_1'}` 和 `Q_{\psi_2'}` 是 target critic，使用 EMA/Polyak update 从在线 critic 更新：

$$
\psi_i' \leftarrow \tau \psi_i + (1-\tau)\psi_i'
$$

第一版建议：

```text
τ = 0.005
policy_delay = 2
target_policy_noise = 可先不加；如果训练不稳定，再加 clipped target noise
```

## 文件规划

建议新增：

```text
src/openpi/models/rlt.py
src/openpi/models/rlt_config.py
src/openpi/training/rlt_replay_buffer.py
src/openpi/training/rlt_data.py
scripts/train_rlt.py
examples/aloha_real/rlt_reward.py
examples/aloha_real/rlt_rollout.py
```

职责：

```text
rlt.py:
    RLTActor
    RLTCritic
    RLTActorCritic

rlt_config.py:
    actor/critic hidden dim、action horizon、state dim、beta、gamma

rlt_replay_buffer.py:
    存 transition
    sample batch

rlt_data.py:
    把 rollout 数据转成训练 batch

train_rlt.py:
    加载 frozen VLA + RL token
    加载 actor/critic
    更新 Q 和 actor

rlt_reward.py:
    人工 S/E 控制关键区域开始和结束
    E 后 1/0 人工评分
    10 秒评分超时默认失败
    自动辅助信号

rlt_rollout.py:
    在线 rollout
    VLA 参考动作
    actor 修正动作
    记录 transition
```

## 第一版最小实现顺序

### Step 1：只做离线训练管线

先不接真实机器人在线训练。用保存的关键区域 transition 文件做：

```text
load key-region transition -> train critic -> train actor
```

注意：如果同时保存整条 episode 的视频或 HDF5，它们只能作为调试数据；`train_rlt.py` 默认只读取关键区域 replay buffer。

### Step 2：做人工关键区域和 reward 采集

在实时 rollout 中加入按键：

```text
S key region start
E key region end
1 success after E
0 fail after E
10 秒无评分 -> 默认 fail
```

### Step 3：Warmup 收集 warmup_target 条关键区域样本

warmup 阶段：

```text
非关键区域：执行 VLA，不写入 RLT replay buffer
S 到 E：执行 VLA，写入 RLT replay buffer
E 后：人工 1/0 评分，10 秒无评分默认 0
actor：不接管
目标：收集 warmup_target 条关键区域样本
```

### Step 4：上线 actor

warmup_count >= warmup_target 并且 critic/actor 训练流程验证后，才允许 actor 在关键区域接管：

```text
非关键区域：执行 VLA
关键区域 S 到 E：执行 actor(VLA action + residual)
关键区域外：actor 不输出到机器人
```

## 风险和处理

### 风险 1：人工评分延迟或忘记评分

人按 `E` 后可能忘记输入 `1/0`。

处理：

```text
E 后启动 10 秒倒计时。
10 秒内没有输入 1/0，则默认 reward=0。
metadata 标记 score_timeout_default_failure=True。
```

### 风险 2：人工 S/E 边界不准

处理：

```text
保存 S/E 前后短窗口到普通日志，便于回看。
RLT replay buffer 仍只用 S 到 E 的 transition。
后续可以用日志分析是否需要自动边界辅助。
```

### 风险 3：actor 改坏 VLA 原本正确动作

处理：

```text
只在关键区域启用 actor。
使用 β ||a - ã||²。
限制 Δa 最大幅度。
保留 safety stop。
```

### 风险 4：0/1 reward 稀疏

处理：

```text
先收集足够关键区域样本。
后续加入距离/姿态 shaping。
```

## 推荐第一版决策

第一版不要追求全自动。建议：

```text
关键区域边界：人工 S/E 为准
reward：E 后人工 1/0 为准
评分超时：E 后 10 秒无评分默认 reward=0
自动夹爪判断：只记录，不作为主 reward
warmup：先收集 warmup_target 条关键区域样本，actor 不接管；warmup_target 默认 100，但可配置
训练目标：提高第一次夹住瓶盖成功率
actor 接管范围：warmup 后，只在 S 到 E 的关键区域
```

这样最符合当前问题，也最容易 debug。
