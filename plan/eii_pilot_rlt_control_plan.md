# EII Pilot RLT 第二阶段控制台改造规划

本文档只做实施规划。你确认后，再开始写代码。

## 目标

当前 EII Pilot 中的 voice、talk、Robot Viewer 对 RLT 第二阶段训练没有价值，需要删除。新的 EII Pilot 首页应该服务于 RLT critical phase 数据采集、warmup、actor/critic 在线训练和人工 reward 标注。

核心目标：

```text
1. 删除 voice/talk 前端和后端。
2. 删除 Robot Viewer 视图。
3. 保留实时摄像头显示，尤其 right wrist 视角。
4. 新增 RLT 控制按钮：S/E/1/0，并绑定键盘。
5. 只采集 S 到 E 之间的关键区域 transition。
6. E 后人工输入 1/0 作为 terminal reward。
7. E 后 10 秒内不评分，默认 reward=0。
8. warmup 阶段先收集一批关键区域数据，actor 不允许接管；默认目标 100 条，但必须可配置，例如 90 条也可以解锁 actor。
9. warmup 后允许 actor 在关键区域介入，并支持实时调节 beta 和介入程度。
10. actor/critic loss 通过 wandb 记录和显示。
```

## 当前代码结构判断

当前 EII Pilot 代码在：

```text
voice_assistant_web/
```

虽然目录名叫 `voice_assistant_web`，但当前它实际承担了 EII Pilot 前端和后端功能。

前端：

```text
voice_assistant_web/frontend/src/App.tsx
voice_assistant_web/frontend/src/components/CameraGrid.tsx
voice_assistant_web/frontend/src/components/VoicePanel.tsx
voice_assistant_web/frontend/src/components/RobotViewer.tsx
voice_assistant_web/frontend/src/components/RolloutBrowser.tsx
voice_assistant_web/frontend/src/services/api.ts
voice_assistant_web/frontend/src/styles.css
voice_assistant_web/frontend/src/i18n.ts
```

后端：

```text
voice_assistant_web/backend/app/main.py
voice_assistant_web/backend/app/schemas.py
voice_assistant_web/backend/app/config.py
voice_assistant_web/backend/app/redis_commands.py
voice_assistant_web/backend/app/voice_session.py
voice_assistant_web/backend/app/camera_bridge.py
voice_assistant_web/backend/app/robot_state_bridge.py
```

Docker Compose：

```text
docker-compose.yml
```

当前前端 `App.tsx` 的 live 页面包含：

```text
CameraGrid
RobotViewer
VoicePanel
```

当前后端 `main.py` 暴露了：

```text
/ws/realtime
/api/cameras/{camera_name}/latest.jpg
/api/cameras/{camera_name}/stream.mjpg
/api/rollouts/tree
/api/rollouts/video
/api/voice/text
/api/voice/audio
/api/tasks/{task_number}
```

其中实时相机和 rollout 浏览可以保留；voice 和 task dispatch 要删除或替换。

## 删除范围

### 前端删除

删除或停止引用：

```text
voice_assistant_web/frontend/src/components/VoicePanel.tsx
voice_assistant_web/frontend/src/components/RobotViewer.tsx
voice_assistant_web/frontend/src/lib/loaders/
```

`src/lib/loaders/` 只服务于 Robot Viewer 的 URDF/Three.js 加载，删除 Robot Viewer 后也应该删除。

`App.tsx` 中删除：

```text
import { RobotViewer } from './components/RobotViewer'
import { VoicePanel } from './components/VoicePanel'
TASK_NUMBERS
dispatchTask
dispatchError
1-6 数字键触发任务的键盘逻辑
右侧 control-rail 中 RobotViewer 和 VoicePanel
```

`styles.css` 中删除或重写：

```text
.voice-*
.robot-*
.talk-*
旧 command/task button 相关样式
```

保留：

```text
CameraGrid
RolloutBrowser
apiBase/wsBase
```

### 后端删除

删除或停止引用：

```text
voice_assistant_web/backend/app/voice_session.py
VoiceAssistantEngine
VoiceRequest
VoiceResponse
/api/voice/text
/api/voice/audio
OpenAI voice/chat/tts/transcription 配置
```

删除或替换：

```text
/api/tasks/{task_number}
TASK_MAPPING
publish_task
VOICE_COMMAND_CHANNEL
aloha_voice_commands
```

注意：如果 runtime 当前仍监听 `aloha_voice_commands` 来执行旧任务，则不能简单删除 Redis 通道本身，而应该把它替换为新的 RLT 控制通道。实现时需要同步改 runtime，否则前端按钮无法影响机器人流程。

### Docker/README 命名

当前 docker compose 服务名是：

```text
voice_web_backend
voice_web_frontend
```

第一版可以先不改服务名，降低改动风险；但页面、API、README 应去 voice 化。

第二版可以改名为：

```text
eii_pilot_backend
eii_pilot_frontend
```

建议第一版先保留服务名，避免 docker compose、镜像名、端口和脚本产生无关 churn。

## 新页面布局

EII Pilot 首页不做 marketing 页面，直接进入操作台。

建议布局：

```text
顶部状态栏:
    EII Pilot
    runtime 状态
    当前阶段: idle / warmup / actor_enabled / scoring
    实时连接状态

左侧主区域:
    四摄像头实时画面
    right wrist 视角应可放大/置顶

右侧控制区:
    RLT Key Region Controls
    Warmup Progress
    Actor Intervention
    Training Metrics Summary
```

保留 Rollouts 页面作为第二 tab：

```text
Live RLT
Rollouts
```

Rollouts 页面仍用于查看已采集视频，不参与 RLT 控制。

## RLT 控制按钮

页面必须提供四个大按钮：

```text
S  Start Key Region
E  End Key Region
1  Success
0  Failure
```

键盘绑定：

```text
按 S -> 调用 start key region
按 E -> 调用 end key region
按 1 -> 提交 success reward
按 0 -> 提交 failure reward
```

键盘规则：

```text
如果焦点在 input/textarea/select/contenteditable 内，不触发快捷键。
如果 ctrl/meta/alt 被按下，不触发快捷键。
重复按键 event.repeat 不触发重复请求。
```

按钮状态机：

```text
idle:
    S 可用
    E 禁用
    1/0 禁用

key_region:
    S 禁用
    E 可用
    1/0 禁用

await_score:
    S 禁用
    E 禁用
    1/0 可用
    显示 10 秒倒计时

score_submitted:
    短暂显示结果
    回到 idle
```

异常处理：

```text
重复按 S: 后端拒绝，提示 already in key region。
未 S 直接 E: 后端拒绝，提示 no active key region。
未 E 直接 1/0: 后端拒绝，提示 not awaiting score。
E 后 10 秒无 1/0: 后端自动写 reward=0。
```

## Warmup 数据显示

页面显示：

```text
warmup_target = 可配置目标条数，默认 100
warmup_count = 已完成评分的关键区域条数
warmup_success = reward=1 的条数
warmup_failure = reward=0 的条数
warmup_remaining = max(warmup_target - warmup_count, 0)
```

warmup 阶段硬约束：

```text
actor_enabled = false
actor 不允许输出动作到机器人
S 到 E 之间仍执行 VLA reference action
只把 S 到 E 的 transition 写入 RLT replay buffer
```

页面提示：

```text
Warmup: 37 / warmup_target
Actor takeover: locked during warmup
```

当 `warmup_count >= warmup_target` 后：

```text
允许开启 actor 接管
默认仍不自动开启，需要人手动打开 Enable Actor
```

硬约束：

```text
后端不能把 warmup_target 写死为 100。
warmup_target 必须是 RLT runtime/config 状态的一部分。
前端应提供 Warmup Target numeric input。
修改 warmup_target 后，actor 解锁条件立即按新值重新计算。
```

## 自动 rollout 轨迹显示

页面显示：

```text
auto_rollout_count
auto_rollout_success
auto_rollout_failure
last_auto_rollout_reward
```

定义：

```text
warmup rollout:
    actor_enabled=false 时采集的关键区域样本

auto rollout:
    warmup 完成后，actor_enabled=true 时采集的关键区域样本
```

注意：这里的 auto rollout 不是整条 episode，而是 actor 参与后的关键区域 rollout。

## Beta 实时配置

`beta` 是 actor loss 中约束 actor 不偏离 VLA reference action 的系数：

$$
\mathcal{L}_{\pi}
=
\mathbb{E}_{b}
\left[
-Q_{\psi_1}(x,a)
+
\beta \|a-\tilde{a}\|_2^2
\right]
$$

页面需要：

```text
显示当前 beta
提供 slider
提供 numeric input
点击 Apply 或失焦后提交
显示最后更新时间
```

建议范围：

```text
min_beta = 0
max_beta = 20
default_beta = 10
step = 0.1
```

解释：

```text
beta 越大:
    actor 越贴近 VLA，动作更保守。

beta 越小:
    actor 更自由，可能提高成功率，也更可能破坏 VLA 原本动作。
```

## Actor 介入程度

除了 `beta`，还需要单独控制 actor 对最终动作的实际影响。建议增加：

```text
actor_enabled: 是否允许 actor 接管
intervention_scale: actor residual 缩放系数
max_delta: actor 每维最大动作修正幅度
```

动作组合：

```text
a_actor = πθ(x, ã)
delta = clip(a_actor - ã, -max_delta, max_delta)
a_execute = ã + intervention_scale * delta
```

页面控件：

```text
Enable Actor: toggle
Intervention Scale: slider [0, 1], default 0.25
Max Delta: numeric input, default 0.1 或按 action normalization 定义
```

阶段限制：

```text
warmup_count < warmup_target:
    Enable Actor 强制 false
    Intervention Scale 只读

warmup_count >= warmup_target:
    可以启用 actor
```

这比只调 beta 更直接，因为 beta 影响训练目标，`intervention_scale` 影响在线执行安全性。

## 后端 API 设计

新增 RLT control schema：

```python
RLTControlState:
    phase: "idle" | "key_region" | "await_score"
    training_phase: "warmup" | "rl"
    warmup_target: int
    warmup_count: int
    warmup_success: int
    warmup_failure: int
    auto_rollout_count: int
    auto_rollout_success: int
    auto_rollout_failure: int
    actor_enabled: bool
    actor_locked_reason: str | None
    beta: float
    intervention_scale: float
    max_delta: float
    active_key_region_id: str | None
    score_deadline: float | None
    last_reward: int | None
    last_event: str | None
```

新增接口：

```text
GET  /api/rlt/status
POST /api/rlt/key-region/start
POST /api/rlt/key-region/end
POST /api/rlt/key-region/score
POST /api/rlt/config
```

接口语义：

```text
POST /api/rlt/key-region/start
    body: optional {source: "ui" | "keyboard"}
    effect:
        phase idle -> key_region
        publish Redis event rlt_key_region_start

POST /api/rlt/key-region/end
    effect:
        phase key_region -> await_score
        score_deadline = now + 10
        publish Redis event rlt_key_region_end

POST /api/rlt/key-region/score
    body: {reward: 0 | 1}
    effect:
        phase await_score -> idle
        publish Redis event rlt_score
        update warmup/auto counts

POST /api/rlt/config
    body:
        beta?: float
        actor_enabled?: bool
        intervention_scale?: float
        max_delta?: float
    effect:
        validate constraints
        publish Redis event rlt_config_update
```

`/ws/realtime` 应扩展 payload：

```text
robot
camera_status
camera_timestamps
camera_jpeg_b64
rlt
```

这样前端不用轮询 `/api/rlt/status`，实时页面直接刷新。

## Redis / runtime 通信设计

当前 runtime 通过 Redis 接收旧 voice task。新方案应改为 RLT 控制通道。

建议通道：

```text
RLT_CONTROL_CHANNEL = "aloha_rlt_control"
RLT_STATE_CHANNEL = "aloha_rlt_state"
```

前端后端发布控制事件：

```json
{
  "type": "key_region_start",
  "timestamp": 123.0,
  "source": "ui"
}
```

```json
{
  "type": "key_region_end",
  "timestamp": 124.0,
  "key_region_id": "..."
}
```

```json
{
  "type": "score",
  "timestamp": 130.0,
  "reward": 1,
  "key_region_id": "..."
}
```

```json
{
  "type": "config_update",
  "timestamp": 131.0,
  "beta": 8.0,
  "actor_enabled": true,
  "intervention_scale": 0.25,
  "max_delta": 0.1
}
```

runtime 发布状态：

```json
{
  "phase": "key_region",
  "training_phase": "warmup",
  "warmup_count": 37,
  "auto_rollout_count": 0,
  "actor_enabled": false,
  "beta": 10.0,
  "intervention_scale": 0.0,
  "critic_loss": 0.12,
  "actor_loss": 0.03
}
```

## 关键区域数据保存

RLT replay buffer 只保存：

```text
S 到 E 之间的 transition
```

保存字段：

```python
{
    "key_region_id": str,
    "episode_id": str,
    "chunk_idx": int,
    "timestamp_start": float,
    "timestamp_end": float,
    "z_rl": float32[2048],
    "proprio": float32[32],
    "vla_action": float32[50, 32],
    "actor_action": float32[50, 32],
    "executed_action": float32[50, 32],
    "reward": float32,
    "is_terminal": bool,
    "next_z_rl": float32[2048],
    "next_proprio": float32[32],
    "metadata": {
        "training_phase": "warmup" | "rl",
        "actor_enabled": bool,
        "beta": float,
        "intervention_scale": float,
        "max_delta": float,
        "human_score": 0 | 1,
        "score_timeout": bool,
    },
}
```

E 后评分写法：

```text
关键区域内非 terminal transition reward=0。
最后一个 transition:
    reward=1 if score == 1
    reward=0 if score == 0 or timeout
    is_terminal=True
```

## wandb 指标

actor/critic 训练脚本必须记录到 wandb：

```text
rlt/critic_loss
rlt/critic_q1_loss
rlt/critic_q2_loss
rlt/actor_loss
rlt/actor_q_value
rlt/beta
rlt/intervention_scale
rlt/max_delta
rlt/warmup_count
rlt/auto_rollout_count
rlt/reward_mean
rlt/success_rate_window
rlt/replay_size
rlt/q1_mean
rlt/q2_mean
rlt/q_min_mean
rlt/actor_delta_norm
rlt/reference_deviation
```

EII Pilot 页面不需要自己画完整 loss 曲线。更稳定的做法：

```text
页面显示 wandb run 链接和最新几个标量摘要。
完整 actor/critic loss 曲线在 wandb 查看。
```

如果要在页面嵌入曲线，建议后续再做，因为 wandb 嵌入和鉴权会增加复杂度。

## 前端组件规划

新增：

```text
voice_assistant_web/frontend/src/components/RLTControlPanel.tsx
voice_assistant_web/frontend/src/components/RLTStatsPanel.tsx
voice_assistant_web/frontend/src/components/RLTConfigPanel.tsx
```

职责：

```text
RLTControlPanel:
    S/E/1/0 按钮
    键盘快捷键
    10 秒评分倒计时
    当前 phase 显示

RLTStatsPanel:
    warmup_count / warmup_target
    warmup success/failure
    auto rollout count
    last reward
    replay size

RLTConfigPanel:
    beta slider + numeric input
    actor_enabled toggle
    intervention_scale slider
    max_delta input
    warmup_target numeric input
    warmup 未完成时锁定 actor_enabled
```

`App.tsx` 改成：

```text
CameraGrid + RLT side panel
```

不再有：

```text
RobotViewer
VoicePanel
talk button
task number buttons
```

## 后端文件规划

新增：

```text
voice_assistant_web/backend/app/rlt_control.py
```

职责：

```text
维护 RLTControlState
处理 S/E/1/0
处理 beta/actor/intervention config
处理 E 后 10 秒超时默认失败
发布 Redis 控制事件
订阅 runtime RLT 状态
```

修改：

```text
voice_assistant_web/backend/app/main.py
    删除 voice endpoints
    删除 task endpoint
    注册 rlt endpoints
    websocket payload 增加 rlt 状态

voice_assistant_web/backend/app/schemas.py
    删除 VoiceRequest/VoiceResponse
    新增 RLT schemas

voice_assistant_web/backend/app/config.py
    删除 OpenAI voice 配置
    新增 RLT Redis channel 配置
```

删除：

```text
voice_assistant_web/backend/app/voice_session.py
```

## runtime 侧改造规划

仅改前后端还不够。按钮必须影响真实数据采集和 actor 接管，所以 runtime 也要支持 RLT 控制事件。

需要修改：

```text
packages/openpi-client/src/openpi_client/runtime/runtime.py
examples/aloha_real/main.py
```

runtime 新状态：

```text
rlt_phase = idle | key_region | await_score
training_phase = warmup | rl
warmup_target = 可配置，默认 100，不能后端写死
warmup_count
auto_rollout_count
actor_enabled
beta
intervention_scale
max_delta
```

runtime 行为：

```text
收到 key_region_start:
    开始把 transition 写入 RLT replay buffer
    如果 warmup_count < warmup_target:
        actor_enabled_effective = false
    否则:
        actor_enabled_effective = user actor_enabled

收到 key_region_end:
    停止追加新的 transition
    等待 score

收到 score:
    给最后一个关键区域 transition 写 terminal reward
    更新 warmup/auto count
    保存 replay buffer
```

actor 接管规则：

```text
非关键区域:
    always execute VLA

关键区域 + warmup:
    execute VLA
    save transition

关键区域 + rl + actor_enabled:
    execute VLA + intervention_scale * actor_delta
    save transition
```

## 不足与补充建议

### 1. 需要一个明确的 RLT run/session id

每次开始 RLT 采集/训练，需要生成：

```text
rlt_session_id
```

用于关联：

```text
replay buffer
wandb run
rollout videos
key region labels
actor/critic checkpoint
```

### 2. 需要安全锁

页面应显示：

```text
Actor locked during warmup
Actor disabled
Actor enabled only in key region
```

后端也必须 enforce，不能只靠前端禁用按钮。

### 3. 需要 Undo Last Label

人工评分可能误按。建议新增：

```text
Undo Last Score
```

第一版可以先只做后端接口和页面按钮，但默认需要确认弹窗。

### 4. 需要事件日志

页面显示最近事件：

```text
12:01:03 S start key region
12:01:08 E end key region
12:01:10 score=1
12:01:10 warmup 38/warmup_target
```

这对调试非常重要。

### 5. 需要状态持久化

后端重启后不应该丢失计数。建议保存：

```text
/app/rollouts/rlt_sessions/<session_id>/control_state.json
```

至少保存：

```text
warmup_count
auto_rollout_count
last key_region_id
beta
intervention_scale
max_delta
```

### 6. 需要区分 UI 状态和 runtime 状态

前端按钮按下后后端状态变化，不等于 runtime 已经执行。建议状态里保留：

```text
backend_phase
runtime_phase
last_runtime_ack_time
```

第一版可以先简化，但最终最好加 ack，避免误以为已经开始记录。

## 实施顺序

### Step 1：前端去 voice / robot viewer

```text
删除 VoicePanel、RobotViewer 引用。
保留 CameraGrid 和 Rollouts。
新增 RLT side panel 静态 UI。
```

### Step 2：后端去 voice endpoints

```text
删除 /api/voice/text
删除 /api/voice/audio
删除 VoiceAssistantEngine
删除 OpenAI voice 配置
```

### Step 3：新增 RLT control API

```text
GET /api/rlt/status
POST /api/rlt/key-region/start
POST /api/rlt/key-region/end
POST /api/rlt/key-region/score
POST /api/rlt/config
```

### Step 4：前端按钮接 API 和键盘

```text
S/E/1/0 按钮可用
键盘可用
状态机正确
10 秒倒计时正确
```

### Step 5：runtime 接 Redis RLT 控制事件

```text
接收 S/E/1/0/config
只保存 S 到 E transition
warmup_count < warmup_target 时 actor 不接管
```

### Step 6：actor/critic 训练与 wandb

```text
训练脚本记录 actor/critic loss 到 wandb
页面展示 wandb run link 和最新摘要
```

## 验收标准

确认实现完成时，应满足：

```text
1. EII Pilot 页面没有 voice/talk 入口。
2. 页面没有 Robot Viewer。
3. S/E/1/0 按钮和键盘都能工作。
4. S 后开始关键区域，E 后结束关键区域。
5. E 后 1/0 能评分。
6. E 后 10 秒无评分自动 reward=0。
7. warmup_count 正确增加。
8. warmup_count < warmup_target 时 actor 无法接管。
9. warmup_count >= warmup_target 后可以手动启用 actor。
10. beta 可以实时修改，并传到 runtime/训练进程。
11. intervention_scale 可以实时修改，并影响 actor residual 执行幅度。
12. actor/critic loss 进入 wandb。
13. RLT replay buffer 只包含 S 到 E 的关键区域 transition。
14. Rollouts 视频浏览仍可用。
```
