# 异步 RLT KeyRegionReplayRecorder 设计

本文档基于当前代码、`plan/rlt_actor_critic_design.md`、`plan/eii_pilot_rlt_control_plan.md`，以及两项只读研究结论，设计一个用于 RLT 第二阶段 warmup/online RL 的关键区域异步记录系统。

目标：

- 机器人 50Hz 控制循环中低开销记录关键区域数据，避免因为落盘、视频编码、HDF5 写入导致机械臂卡顿。
- `S/E/score` 事件能够真正切分关键区域 rollout。
- 每个关键区域单独保存 MP4 视频，前端能复用现有 RolloutBrowser 播放。
- actor/critic 训练进程能够异步读取 replay buffer，不间断训练。
- 保存格式直接服务 `scripts/train_rlt.py` / `src/openpi/training/rlt_training.py` 需要的 RLT batch contract。

## 当前代码结论

### 现有 RLT 事件只更新状态，没有保存关键区域数据

当前前后端已有 RLT 控制事件：

- `POST /api/rlt/key-region/start`
- `POST /api/rlt/key-region/end`
- `POST /api/rlt/key-region/score`

后端位置：

- `voice_assistant_web/backend/app/rlt_control.py`
- `voice_assistant_web/backend/app/main.py`

runtime 监听 Redis 后只更新 `_rlt_state`：

- `packages/openpi-client/src/openpi_client/runtime/runtime.py`

现在缺少：

- `start_step`
- `end_step`
- key region 内每一步的 `z_rl`
- `proprio`
- 实际执行 `action`
- VLA reference action `reference_action`
- `reward_seq`
- `next_z_rl`
- `next_proprio`
- `next_reference_action`
- `done`
- 对应短轨迹视频和 metadata

### 当前 VideoHdf5Saver 可复用写 MP4/HDF5，但不能直接用于 RLT replay

当前长轨迹保存：

- `examples/aloha_real/video_hdf5_saver.py`

可复用：

- `FfmpegH264Writer`
- H.264 MP4 外置视频格式
- `episode.hdf5 + cam_*.mp4` 目录结构
- HDF5 metadata 写法

不能直接复用：

- `_on_step_split_on_reset()`：它按离开 home / 回到 home 切长轨迹，不适合人工 `S/E` 关键区域。
- `_append_step()`：它在 `on_step()` 内同步写 MP4 pipe，有卡顿风险。
- `_finalize_current_episode()`：它保存的是普通 episode metadata，不包含 RLT replay 训练所需字段。

### 当前 RolloutBrowser 和后端视频 API 可以复用

前端 `RolloutBrowser` 只依赖：

- `/api/rollouts/tree`
- `/api/rollouts/video?path=...`
- 文件扩展名 `.mp4`

后端 `/api/rollouts/video` 已支持 Range streaming，浏览器播放和拖动进度条的方式是合理的。

因此关键区域短视频只要保存到：

```text
/data/openpi0.5-rtc-reward-learning/rollouts/key_regions/...
```

并且每个相机保存为 `.mp4`，前端即可浏览。后续可以增加 `rootPath=key_regions` 做单独页签，但不是第一阶段必要条件。

## 总体架构

建议新增三个模块：

```text
examples/aloha_real/rlt_key_region_recorder.py
src/openpi/training/rlt_replay_store.py
scripts/train_rlt_online.py
```

职责划分：

```text
runtime 50Hz loop
  |
  | on_step 只做轻量入内存 ring buffer
  v
KeyRegionReplayRecorder
  |
  | S/E/score 事件切片
  v
background writer thread
  |
  | 写 replay shard npz/jsonl
  | 写短轨迹 mp4/hdf5/manifest
  v
rollouts/key_regions/...              replay/rlt_key_regions/...
  |                                    |
  | 前端 RolloutBrowser 查看视频       | actor/critic online trainer 轮询读取
```

核心原则：

- `on_step()` 不做磁盘 IO。
- `on_step()` 不写 ffmpeg stdin。
- `on_step()` 不创建目录、不写 HDF5、不做大数组压缩。
- 关键区域结束后由后台线程写视频和 replay。
- replay 训练数据和视频调试数据分开保存。

## Subscriber 事件接口

当前 `Subscriber` 只有：

```python
on_episode_start()
on_step(observation, action)
on_episode_end(episode_subdir=None)
```

建议扩展为可选 hook，保持向后兼容：

```python
def on_key_region_start(self, event: dict) -> None:
    pass

def on_key_region_end(self, event: dict) -> None:
    pass

def on_key_region_score(self, event: dict) -> None:
    pass
```

runtime 在 `_handle_rlt_control_event()` 中收到 Redis 事件后，除更新 `_rlt_state` 外，还应转发给 subscribers：

```python
if event_type == "key_region_start":
    for subscriber in self._subscribers:
        hook = getattr(subscriber, "on_key_region_start", None)
        if hook:
            hook(data)

elif event_type == "key_region_end":
    for subscriber in self._subscribers:
        hook = getattr(subscriber, "on_key_region_end", None)
        if hook:
            hook(data)

elif event_type == "score":
    for subscriber in self._subscribers:
        hook = getattr(subscriber, "on_key_region_score", None)
        if hook:
            hook(data)
```

事件 payload 至少包含：

```json
{
  "type": "key_region_start",
  "timestamp": 1760000000.0,
  "key_region_id": "...",
  "state": {
    "training_phase": "warmup",
    "warmup_count": 12,
    "warmup_target": 90,
    "actor_enabled": false,
    "actor_effective": false,
    "beta": 10.0,
    "intervention_scale": 0.25,
    "max_delta": 0.1
  }
}
```

score 事件：

```json
{
  "type": "score",
  "timestamp": 1760000005.0,
  "key_region_id": "...",
  "reward": 1,
  "score_timeout": false,
  "source": "ui",
  "state": {
    "training_phase": "warmup"
  }
}
```

注意：`key_region_id` 必须由后端生成并随 start/end/score 全链路传递。当前后端 `score` 已带 `key_region_id`，但 start/end publish 时应显式带上，避免 runtime 从 state 推断失败。

## Ring Buffer 设计

### 为什么需要 ring buffer

人工按 `←` 开始关键区域时，用户可能晚按 100 到 300ms。对于夹瓶盖任务，关键动作很短，如果没有 pre-roll，可能丢掉接近瓶盖前几帧。因此 recorder 应持续保存最近若干秒的轻量 step 到内存 ring buffer。

推荐：

```text
fps = 50
pre_roll_seconds = 2.0
max_key_region_seconds = 20.0
post_roll_seconds = 0.3
ring_seconds = pre_roll + max_key_region + post_roll + safety
ring_seconds = 24 到 30 秒
```

50Hz 下 30 秒约 1500 steps。只存轻量字段和必要图像引用/拷贝，内存可控。

### StepRecord 字段

`on_step()` 每一步生成 `StepRecord`：

```python
@dataclass
class StepRecord:
    step_index: int
    timestamp: float
    qpos: np.ndarray
    qvel: np.ndarray
    effort: np.ndarray
    proprio: np.ndarray
    action: np.ndarray
    reference_action: np.ndarray
    z_rl: np.ndarray
    images: dict[str, np.ndarray] | None
    runtime_state: dict
```

字段来源：

- `qpos/qvel/effort/images` 来自 `observation["origin_observation"]`。
- `action` 来自实际执行的 `action["actions"]`。
- `reference_action` 来自 VLA reference chunk。当前 runtime/ActionChunkBroker 需要暴露这个字段；如果 actor 未接管，`reference_action == action chunk`。
- `z_rl` 来自 policy/RL Token 网络输出。当前 action 返回结构如果没有 `z_rl`，需要在 policy server 或 action broker 返回 metadata。
- `proprio` 建议直接保存模型侧使用的 proprio 表示，不能只保存 raw qpos；否则训练时还要重复 transform，容易不一致。

如果第一阶段还没有打通 `z_rl/reference_action` 输出，可以先保存 raw 字段和视频，但该数据不能直接进入 actor/critic 训练。

### on_step 轻量操作

`KeyRegionReplayRecorder.on_step()` 只做：

1. 读取当前时间和递增 step index。
2. 将 numpy 数组转为 `np.asarray(..., dtype=np.float32)`。
3. 对低维数据做 copy，避免后续被环境复用修改。
4. 对图像只在两种策略中二选一：
   - 策略 A：只在 key region active 时 copy 图像到 active buffer。
   - 策略 B：ring buffer 保存 JPEG/MP4 所需 raw RGB 图像引用的浅拷贝不可取；安全起见要 copy，但只保留最近 2 秒 pre-roll。
5. append 到 `collections.deque(maxlen=ring_steps)`。
6. 如果当前处于 active key region，将 step index 追加到 active segment。

不能在 `on_step()` 做：

- HDF5 写入。
- MP4 编码。
- ffmpeg stdin write。
- npz 压缩。
- S3 上传。
- 大量 JSON 写入。
- 数据集目录扫描。

## Key Region 状态机

Recorder 内部状态：

```text
idle
  |
  | on_key_region_start
  v
recording
  |
  | on_key_region_end
  v
await_score
  |
  | on_key_region_score
  v
enqueue_write -> idle
```

### on_key_region_start

行为：

- 记录 `key_region_id`
- 记录 `start_timestamp`
- 从 ring buffer 中回溯 `pre_roll_seconds` 的 step，作为 segment 前缀
- 设置 `recording=True`
- 创建 `ActiveSegment`，但不创建磁盘目录

### on_key_region_end

行为：

- 记录 `end_timestamp`
- 将状态设为 `await_score`
- 可以继续额外收集 `post_roll_seconds`，用于视频查看
- 不 finalize，因为 reward 还没来

### on_key_region_score

行为：

- 读取 reward 0/1
- 记录 `score_timeout`
- 冻结 active segment 的 step list
- 将写任务放入 `queue.Queue`
- 清空 active segment，状态回 `idle`

如果 `score_timeout=True`，reward 固定为 0。

## 后台 Writer 线程

### 线程结构

```python
class KeyRegionReplayRecorder(Subscriber):
    def __init__(...):
        self._ring = deque(maxlen=ring_steps)
        self._active = None
        self._write_queue = queue.Queue(maxsize=8)
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()
```

`maxsize=8` 是重要限流。如果 writer 明显落后，说明磁盘或编码跟不上，应在前端/日志报警。

### WriteJob 字段

```python
@dataclass
class WriteJob:
    key_region_id: str
    task: str
    phase: Literal["warmup", "rl"]
    reward: int
    score_timeout: bool
    start_time: float
    end_time: float
    score_time: float
    records: list[StepRecord]
    config_snapshot: dict
```

### 后台写入内容

每个 WriteJob 写两个目录：

```text
/data/openpi0.5-rtc-reward-learning/rollouts/key_regions/...
/data/openpi0.5-rtc-reward-learning/replay/rlt_key_regions/...
```

视频展示目录：

```text
rollouts/key_regions/unscrew_bottle_cap/2026-05-29/warmup/
  key_region_000012_<id>_success/
    manifest.json
    episode.hdf5
    cam_high.mp4
    cam_low.mp4
    cam_left_wrist.mp4
    cam_right_wrist.mp4
```

训练 replay 目录：

```text
replay/rlt_key_regions/unscrew_bottle_cap/2026-05-29/
  manifest.jsonl
  shards/
    shard_000000.npz
    shard_000001.npz
```

建议 replay shard 不是每个 key region 一个长期文件，而是每个 key region 先写一个临时 `.npz.tmp`，完成后原子 rename 为 `.npz`。训练进程只读取已完成的 `.npz`。

```text
key_region_<index>_<id>.npz.tmp
key_region_<index>_<id>.npz
```

## RLT replay transition 生成

`scripts/train_rlt.py` 当前要求单个 `.npz` 包含：

```text
z_rl
proprio
action
reference_action
reward_seq
next_z_rl
next_proprio
next_reference_action
done
```

在线训练建议改成可读取目录下多个 shard，但每个 shard 内仍保持同样字段。

### Chunk 切分

论文和当前 RLT 设计是 action chunk transition。建议参数：

```text
action_horizon = C
stride = 2
```

对一个关键区域 segment，生成 transition：

```text
for i in range(0, T - C - 1, stride):
    x_i = (z_rl[i], proprio[i])
    action_i = executed_actions[i : i+C]
    reference_i = reference_actions[i : i+C]
    reward_seq_i = zeros(C)
    next_x_i = (z_rl[i+C], proprio[i+C])
    next_reference_i = reference_actions[i+C : i+2C]
    done_i = False
```

终止 reward 处理：

- 成功/失败打分是关键区域级别的 terminal reward。
- 推荐将 reward 放到最后一个 chunk 的最后一位：

```text
reward_seq[-1] = reward
done = True
```

- 非 terminal 中间 chunk 的 `reward_seq` 全 0，`done=False`。

如果关键区域太短，不足 `C + 1`：

- 不生成训练 transition。
- 仍保存视频和 manifest。
- manifest 标记 `replay_status="too_short"`。

### next_reference_action 不足问题

如果接近 segment 末尾 `next_reference_action` 不足 `C`：

- 方案 A：只生成满足 `i + 2C <= T` 的 transition，最严格。
- 方案 B：允许 padding，但必须写 mask。当前 `RLTReplayBatch` 没有 mask，所以第一阶段建议用方案 A。

## actor/critic 异步读取 buffer 训练

当前 `scripts/train_rlt.py` 只读一个静态 `.npz`，不适合在线 warmup/训练。

建议新增 `scripts/train_rlt_online.py`：

- 监听 `replay/rlt_key_regions/.../shards/*.npz`
- 只读取完整 rename 后的 `.npz`
- 内部维护 replay buffer index
- 每隔 `scan_interval=1s` 扫描新增 shard
- replay size 达到 `min_replay_samples` 后开始训练 critic
- warmup 条件满足后才开始 actor update
- actor publish 仍按 `actor_publish_interval` 导出稳定 inference actor

训练状态通过 Redis 发布给前端：

```json
{
  "replay_size": 5320,
  "critic_loss": 0.08,
  "actor_loss": -0.12,
  "q1_mean": 0.42,
  "q2_mean": 0.39,
  "q_gap": 0.03,
  "actor_delta_norm": 0.015,
  "latest_actor_step": 3000
}
```

前端已有 `RLTControlState` 字段可以继续扩展。

## 视频保存策略

### 推荐复用 FfmpegH264Writer

后台 writer 可以复用：

- `examples/aloha_real/video_hdf5_saver.py::FfmpegH264Writer`

但不要复用同步 `_append_step()`。

后台写视频时：

```python
for camera in camera_names:
    writer = FfmpegH264Writer(path, fps, width, height)
    for record in records:
        writer.write(record.images[camera])
    writer.release()
```

这种写法在后台线程里执行，即使 ffmpeg 慢，也不会阻塞机械臂控制。

### HDF5 内容

短轨迹 `episode.hdf5` 保存调试 metadata：

```text
attrs:
  sim = False
  images_external = True
  image_format = "mp4"
  is_key_region = True
  key_region_id
  task
  phase
  reward
  score_timeout
  fps
  camera_names

datasets:
  observations/qpos
  observations/qvel
  observations/effort
  action
  reference_action
  timestamps
```

`z_rl` 不建议放进视频 HDF5 主路径，训练用 `.npz` 更直接。可以在 HDF5 attrs 里记录对应 replay shard path。

### manifest.json

每个短轨迹目录保存：

```json
{
  "key_region_id": "...",
  "task": "unscrew_bottle_cap",
  "phase": "warmup",
  "reward": 1,
  "score_timeout": false,
  "start_time": 1760000000.0,
  "end_time": 1760000003.2,
  "score_time": 1760000004.0,
  "num_frames": 160,
  "num_replay_transitions": 65,
  "fps": 50.0,
  "cameras": ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"],
  "replay_shard": "replay/rlt_key_regions/.../key_region_000012_<id>.npz"
}
```

## 前端展示方案

第一阶段最小改动：

- 短轨迹保存到 `rollouts/key_regions/...`
- 现有 RolloutBrowser 自动显示
- 用户从文件树打开 `cam_right_wrist.mp4`

第二阶段推荐改动：

- `RolloutBrowser` 增加可选 `rootPath`
- 后端 `/api/rollouts/tree?path=key_regions`
- 前端增加 “Key Regions” tab
- 文件列表展示 reward、phase、duration、created time
- 读取每个目录的 `manifest.json`，显示成功/失败、是否 timeout、对应 replay samples 数

## 卡顿风险控制策略

### 1. on_step 最大耗时预算

50Hz 控制周期是 20ms。建议 `on_step()` 的 recorder 部分目标低于 1 到 2ms。

必须记录：

```text
rlt_recorder/on_step_ms
rlt_recorder/write_queue_size
rlt_recorder/dropped_video_frames
rlt_recorder/dropped_segments
```

如果 `on_step_ms > 5ms`，前端报警。

### 2. 图像复制策略

图像是最大内存和 CPU 来源。建议：

- pre-roll 只保留 2 秒。
- active key region 最长限制 20 秒。
- 如果超过 `max_key_region_seconds`，自动 end 并进入 await_score。
- 图像只用于视频展示，不进入 actor/critic replay。
- 训练 replay 只保存 `z_rl/proprio/action/reference_action/reward`，避免训练时解码视频。

### 3. 后台写队列限流

`write_queue.maxsize=8`。

如果队列满：

- 不阻塞 `on_step()`。
- 标记当前 segment `video_status="dropped_due_to_backpressure"`。
- replay 低维数据优先保留，视频可以降级丢弃。

### 4. 原子写入

所有训练 shard：

```text
*.npz.tmp -> fsync/close -> rename *.npz
```

manifest：

```text
manifest.json.tmp -> rename manifest.json
manifest.jsonl append 可加文件锁
```

训练进程只读非 `.tmp` 文件，避免读到半写文件。

### 5. 视频编码策略与降级

默认环境通常有 NVIDIA GPU，因此短轨迹视频编码应优先尝试 GPU NVENC，而不是默认走 CPU。

但是“有 GPU”不等于容器里的 `ffmpeg` 一定支持 `h264_nvenc`：镜像中的 ffmpeg 可能没有编译 NVENC，Docker 也可能没有正确暴露 NVIDIA runtime，或者驱动/ffmpeg 版本不匹配。因此编码策略应是：

```text
1. 首选 GPU NVENC: h264_nvenc
2. 如果 h264_nvenc 不可用或启动失败，自动 fallback 到 CPU libx264
3. 不管使用 GPU 还是 CPU，编码都只能在后台 writer 中执行，不能阻塞 50Hz 控制循环
```

启动时检测：

```bash
ffmpeg -hide_banner -encoders | grep h264_nvenc
```

推荐 GPU 编码参数：

```text
h264_nvenc preset=p4 rc=vbr cq=23 pix_fmt=yuv420p faststart
```

对应 ffmpeg 参数形态：

```bash
-c:v h264_nvenc -preset p4 -rc vbr -cq 23 -pix_fmt yuv420p -movflags +faststart
```

CPU fallback 参数：

```text
libx264 veryfast crf=23 yuv420p faststart
```

对应 ffmpeg 参数形态：

```bash
-c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -movflags +faststart
```

如果编码仍然跟不上：

- 降低短轨迹视频 fps 到 25，仅视频降采样，replay 仍按 50Hz。
- 只保存 `cam_right_wrist` 和 `cam_high`，其他相机可配置关闭。
- CPU fallback 下将 `veryfast` 降级为 `ultrafast`。
- 记录 `writer/encoder_backend`、`writer/encode_latency_ms`、`writer/dropped_video_frames` 到日志或 wandb。

### 6. 训练进程与采集进程隔离

actor/critic 训练应是独立进程，不应运行在 runtime 控制循环线程中。

建议：

- runtime 只负责采集和发布稳定 inference actor 版本切换信号。
- online trainer 只读 replay shard，写 actor checkpoint。
- runtime 只在安全边界加载新的 inference actor，不能每个训练 step 替换。

## 可复用与不可复用清单

### 可复用

- `voice_assistant_web/backend/app/rlt_control.py`
  - RLT 状态机基础、score timeout、warmup count、Redis publish。
- `voice_assistant_web/backend/app/main.py`
  - RLT API。
  - `/api/rollouts/tree`。
  - `/api/rollouts/video` Range streaming。
- `voice_assistant_web/frontend/src/components/RolloutBrowser.tsx`
  - 文件树和 MP4 播放。
- `examples/aloha_real/video_hdf5_saver.py::FfmpegH264Writer`
  - H.264 MP4 写入。
- `src/openpi/training/rlt_training.py`
  - actor/critic 训练 step。
- `src/openpi/models/rlt.py`
  - actor、target actor、twin critic、TD3 target。

### 需要改造

- `packages/openpi-client/src/openpi_client/runtime/subscriber.py`
  - 增加可选 key region hook。
- `packages/openpi-client/src/openpi_client/runtime/runtime.py`
  - 将 RLT Redis 事件转发给 subscribers。
  - 在 action metadata 中保留 `z_rl/reference_action`。
- `examples/aloha_real/main.py`
  - 增加 `save_format="rlt_key_region"` 或额外 subscriber 参数。
- `scripts/train_rlt.py`
  - 保留静态训练。
  - 新增 online trainer，而不是硬改原脚本。

### 不应复用为主路径

- `examples/aloha_real/hdf5_utils.py`
  - 它把图像 JPEG 写入 HDF5，不适合网页直接播放。
- `VideoHdf5Saver._on_step_split_on_reset`
  - 长轨迹 home/reset 切分逻辑，不适合关键区域。
- `VideoHdf5Saver._append_step`
  - 同步写视频，不能直接用于 50Hz RLT recorder。

## 实施顺序建议

### 阶段 1：打通低开销关键区域保存

1. 扩展 subscriber hook。
2. runtime 转发 key region start/end/score。
3. 新增 `KeyRegionReplayRecorder`，只保存低维 replay `.npz`，先不保存视频。
4. 用假数据或短时间实机验证 `.npz` 能被 `scripts/train_rlt.py` 读取。

### 阶段 2：增加短视频展示

1. 后台 writer 复用 `FfmpegH264Writer` 写 key region MP4。
2. 保存 `episode.hdf5` 和 `manifest.json`。
3. 确认 RolloutBrowser 能看到 `rollouts/key_regions/.../cam_right_wrist.mp4`。

### 阶段 3：在线训练

1. 新增 `train_rlt_online.py`。
2. 支持扫描 replay shard 目录。
3. replay size 达到阈值后 critic-only burn-in。
4. warmup target 满足后 actor update。
5. actor publish interval 导出稳定 inference actor。

### 阶段 4：运行时 actor 安全切换

1. runtime 只在 idle/chunk boundary 加载新 inference actor。
2. 每次切换记录 actor version。
3. manifest 记录每个 key region 使用的 actor version。

## 最关键设计判断

RLT 第二阶段不应该把 LeRobot 作为在线 replay 主格式。当前 actor/critic 训练需要的是已经提取好的 `z_rl/proprio/action/reference_action/reward` transition，视频只是人工复查和调试材料。

最合适的主格式是：

```text
npz replay shards + manifest.jsonl
```

最合适的视频展示格式是：

```text
rollouts/key_regions/.../cam_*.mp4 + episode.hdf5 + manifest.json
```

这样可以同时满足：

- 控制循环低延迟。
- 前端能看短轨迹视频。
- actor/critic 能不间断读取新数据训练。
- 后续如果需要，也可以离线导出成 LeRobot 数据集。
