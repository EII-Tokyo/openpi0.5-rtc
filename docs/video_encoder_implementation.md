## 视频 Encoder 实现说明

这份文档说明当前视频 memory 图像 encoder 的实现方式，主要对应：

- [siglip.py](/home/eii/openpi0.5-rtc-feat-pi05-subtask-train-lambda-20260218/src/openpi/models/siglip.py)

同时记录训练/推理两侧的数据流，以及本地的延迟测试结果。

### 目标

当前实现是在原始单帧 SigLIP image encoder 基础上，扩展出多帧输入能力，同时尽量保持对旧代码的最小改动：

- 保留原始的 `nn.MultiHeadDotProductAttention`
- 保留原始的 `spatial attention + MLP` block 结构
- 只在每隔 `N` 层时加入 temporal mixing
- 保持旧 checkpoint 可加载

实现路线更接近 TimeSformer 风格的 divided attention，而不是新写一套更吃显存的自定义 separable attention。

### 输入 / 输出约定

encoder 现在同时支持两种输入：

- 单帧图像：`[B, H, W, C]`
- 视频输入：`[B, T, H, W, C]`

对视频输入，处理过程是：

1. 每一帧单独做 patchify
2. 每一帧都加空间位置编码
3. 再沿时间维加固定的时间位置编码
4. transformer 在 `[B, T, P, D]` 上运行
5. 最后只把最新时刻 `x[:, -1, :, :]` 的输出传给下游

其中：

- `P` 是空间 patch 数
- `D` 是 token hidden dim

### 空间位置编码和时间位置编码

当前实现保留了原始 SigLIP 的空间位置编码：

- `pos_embedding`

对视频输入，额外加入：

- 固定的 1D sinusoidal temporal position embedding

这里有一个关键细节：

- 会把“当前帧”平移到时间偏移 `0`
- 也就是 `e(0) = 0`

这样做的目的，是让单帧路径尽量和原模型行为一致。

对应代码位置：

- `posemb_sincos_1d(...)`
- `_Module.__call__(...)`

### Transformer Block 结构

encoder block 仍然基于原来的 `Encoder1DBlock`，继续使用这些老模块：

- `LayerNorm_0`
- `MultiHeadDotProductAttention_0`
- `LayerNorm_1`
- `MlpBlock_0`

现在一共有两种层：

1. 普通层

- spatial attention
- MLP

2. temporal 层（满足 `(lyr + 1) % temporal_every_n_layers == 0`）

- temporal attention
- spatial attention
- MLP

实现方式是：

- 在同一个 block 里复用同一套 attention 模块两次
- 当 `temporal_first=True` 时，先走时间注意力，再走空间注意力

#### Temporal attention 的具体做法

- 输入先从 `[B, T, P, D]` reshape 成 `[B * P, T, D]`
- 在时间维上做 causal attention
- causal mask 形状是 `[1, 1, T, T]`

#### Spatial attention 的具体做法

- temporal 输出再 reshape 回 `[B, T, P, D]`
- 然后变成 `[B * T, P, D]`
- 在每一帧内部按 patch 做空间 attention

这样做的好处是：

- 继续复用原始 attention 实现
- 不需要引入新的自定义 space-time attention kernel
- 显存开销比之前那版新 attention 更小

### 为什么说这是“最小改动”

相对旧版单帧代码，这次必须引入的改动只有这些：

- `_Module.__call__` 支持视频形状输入
- 增加时间位置编码
- `MlpBlock` 读取最后一维作为通道宽度，兼容 4 维 token tensor
- 在 `Encoder1DBlock` 里加 `temporal_first` 分支
- 在 `Encoder` 里按周期触发 temporal 层
- 图像 encoder 输出只保留最新时刻

同时旧 checkpoint 的参数名保持不变：

- 没有新增一棵 temporal-only 参数子树
- attention / MLP 还是复用原来的权重

### 训练数据流

多帧样本是在：

- [data_loader.py](/home/eii/openpi0.5-rtc-feat-pi05-subtask-train-lambda-20260218/src/openpi/training/data_loader.py)

里通过：

- `TemporalFrameStackDataset`

构造出来的。

当前配置是：

- `video_memory_num_frames = 4`
- `video_memory_stride_seconds = 1.0`

所以一个时间点 `t` 的样本，会取：

- `t-3s`
- `t-2s`
- `t-1s`
- `t`

而且对每一路相机都这样取。

如果样本靠近 episode 开头，历史帧不够：

- 就重复同一个 episode 内最早可用的那一帧

当前实现依赖两个前提：

- `frame_index` 从 `0` 开始
- 同一个 episode 内 `frame_index` 连续

为了避免 trainability remapping 干扰 temporal lookup：

- `IsForTrainingWrapper` 只作用在顶层采样的那个 idx
- 历史帧索引直接对原始 dataset 做连续索引

### 推理数据流

在线推理侧和训练保持同样的时间结构，主要逻辑在：

- [runtime.py](/home/eii/openpi0.5-rtc-feat-pi05-subtask-train-lambda-20260218/packages/openpi-client/src/openpi_client/runtime/runtime.py)

当前 runtime 行为是：

- 为每一路相机维护历史帧缓冲
- 每次推理时取：
  - `now-3s`
  - `now-2s`
  - `now-1s`
  - `now`
- 如果历史不够，就重复最早可用帧

所以训练和推理在时间采样结构上是一致的。

### 本地延迟测试

本地测试环境：

- GPU：`NVIDIA GeForce RTX 5090`
- checkpoint：
  `checkpoints/twist_and_static_mixture_full_finetune/<exp_name>/<step>`
- 输入样本：
  `tmp/test_hdf5/episode_0.hdf5`
- 相机：
  - `cam_high`
  - `cam_low`
  - `cam_left_wrist`
  - `cam_right_wrist`

测试方法：

- 同一份 checkpoint
- 同一个 state / prompt / structured subtask
- 比较单帧输入和 4 帧输入
- 先 warmup 1 次
- 然后各跑 5 次取平均

测试结果：

- 单帧：
  - low-level `infer`: `58.49 ms`
  - high-level decode: `583.95 ms`
  - high-level decode per token: `8.225 ms/token`
  - 平均生成 token 数：`71`

- 4 帧（1 秒间隔）：
  - low-level `infer`: `72.86 ms`
  - high-level decode: `584.95 ms`
  - high-level decode per token: `8.356 ms/token`
  - 平均生成 token 数：`70`

从单帧到 4 帧的增量：

- low-level：`+14.37 ms`
- high-level 总 decode：`+1.00 ms`
- high-level 每 token：`+0.131 ms/token`

可以这样理解：

- low-level 的延迟增加比较明显，但还在可接受范围
- high-level 总 decode 时间在这次测试里几乎没变
- high-level 每 token 成本有轻微上升

### 当前限制

- temporal attention 不是每层都有，而是周期性插入
- 当前实现是基于现有 attention 模块的 TimeSformer 风格 divided attention
- 不是一套新的联合 space-time attention kernel
- 下游 VLA 仍然只消费“当前时刻”的表示，不消费所有 temporal token

### 本次功能涉及的文件

- [siglip.py](/home/eii/openpi0.5-rtc-feat-pi05-subtask-train-lambda-20260218/src/openpi/models/siglip.py)
- [data_loader.py](/home/eii/openpi0.5-rtc-feat-pi05-subtask-train-lambda-20260218/src/openpi/training/data_loader.py)
- [config.py](/home/eii/openpi0.5-rtc-feat-pi05-subtask-train-lambda-20260218/src/openpi/training/config.py)
- [model.py](/home/eii/openpi0.5-rtc-feat-pi05-subtask-train-lambda-20260218/src/openpi/models/model.py)
- [aloha_policy.py](/home/eii/openpi0.5-rtc-feat-pi05-subtask-train-lambda-20260218/src/openpi/policies/aloha_policy.py)
- [runtime.py](/home/eii/openpi0.5-rtc-feat-pi05-subtask-train-lambda-20260218/packages/openpi-client/src/openpi_client/runtime/runtime.py)
