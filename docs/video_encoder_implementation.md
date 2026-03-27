## Video Encoder Implementation

This document describes the current video-memory image encoder implementation in
`src/openpi/models/siglip.py`, together with the matching training/inference
data flow and a local latency measurement.

### Goal

The current implementation extends the original single-frame SigLIP image
encoder so that it can consume multiple frames per camera, while keeping the
change set close to the old code:

- keep the original `nn.MultiHeadDotProductAttention`
- keep the original spatial attention + MLP block structure
- add temporal mixing only every `N` layers
- keep old checkpoints loadable

This is intentionally closer to TimeSformer-style divided attention than to a
custom memory-intensive separable attention implementation.

### Input / Output Contract

The encoder now supports both:

- image input: `[B, H, W, C]`
- video input: `[B, T, H, W, C]`

For video input:

1. each frame is patchified independently
2. spatial position embeddings are added per frame
3. fixed temporal sinusoidal position embeddings are added across time
4. the transformer runs on `[B, T, P, D]`
5. only the latest timestep output `x[:, -1, :, :]` is exposed downstream

`P` is the number of spatial patches and `D` is the token width.

### Spatial and Temporal Position Embeddings

The current implementation keeps the original spatial position embedding:

- `pos_embedding` from the old SigLIP path

For video input it additionally adds:

- fixed 1D sinusoidal temporal position embedding

Important detail:

- the latest frame is shifted to temporal offset `0`
- i.e. `e(0) = 0`

This keeps the single-frame path as close as possible to the original model
behavior.

Code path:

- `posemb_sincos_1d(...)`
- `_Module.__call__(...)`

### Transformer Block Structure

The encoder block remains based on the old `Encoder1DBlock` using:

- `LayerNorm_0`
- `MultiHeadDotProductAttention_0`
- `LayerNorm_1`
- `MlpBlock_0`

There are now two behaviors:

1. normal layer

- spatial attention
- MLP

2. temporal layer (`(lyr + 1) % temporal_every_n_layers == 0`)

- temporal attention
- spatial attention
- MLP

This is implemented by reusing the same attention module twice in the same
block when `temporal_first=True`.

Temporal attention details:

- input is reshaped from `[B, T, P, D]` to `[B * P, T, D]`
- attention is causal over time
- causal mask shape is `[1, 1, T, T]`

Spatial attention details:

- temporal output is reshaped back to `[B, T, P, D]`
- then reshaped to `[B * T, P, D]`
- spatial attention is applied patch-wise within each frame

This preserves the old attention implementation and avoids introducing a new
custom separable attention kernel.

### Why This Is a Minimal Change

Compared with the old single-frame code, the required changes are limited to:

- allow video-shaped input in `_Module.__call__`
- add temporal position embedding
- allow `MlpBlock` to operate on tensors with more than 3 dims by reading the
  last dimension as channel width
- add a `temporal_first` path inside `Encoder1DBlock`
- add a periodic temporal layer trigger in `Encoder`
- keep only the latest timestep after the image encoder

The old checkpoint parameter names are preserved:

- no new temporal-only parameter subtree is introduced
- existing attention / MLP weights are reused

### Training Data Flow

Multi-frame samples are built in:

- `src/openpi/training/data_loader.py`

using:

- `TemporalFrameStackDataset`

Current behavior:

- `video_memory_num_frames = 4`
- `video_memory_stride_seconds = 1.0`

So a sample at time `t` uses:

- `t-3s`
- `t-2s`
- `t-1s`
- `t`

for every camera.

If the sample is near the beginning of an episode and there is not enough
history:

- the earliest available frame in the same episode is repeated

Implementation assumptions:

- `frame_index` starts at `0`
- `frame_index` is contiguous within an episode

To avoid mixing trainability remapping with temporal lookup:

- `IsForTrainingWrapper` logic is applied only to the top-level sampled index
- temporal history indices are resolved directly against the underlying raw
  dataset

### Inference Data Flow

Online inference mirrors the training structure:

- `packages/openpi-client/src/openpi_client/runtime/runtime.py`

Current runtime behavior:

- keep per-camera history buffers
- sample frames at `now-3s`, `now-2s`, `now-1s`, `now`
- if history is insufficient, repeat the earliest available frame

So training and inference use the same temporal layout.

### Local Latency Measurement

Local measurement was run on:

- GPU: `NVIDIA GeForce RTX 5090`
- checkpoint:
  `checkpoints/twist_off_the_bottle_cap_subtask_full_finetune/twist_off_the_bottle_cap_subtask_full_finetune_bs128_20260321_232825/10000`
- input:
  `tmp/test_hdf5/episode_0.hdf5`
- cameras:
  `cam_high`, `cam_low`, `cam_left_wrist`, `cam_right_wrist`

Measurement method:

- same checkpoint
- same current state / prompt / structured subtask
- compare single-frame input vs 4-frame input
- warmup once
- then average 5 runs

Results:

- 1 frame:
  - low-level `infer`: `58.49 ms`
  - high-level decode: `583.95 ms`
  - high-level decode per token: `8.225 ms/token`
  - average generated steps: `71`

- 4 frames (1s stride):
  - low-level `infer`: `72.86 ms`
  - high-level decode: `584.95 ms`
  - high-level decode per token: `8.356 ms/token`
  - average generated steps: `70`

Delta from 1 frame to 4 frames:

- low-level `+14.37 ms`
- high-level total decode `+1.00 ms`
- high-level decode per token `+0.131 ms/token`

Interpretation:

- low-level latency increases noticeably but remains moderate
- high-level total decode is almost unchanged in this measurement
- per-token decode cost rises slightly

### Current Limitations

- temporal attention is inserted periodically, not in every layer
- the implementation uses TimeSformer-style divided attention using the
  existing attention module, not a new custom joint space-time attention kernel
- latest-frame-only output means downstream VLA still consumes the current
  timestep representation, not all temporal tokens

### Files Touched by This Feature

- `src/openpi/models/siglip.py`
- `src/openpi/training/data_loader.py`
- `src/openpi/training/config.py`
- `src/openpi/models/model.py`
- `src/openpi/policies/aloha_policy.py`
- `packages/openpi-client/src/openpi_client/runtime/runtime.py`
