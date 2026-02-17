# Qwen3-VL Migration Guide

This document explains how to use the new Qwen3-VL value-function path while keeping the legacy SigLIP+Gemma path available.

## What Changed

- Added dual backbone support in `PiValueConfig`:
  - `backbone="siglip_gemma3"` (legacy JAX path)
  - `backbone="qwen3vl"` (new PyTorch path)
- Added a PyTorch value model for Qwen3-VL:
  - `pi_value_function/pi_value_qwen3vl_torch.py`
- Added Qwen-specific collate + dataloader helpers:
  - `pi_value_function/training/qwen_collate.py`
- Added Qwen torch trainer:
  - `pi_value_function/training/train_qwen_torch.py`
- Updated value server with backend selection:
  - `scripts/value_server.py --backend {auto,jax,torch}`

## Checkpoint Formats

### Legacy JAX (SigLIP + Gemma)
- Orbax format with `state/` subtree.

### Qwen3-VL Torch
- Per-step directory contains:
  - `model.safetensors`
  - `optimizer.pt`
  - `metadata.pt`
- `metadata.pt` includes:
  - `backbone`, `hf_model_id`, `value_head_layers`, `value_dims`, `value_min`, `value_max`

## Training Example (Qwen3-VL)

```python
from pi_value_function.config import PiValueConfig
from pi_value_function.training.train_config import TrainConfig

config = TrainConfig(
    exp_name="battery_bank_qwen3vl",
    model_config=PiValueConfig(
        backbone="qwen3vl",
        hf_model_id="Qwen/Qwen3-VL-2B-Instruct",
        backbone_dtype="bfloat16",
        value_dims=201,
        value_min=-1.0,
        value_max=0.0,
    ),
)
```

Then run your training entrypoint as usual. `train.py` dispatches automatically based on `model_config.backbone`.

## Serving

Start server with auto detection:

```bash
python scripts/value_server.py --checkpoint /path/to/checkpoints/pi_value/<exp>/<step> --backend auto
```

- If checkpoint contains torch files, the Qwen torch policy is loaded.
- If checkpoint contains `state/`, legacy JAX policy is loaded.

## Resource Notes

- Qwen3-VL backbone is frozen; only value head is trained.
- Inference/training still processes all three camera streams.
- GPU is strongly recommended for practical throughput.
