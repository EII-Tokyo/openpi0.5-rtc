# Training Configuration Guide

This document describes the training configuration system for the Pi Value Function model.

## Overview

The training configuration is defined in `train_config.py` and follows the same patterns as the main OpenPI training configuration. It provides a comprehensive, type-safe way to configure all aspects of value function training.

The project now supports two backbone paths:

- Legacy JAX path: `backbone="siglip_gemma3"` (SigLIP + Gemma 3)
- New PyTorch path: `backbone="qwen3vl"` (Qwen3-VL multimodal backbone)

## Main Components

### 1. TrainConfig

The main configuration class that composes all other config classes:

```python
from pi_value_function.training.train_config import TrainConfig
from pi_value_function.config import PiValueConfig

config = TrainConfig(
    exp_name="my_experiment",
    model_config=PiValueConfig(value_dims=1),
    batch_size=32,
    num_train_steps=10_000,
)
```

### 2. ValueDataConfig

Defines data loading and preprocessing:

- `data_path`: Path to training data
- `train_split` / `val_split`: Train/validation split ratios
- `shuffle_buffer_size`: Size of shuffle buffer
- `value_horizon`: Horizon for computing value targets
- `gamma`: Discount factor for returns
- `normalize_rewards`: Whether to normalize rewards
- `data_format`: Data format type ("rlds", "lerobot", "custom")

### 2b. PiValueConfig Backbone Fields

`PiValueConfig` exposes backbone controls:

- `backbone`: `"siglip_gemma3"` or `"qwen3vl"`
- `hf_model_id`: Hugging Face model id (used by Qwen path)
- `backbone_dtype`: `"bfloat16"`, `"float16"`, or `"float32"` (used by Qwen path)

### 3. CheckpointConfig

Defines checkpointing behavior:

- `checkpoint_dir`: Directory for saving checkpoints
- `save_every_n_steps`: Checkpoint frequency
- `keep_n_checkpoints`: Maximum number of checkpoints to keep
- `keep_period`: Period for permanent checkpoints
- `resume_checkpoint_path`: Path to resume from
- `resume` / `overwrite`: Resume/overwrite flags

### 4. LoggingConfig

Defines logging and experiment tracking:

- `log_every_n_steps`: Logging frequency
- `wandb_enabled`: Enable W&B logging
- `wandb_project`: W&B project name
- `wandb_run_name`: W&B run name
- `wandb_entity`: W&B team/username
- `val_every_n_steps`: Validation frequency

## Key Features

### Optimizer Configuration

Reuses OpenPI's optimizer configs:

```python
import openpi.training.optimizer as optimizer

config = TrainConfig(
    lr_schedule=optimizer.CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=3e-4,
        decay_steps=30_000,
        decay_lr=3e-5,
    ),
    optimizer=optimizer.AdamW(
        weight_decay=0.01,
        clip_gradient_norm=1.0,
    ),
)
```

### Freezing Layers

Control which parts of the model to freeze:

```python
# Freeze entire backbone (train only value head)
config = TrainConfig(freeze_backbone=True, ...)

# Freeze only vision encoder
config = TrainConfig(freeze_siglip=True, ...)

# Freeze only language model
config = TrainConfig(freeze_gemma=True, ...)

# Custom freeze filter
config = TrainConfig(freeze_filter=custom_filter, ...)
```

For `backbone="qwen3vl"`, training uses a frozen Qwen backbone with a trainable value head in the PyTorch trainer.

### Multi-Device Training

Support for FSDP and multi-device training:

```python
config = TrainConfig(
    fsdp_devices=8,  # Shard across 8 devices
    use_sharding=True,
    num_devices=8,
    batch_size=128,  # Large batch across devices
)
```

### Helpful Methods

#### `total_steps` property

Computes total training steps from epochs or num_train_steps:

```python
print(f"Total steps: {config.total_steps}")
```

#### `checkpoint_dir_path` property

Returns the full checkpoint directory path:

```python
print(f"Checkpoints: {config.checkpoint_dir_path}")
```

#### `trainable_filter` property

Returns a filter for trainable parameters based on freeze settings:

```python
trainable_params = nnx.split(model, config.trainable_filter)[0]
```

### Validation

The `__post_init__` method validates all settings:

- Ensures resume and overwrite are not both True
- Validates train/val splits sum to ≤ 1.0
- Validates batch size is positive
- Ensures dtype matches model dtype
- Validates discount factor is in [0, 1]

### Quick Start Configs

#### Debug Config

For quick testing during development:

```python
config = TrainConfig.debug_config()
# Uses tiny settings: 100 steps, batch_size=2, wandb disabled
```

#### Quick Experiment Config

For rapid experimentation:

```python
config = TrainConfig.quick_experiment_config(
    data_path="/path/to/data",
    exp_name="quick_test",
    learning_rate=5e-4,
    batch_size=16,
    num_steps=1_000,
)
```

## Common Use Cases

### 1. Value Head Only Training

Train only the value head with frozen backbone:

```python
config = TrainConfig(
    exp_name="value_head_only",
    freeze_backbone=True,
    lr_schedule=optimizer.CosineDecaySchedule(
        peak_lr=1e-3,  # Higher LR for head-only
        decay_steps=5_000,
    ),
    num_train_steps=5_000,
    batch_size=64,
)
```

### 2. Full Fine-tuning

Train all parameters:

```python
config = TrainConfig(
    exp_name="full_finetune",
    freeze_backbone=False,
    lr_schedule=optimizer.CosineDecaySchedule(
        warmup_steps=2_000,
        peak_lr=1e-4,  # Lower LR for full fine-tuning
        decay_steps=50_000,
    ),
    num_train_steps=50_000,
    batch_size=64,
)
```

### 4. Qwen3-VL Value Training

```python
config = TrainConfig(
    exp_name="qwen3vl_value_head",
    model_config=PiValueConfig(
        backbone="qwen3vl",
        hf_model_id="Qwen/Qwen3-VL-2B-Instruct",
        backbone_dtype="bfloat16",
        value_dims=201,
        value_min=-1.0,
        value_max=0.0,
    ),
    num_train_steps=30_000,
    batch_size=64,
)
```

When `backbone="qwen3vl"`, `train.py` dispatches to the torch trainer and checkpoints are written as:

- `model.safetensors`
- `optimizer.pt`
- `metadata.pt`

### 3. RLDS Data Loading

Use RLDS format for large datasets:

```python
config = TrainConfig(
    data=ValueDataConfig(
        data_format="rlds",
        rlds_data_dir="/path/to/rlds",
        value_horizon=15,
        gamma=0.98,
    ),
    num_workers=0,  # RLDS handles multi-processing internally
)
```

## Integration Points

The config integrates with:

1. **PiValueConfig**: Composes the model configuration
2. **OpenPI Optimizer**: Reuses optimizer and LR schedule configs
3. **JAX/Flax**: Uses nnx filters for parameter selection
4. **W&B**: Built-in experiment tracking support

## Default Values

Key defaults (suitable for fine-tuning):

- Learning rate: 3e-4 (peak)
- Batch size: 32
- Warmup steps: 1,000
- Weight decay: 0.01
- Gradient clip norm: 1.0
- Optimizer: AdamW
- Dtype: bfloat16
- Save checkpoints: every 1,000 steps
- Log: every 100 steps
- EMA decay: 0.99
- Discount factor (gamma): 0.99
- Value horizon: 10

## Examples

See `examples/train_config_usage.py` for comprehensive usage examples covering:

1. Basic configuration
2. Freeze backbone (value head only)
3. Freeze vision encoder only
4. Full fine-tuning
5. Debug mode
6. Quick experiments
7. Custom data settings
8. Multi-device training with FSDP

## File Structure

```
packages/pi-value-function/
├── src/pi_value_function/
│   └── training/
│       └── train_config.py       # Main config module
├── examples/
│   └── train_config_usage.py     # Usage examples
└── docs/
    └── training_config.md         # This document
```
