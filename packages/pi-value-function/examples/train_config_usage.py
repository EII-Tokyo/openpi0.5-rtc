"""Example usage of the TrainConfig for Pi Value Function training.

This demonstrates various ways to create and customize training configurations.
"""

from pi_value_function.training.train_config import (
    TrainConfig,
    ValueDataConfig,
    CheckpointConfig,
    LoggingConfig,
)
from pi_value_function.config import PiValueConfig
import openpi.training.optimizer as optimizer


def example_1_basic_config():
    """Create a basic training configuration."""
    config = TrainConfig(
        exp_name="value_function_basic",
        model_config=PiValueConfig(value_dims=1),
        batch_size=32,
        num_train_steps=10_000,
        data=ValueDataConfig(data_path="/path/to/droid/data"),
    )
    print(f"Total training steps: {config.total_steps}")
    print(f"Checkpoint directory: {config.checkpoint_dir_path}")
    return config


def example_2_freeze_backbone():
    """Create a config with frozen backbone (only train value head)."""
    config = TrainConfig(
        exp_name="value_head_only",
        model_config=PiValueConfig(value_dims=1),
        freeze_backbone=True,  # Freeze SigLIP + Gemma
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=1e-3,  # Higher LR for head-only training
            decay_steps=5_000,
            decay_lr=1e-4,
        ),
        batch_size=64,
        num_train_steps=5_000,
        data=ValueDataConfig(data_path="/path/to/data"),
    )
    return config


def example_3_freeze_vision_only():
    """Create a config with frozen vision encoder (train Gemma + value head)."""
    config = TrainConfig(
        exp_name="finetune_gemma_value",
        model_config=PiValueConfig(value_dims=1),
        freeze_siglip=True,  # Freeze only SigLIP
        freeze_gemma=False,  # Train Gemma
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=3e-4,
            decay_steps=20_000,
            decay_lr=3e-5,
        ),
        batch_size=32,
        num_train_steps=20_000,
        data=ValueDataConfig(data_path="/path/to/data"),
    )
    return config


def example_4_full_finetuning():
    """Create a config for full model fine-tuning."""
    config = TrainConfig(
        exp_name="full_finetune",
        model_config=PiValueConfig(
            value_dims=1,
            gemma_variant="gemma-3-270m",
            siglip_variant="siglip2-so400m-patch16-384",
        ),
        # No freezing - train everything
        freeze_backbone=False,
        freeze_siglip=False,
        freeze_gemma=False,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=1e-4,  # Lower LR for full fine-tuning
            decay_steps=50_000,
            decay_lr=1e-5,
        ),
        optimizer=optimizer.AdamW(
            weight_decay=0.01,
            clip_gradient_norm=1.0,
        ),
        batch_size=64,
        num_train_steps=50_000,
        data=ValueDataConfig(
            data_path="/path/to/large/dataset",
            shuffle_buffer_size=50_000,
            value_horizon=20,
            gamma=0.99,
        ),
        checkpoint=CheckpointConfig(
            checkpoint_dir="./checkpoints",
            save_every_n_steps=2_000,
            keep_n_checkpoints=10,
            keep_period=10_000,
        ),
        logging=LoggingConfig(
            log_every_n_steps=100,
            wandb_enabled=True,
            wandb_project="pi-value-function",
            wandb_run_name="full_finetune_v1",
        ),
        fsdp_devices=4,  # Use FSDP sharding across 4 devices
    )
    return config


def example_5_debug_mode():
    """Create a debug config for quick testing."""
    config = TrainConfig.debug_config()
    print(f"Debug config: {config.exp_name}")
    print(f"Total steps: {config.total_steps}")
    print(f"Batch size: {config.batch_size}")
    return config


def example_6_quick_experiment():
    """Create a quick experiment config."""
    config = TrainConfig.quick_experiment_config(
        data_path="/path/to/small/dataset",
        exp_name="quick_test",
        learning_rate=5e-4,
        batch_size=16,
        num_steps=1_000,
    )
    return config


def example_7_custom_data_settings():
    """Create a config with custom data preprocessing settings."""
    config = TrainConfig(
        exp_name="custom_data_processing",
        model_config=PiValueConfig(value_dims=1),
        batch_size=32,
        num_train_steps=10_000,
        data=ValueDataConfig(
            data_path="/path/to/data",
            data_format="rlds",  # Use RLDS format
            rlds_data_dir="/path/to/rlds/dir",
            train_split=0.95,
            val_split=0.05,
            shuffle_buffer_size=20_000,
            value_horizon=15,  # Longer horizon for value estimation
            gamma=0.98,  # Discount factor
            normalize_rewards=True,
            prompt_from_task=True,
        ),
    )
    return config


def example_8_multi_device_training():
    """Create a config for multi-device training with FSDP."""
    config = TrainConfig(
        exp_name="multi_device_fsdp",
        model_config=PiValueConfig(value_dims=1),
        batch_size=128,  # Large batch across devices
        num_train_steps=100_000,
        data=ValueDataConfig(data_path="/path/to/data"),
        # FSDP settings
        fsdp_devices=8,  # Shard across 8 devices
        use_sharding=True,
        num_devices=8,
        # Optimizer for large-scale training
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=5_000,
            peak_lr=5e-4,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        checkpoint=CheckpointConfig(
            save_every_n_steps=5_000,
            keep_n_checkpoints=20,
        ),
        logging=LoggingConfig(
            log_every_n_steps=50,
            val_every_n_steps=1_000,
        ),
    )
    return config


if __name__ == "__main__":
    print("=" * 60)
    print("Pi Value Function Training Configuration Examples")
    print("=" * 60)

    print("\n1. Basic Configuration")
    config1 = example_1_basic_config()

    print("\n2. Freeze Backbone (Train Value Head Only)")
    config2 = example_2_freeze_backbone()

    print("\n3. Freeze Vision Only (Train Gemma + Value Head)")
    config3 = example_3_freeze_vision_only()

    print("\n4. Full Fine-tuning")
    config4 = example_4_full_finetuning()

    print("\n5. Debug Mode")
    config5 = example_5_debug_mode()

    print("\n6. Quick Experiment")
    config6 = example_6_quick_experiment()

    print("\n7. Custom Data Settings")
    config7 = example_7_custom_data_settings()

    print("\n8. Multi-Device Training with FSDP")
    config8 = example_8_multi_device_training()

    print("\n" + "=" * 60)
    print("All configuration examples created successfully!")
    print("=" * 60)
