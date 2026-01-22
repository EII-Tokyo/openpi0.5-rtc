"""Train value function with dynamic task augmentation.

This training approach augments failure data by treating successful episodes
with different prompts as failures. Key differences from standard training:

Standard training:
- Episode with task A succeeds → label as success for task A
- Episode with task B succeeds → label as success for task B
- Limited failure data

Dynamic task augmentation:
- Per sample, randomly pick a target task (e.g., task A)
- If episode task matches (task A) → label as success
- If episode task differs (task B, C, etc.) → label as failure with high cost
- Teaches: V(state from task B, prompt="task A") should be very low

This effectively multiplies your failure dataset by N_tasks without needing
to train N separate models!

Benefits:
1. One model handles all tasks (not N models)
2. Augments sparse failure data with "wrong task" examples
3. Teaches strong task specificity
4. Each sample in a batch can have a different target task
"""

from pi_value_function.training.train import train
from pi_value_function.training.train_config import TrainConfig, ValueDataConfig, CheckpointConfig, LoggingConfig
from pi_value_function.config import PiValueConfig
import openpi.training.optimizer as _optimizer

# Training configuration with dynamic task augmentation
config = TrainConfig(
    exp_name="value_dynamic_task_aug_v1",
    model_config=PiValueConfig(
        value_dims=201,
        value_min=-1.0,
        value_max=0.0,
        gemma_variant="gemma-3-270m",
        siglip_variant="siglip2-so400m-patch16-384",
    ),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=500,
        peak_lr=1e-5,
        decay_steps=25_000,
        decay_lr=1e-6,
    ),
    optimizer=_optimizer.AdamW(
        weight_decay=0.01,
        clip_gradient_norm=5.0,
    ),
    num_train_steps=30_000,
    batch_size=64,
    data=ValueDataConfig(
        # Load ALL success datasets (all tasks)
        success_repo_ids=[
            "michios/droid_xxjd",
            "michios/droid_xxjd_2",
            "michios/droid_xxjd_3",
            "michios/droid_xxjd_4",
            "michios/droid_xxjd_5",
            "michios/droid_xxjd_6",
            "michios/droid_xxjd_7",
            "michios/droid_xxjd_8_2",
        ],
        failure_repo_ids=[
            "michios/droid_xxjd_fail_1"
        ],
        train_split=0.9,

        # ENABLE DYNAMIC TASK AUGMENTATION
        # Each sample randomly picks a target task, then checks if episode matches
        use_dynamic_task_augmentation=True,

        # Only use tasks with enough episodes
        # Note: With 90/10 train/val split, validation gets 10% of episodes
        # So set threshold low enough for validation set to have enough tasks
        min_episodes_per_task=3,  # ~3 in val means ~30 total episodes per task

        failure_cost_json="configs/failure_costs.json",
        default_c_fail=200.0,  # Moderate cost for "wrong task" failures (was 2000, too extreme)
        success_sampling_ratio=0.5,  # Balance success/failure in training
    ),
    checkpoint=CheckpointConfig(
        checkpoint_dir="./checkpoints",
        save_every_n_steps=5_000,
        keep_n_checkpoints=2,
        overwrite=True,
    ),
    logging=LoggingConfig(
        log_every_n_steps=50,
        wandb_enabled=True,
        wandb_project="pi-value-function",
        wandb_run_name="dynamic_task_aug_v2",
        
    ),
    num_workers=0,
    num_steps_per_validation=100,
    seed=42,
)


if __name__ == "__main__":
    print("=" * 80)
    print("Value Function Training with Dynamic Task Augmentation v2")
    print("=" * 80)
    print()
    print("Key improvements over v1:")
    print(f"  - Filters to tasks with >= {config.data.min_episodes_per_task} episodes")
    print(f"  - BALANCED sampling: 50% success, 50% failure (was ~1% success)")
    print(f"  - Lower failure cost: {config.data.default_c_fail} (was 2000)")
    print()
    print("Strategy:")
    print("  1. Sample an episode")
    print("  2. 50% chance: use episode's own task → SUCCESS")
    print("  3. 50% chance: use a DIFFERENT task → FAILURE")
    print()
    print("This gives balanced training data and better generalization!")
    print("=" * 80)
    print()

    train(config)
