"""Train value function for specific task: 'Put the fan in the cardboard box'.

This script treats:
- Success: Episodes with task "Put the fan in the cardboard box"
- Failure: All other episodes (including other successful tasks) with cost 2000
"""

from pi_value_function.training.train import train
from pi_value_function.training.train_config import TrainConfig, ValueDataConfig, CheckpointConfig, LoggingConfig
from pi_value_function.config import PiValueConfig
import openpi.training.optimizer as _optimizer

# Task-specific value function training
config = TrainConfig(
    exp_name="battery_bank_in_box_task_value",
    model_config=PiValueConfig(
        value_dims=201,
        value_min=-1.0,
        value_max=0.0,
        gemma_variant="gemma-3-270m",
        siglip_variant="siglip2-so400m-patch16-384",
    ),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=500,
        peak_lr=3e-5,
        decay_steps=10_000,
        decay_lr=3e-6,
    ),
    optimizer=_optimizer.AdamW(
        weight_decay=0.01,
        clip_gradient_norm=1.0,
    ),
    num_train_steps=30_000,
    batch_size=16,
    data=ValueDataConfig(
        # Load ALL success datasets
        success_repo_ids=[
            "michios/droid_xxjd_7"
        ],
        failure_repo_ids=[
            "michios/droid_xxjd_fail_1"
        ],
        train_split=0.9, 
        # split_seed=42,

        # Task-specific filtering
        target_task="Put the battery bank in the orange box",
        treat_other_tasks_as_failure=True,  # Treat all other tasks as failures!

        failure_cost_json="configs/failure_costs.json",
        default_c_fail=2000.0,
        success_sampling_ratio=0.5,
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
        wandb_run_name="battery_bank_in_box_task_v1",
    ),
    num_workers=0,
    num_steps_per_validation=500,
    seed=42,
)



if __name__ == "__main__":
    print("=" * 80)
    print("Task-Specific Value Function Training")
    print("=" * 80)
    print(f"Target task: '{config.data.target_task}'")
    print(f"Strategy: Treat all other tasks as failures with cost {config.data.default_c_fail}")
    print(f"Success sampling ratio: {config.data.success_sampling_ratio:.0%}")
    print(f"Total steps: {config.num_train_steps:,}")
    print(f"Validation every: {config.num_steps_per_validation} steps")
    print("=" * 80)
    print()
    train(config)