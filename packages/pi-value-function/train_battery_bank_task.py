"""Train a multi-task value function over all available prompts/tasks."""

from pathlib import Path

from pi_value_function.training.train import train
from pi_value_function.training.train_config import TrainConfig, ValueDataConfig, CheckpointConfig, LoggingConfig
from pi_value_function.config import PiValueConfig
import openpi.training.optimizer as _optimizer

RUN_NAME = "normal_target_unfreeze_sim0p45_bs16_lr1e4_velocity_relu"

# Multi-task value function training
config = TrainConfig(
    exp_name=RUN_NAME,
    model_config=PiValueConfig(
        value_dims=201,
        value_min=-1.0,
        value_max=0.0,
        gemma_variant="gemma-3-270m",
        siglip_variant="siglip2-so400m-patch16-384",
    ),
    freeze_backbone=False,
    freeze_gemma=False,
    freeze_siglip=False,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=2000,
        peak_lr=1e-4,
        decay_steps=25_000,
        decay_lr=5e-5,
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
            # "michios/droid_xxjd",
            # "michios/droid_xxjd_2",
            # "michios/droid_xxjd_3",
            # "michios/droid_xxjd_4",
            # "michios/droid_xxjd_5",
            # "michios/droid_xxjd_6",
            "michios/droid_xxjd_7",
            # "michios/droid_xxjd_8_2",
        ],
        failure_repo_ids=[
            "michios/droid_xxjd_fail_1"
        ],
        train_split=0.9, 
        # split_seed=42,

        # Multi-task mode: do not relabel success episodes as failures.
        target_task="Put the battery bank in the orange box",
        treat_other_tasks_as_failure=True,
        augmented_failure_max_prompt_similarity=0.45,
        include_image_tag=True,
        include_velocity=True,

        failure_cost_json="configs/failure_costs.json",
        default_c_fail=800.0,
        success_sampling_ratio=0.8,
    ),
    checkpoint=CheckpointConfig(
        checkpoint_dir=str(Path(__file__).resolve().parents[2] / "checkpoints"),
        save_every_n_steps=5000,
        keep_n_checkpoints=2,
        overwrite=True,
        resume=False,
        # resume_checkpoint_path=str(Path(__file__).resolve().parents[2] / "checkpoints/pi_value/battery_bank_in_box_task_value")
    ),
    logging=LoggingConfig(
        log_every_n_steps=50,
        wandb_enabled=False,
        wandb_project="pi-value-function",
        wandb_run_name=RUN_NAME,
    ),
    num_workers=0,
    num_steps_per_validation=500,
    num_validation_batches=8,
    overfit_one_batch=False,
    seed=42,
)



if __name__ == "__main__":
    print("=" * 80)
    print("Multi-Task Value Function Training")
    print("=" * 80)
    print("Target-task filtering: disabled")
    print(f"Default C_fail: {config.data.default_c_fail}")
    print(f"Tokenizer include_image_tag: {config.data.include_image_tag}")
    print(f"Success sampling ratio: {config.data.success_sampling_ratio:.0%}")
    print(f"Overfit one-batch mode: {config.overfit_one_batch}")
    print(f"Total steps: {config.num_train_steps:,}")
    print(f"Validation every: {config.num_steps_per_validation} steps ({config.num_validation_batches} batches avg)")
    print("=" * 80)
    print()
    train(config)
