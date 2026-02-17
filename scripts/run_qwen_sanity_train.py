"""Run a Qwen3-VL sanity training job for Pi value function.

This script reuses the base config from:
`packages/pi-value-function/train_battery_bank_task.py`
and applies small-run overrides suitable for quick validation.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import runpy

from pi_value_function.training.train import train
from pi_value_function.training.train_config import CheckpointConfig
from pi_value_function.training.train_config import LoggingConfig


def build_config(
    *,
    steps: int,
    batch_size: int,
    num_workers: int,
    val_every: int,
    checkpoint_root: str,
    exp_suffix: str,
):
    base = runpy.run_path("packages/pi-value-function/train_battery_bank_task.py")["config"]

    exp_name = f"{base.exp_name}_{exp_suffix}"
    checkpoint_dir = str((Path(checkpoint_root).resolve()))

    cfg = dataclasses.replace(
        base,
        exp_name=exp_name,
        num_train_steps=steps,
        batch_size=batch_size,
        num_workers=num_workers,
        num_steps_per_validation=val_every,
        checkpoint=CheckpointConfig(
            checkpoint_dir=checkpoint_dir,
            save_every_n_steps=max(1, min(250, steps // 2)),
            keep_n_checkpoints=2,
            keep_period=None,
            overwrite=True,
            resume=False,
        ),
        logging=LoggingConfig(
            log_every_n_steps=50,
            wandb_enabled=True,
            wandb_project=base.logging.wandb_project,
            wandb_run_name=exp_name,
            wandb_entity=base.logging.wandb_entity,
        ),
    )
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen sanity training with W&B enabled.")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--val-every", type=int, default=100, help="Validation interval in steps (0 disables).")
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default="./checkpoints_sanity",
        help="Root directory for checkpoints.",
    )
    parser.add_argument(
        "--exp-suffix",
        type=str,
        default="qwen_sanity_1k",
        help="Suffix appended to base experiment name.",
    )
    args = parser.parse_args()

    cfg = build_config(
        steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_every=args.val_every,
        checkpoint_root=args.checkpoint_root,
        exp_suffix=args.exp_suffix,
    )

    print("=" * 80)
    print("Qwen Sanity Training")
    print("=" * 80)
    print(f"exp_name: {cfg.exp_name}")
    print(f"steps: {cfg.num_train_steps}")
    print(f"batch_size: {cfg.batch_size}")
    print(f"num_workers: {cfg.num_workers}")
    print(f"val_every: {cfg.num_steps_per_validation}")
    print(f"checkpoint_dir: {cfg.checkpoint.checkpoint_dir}")
    print(f"wandb_enabled: {cfg.logging.wandb_enabled}")
    print(f"wandb_project: {cfg.logging.wandb_project}")
    print(f"wandb_run_name: {cfg.logging.wandb_run_name}")
    print("=" * 80)

    train(cfg)


if __name__ == "__main__":
    main()
