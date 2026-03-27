"""Run a single PiValue ablation variant from environment variables.

This script exists so torch DataLoader workers (num_workers > 0) can spawn
from a real file path. Running equivalent code from stdin breaks with
forkserver/spawn because children cannot import __main__ from '/<stdin>'.
"""

import dataclasses
import os
import pathlib
import sys


def _parse_none(value: str) -> str | None:
    if value in {"", "__NONE__", "none", "None", "null"}:
        return None
    return value


def _parse_float_or_none(value: str) -> float | None:
    if value in {"", "__NONE__", "none", "None", "null"}:
        return None
    return float(value)


def main() -> None:
    repo_root = pathlib.Path(os.environ["REPO_ROOT"]).resolve()
    pkg_root = repo_root / "packages" / "pi-value-function"
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))

    from train_battery_bank_task import config as base_config
    from pi_value_function.training.train import train

    target_task = _parse_none(os.environ["TARGET_TASK"])
    treat_other = os.environ["TREAT_OTHER"] == "1"
    success_ratio = float(os.environ["SUCCESS_RATIO"])
    aug_ratio = _parse_float_or_none(os.environ["AUG_RATIO"])
    default_c_fail = float(os.environ["DEFAULT_C_FAIL"])
    overfit_one_batch = os.environ["OVERFIT_ONE_BATCH"] == "1"
    num_train_steps = int(os.environ["NUM_TRAIN_STEPS"])
    num_workers = int(os.environ["NUM_WORKERS"])
    num_steps_per_validation = int(os.environ["NUM_STEPS_PER_VALIDATION"])
    num_validation_batches = int(os.environ["NUM_VALIDATION_BATCHES"])
    exp_name = os.environ["EXP_NAME"]

    cfg = dataclasses.replace(
        base_config,
        exp_name=exp_name,
        disable_checkpointing=True,
        overfit_one_batch=overfit_one_batch,
        num_train_steps=num_train_steps,
        num_workers=num_workers,
        num_steps_per_validation=num_steps_per_validation,
        num_validation_batches=num_validation_batches,
        data=dataclasses.replace(
            base_config.data,
            target_task=target_task,
            treat_other_tasks_as_failure=treat_other,
            success_sampling_ratio=success_ratio,
            augmented_failure_sampling_ratio=aug_ratio,
            default_c_fail=default_c_fail,
        ),
        checkpoint=dataclasses.replace(
            base_config.checkpoint,
            overwrite=True,
            resume=False,
            resume_checkpoint_path=None,
        ),
        logging=dataclasses.replace(
            base_config.logging,
            wandb_run_name=exp_name,
        ),
    )

    print("=== Variant Config ===")
    print(f"exp_name={cfg.exp_name}")
    print(f"disable_checkpointing={cfg.disable_checkpointing}")
    print(f"target_task={cfg.data.target_task}")
    print(f"treat_other_tasks_as_failure={cfg.data.treat_other_tasks_as_failure}")
    print(f"success_sampling_ratio={cfg.data.success_sampling_ratio}")
    print(f"augmented_failure_sampling_ratio={cfg.data.augmented_failure_sampling_ratio}")
    print(f"default_c_fail={cfg.data.default_c_fail}")
    print(f"overfit_one_batch={cfg.overfit_one_batch}")
    print(f"num_train_steps={cfg.num_train_steps}")
    print(f"num_workers={cfg.num_workers}")
    print(f"num_steps_per_validation={cfg.num_steps_per_validation}")
    print(f"num_validation_batches={cfg.num_validation_batches}")
    print("======================")

    train(cfg)


if __name__ == "__main__":
    main()
