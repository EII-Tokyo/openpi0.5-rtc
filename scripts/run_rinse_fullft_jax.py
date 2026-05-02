import ast
import dataclasses
import os

from flax import nnx
import tyro

def _set_jax_runtime_defaults() -> None:
    # Blackwell/PCIe hosts have been prone to hanging during NCCL clique init.
    # Keep these as defaults for JAX multi-GPU training while still allowing
    # callers to override them explicitly from the environment.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_DEBUG", "WARN")


_set_jax_runtime_defaults()

import openpi.models.pi0_config as pi0_config
import openpi.training.config as config_lib
import train as train_script


@dataclasses.dataclass
class Args:
    exp_name: str
    batch_size: int
    repo_ids: str | None = None
    num_train_steps: int = 10
    num_workers: int = 0
    log_interval: int = 1
    save_interval: int = 1000
    gradient_accumulation_steps: int = 1
    checkpoint_base_dir: str = "/workspace/openpi0.5-rtc/checkpoints"
    assets_base_dir: str = "/workspace/openpi0.5-rtc/assets"
    wandb_enabled: bool = False
    overwrite: bool = True
    resume: bool = False
    fsdp_devices: int = 8
    video_memory_num_frames: int = 1
    video_memory_stride_seconds: float = 1.0


def build_config(args: Args) -> config_lib.TrainConfig:
    base = config_lib.get_config("eii_rinse_cam4_lora")
    repo_ids = None
    if args.repo_ids:
        text = args.repo_ids.strip()
        if text.startswith("["):
            parsed = ast.literal_eval(text)
            if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
                raise ValueError("--repo-ids must be a Python/JSON-style list of strings")
            repo_ids = parsed
        else:
            repo_ids = [part.strip() for part in text.split(",") if part.strip()]
        if not repo_ids:
            raise ValueError("--repo-ids parsed to an empty list")
    model = dataclasses.replace(pi0_config.Pi0Config(pi05=True), dtype=base.model.dtype)
    data = dataclasses.replace(
        base.data,
        repo_ids=repo_ids if repo_ids is not None else base.data.repo_ids,
        video_memory_num_frames=args.video_memory_num_frames,
        video_memory_stride_seconds=args.video_memory_stride_seconds,
    )
    return dataclasses.replace(
        base,
        exp_name=args.exp_name,
        model=model,
        data=data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=args.num_train_steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        checkpoint_base_dir=args.checkpoint_base_dir,
        assets_base_dir=args.assets_base_dir,
        wandb_enabled=args.wandb_enabled,
        overwrite=args.overwrite,
        resume=args.resume,
        fsdp_devices=args.fsdp_devices,
        freeze_filter=nnx.Nothing(),
        ema_decay=0.99,
    )


def main() -> None:
    args = tyro.cli(Args)
    cfg = build_config(args)
    train_script.main(cfg)


if __name__ == "__main__":
    main()
