import dataclasses
from pathlib import Path

from flax import nnx
import tyro

import openpi.models.pi0_config as pi0_config
import openpi.training.config as config_lib
from scripts import train_pytorch


@dataclasses.dataclass
class Args:
    exp_name: str
    batch_size: int
    num_train_steps: int = 10
    num_workers: int = 4
    log_interval: int = 1
    save_interval: int = 1000
    gradient_accumulation_steps: int = 1
    pytorch_weight_path: str = "/workspace/pi05_base_pytorch"
    checkpoint_base_dir: str = "/workspace/openpi0.5-rtc/checkpoints"
    assets_base_dir: str = "/workspace/openpi0.5-rtc/assets"
    wandb_enabled: bool = False
    overwrite: bool = True
    resume: bool = False


def build_config(args: Args) -> config_lib.TrainConfig:
    base = config_lib.get_config("eii_rinse_cam4_lora")
    model = dataclasses.replace(
        pi0_config.Pi0Config(pi05=True),
        dtype=base.pytorch_training_precision,
    )
    return dataclasses.replace(
        base,
        exp_name=args.exp_name,
        model=model,
        pytorch_weight_path=args.pytorch_weight_path,
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
        freeze_filter=nnx.Nothing(),
        ema_decay=0.99,
    )


def main() -> None:
    args = tyro.cli(Args)
    cfg = build_config(args)
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    train_pytorch.init_logging()
    train_pytorch.train_loop(cfg)


if __name__ == "__main__":
    main()
