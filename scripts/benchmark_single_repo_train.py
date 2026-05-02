import dataclasses

import tyro

import run_rinse_fullft_jax
import train as train_script


@dataclasses.dataclass
class Args:
    repo_id: str = "lyl472324464/2026-04-21_direction-lerobot-with-rinse"
    exp_name: str = "debug_single_repo_train"
    batch_size: int = 128
    num_train_steps: int = 8
    num_workers: int = 16
    log_interval: int = 1
    save_interval: int = 1000
    gradient_accumulation_steps: int = 1
    checkpoint_base_dir: str = "/workspace/openpi0.5-rtc/checkpoints"
    assets_base_dir: str = "/workspace/openpi0.5-rtc/assets"
    wandb_enabled: bool = False
    overwrite: bool = True
    resume: bool = False
    fsdp_devices: int = 8
    video_memory_num_frames: int = 6
    video_memory_stride_seconds: float = 5.0


def main() -> None:
    args = tyro.cli(Args)
    cfg = run_rinse_fullft_jax.build_config(
        run_rinse_fullft_jax.Args(
            exp_name=args.exp_name,
            batch_size=args.batch_size,
            num_train_steps=args.num_train_steps,
            num_workers=args.num_workers,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            checkpoint_base_dir=args.checkpoint_base_dir,
            assets_base_dir=args.assets_base_dir,
            wandb_enabled=args.wandb_enabled,
            overwrite=args.overwrite,
            resume=args.resume,
            fsdp_devices=args.fsdp_devices,
            video_memory_num_frames=args.video_memory_num_frames,
            video_memory_stride_seconds=args.video_memory_stride_seconds,
        )
    )
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, repo_ids=[args.repo_id]))
    train_script.main(cfg)


if __name__ == "__main__":
    main()
