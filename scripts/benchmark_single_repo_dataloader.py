import dataclasses
import statistics
import time
import traceback

import jax
import tyro

import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import run_rinse_fullft_jax


@dataclasses.dataclass
class Args:
    repo_id: str = "lyl472324464/2026-04-21_direction-lerobot-with-rinse"
    batch_size: int = 128
    num_workers: int = 4
    timed_batches: int = 8
    warmup_batches: int = 2
    fsdp_devices: int = 8
    video_memory_num_frames: int = 6
    video_memory_stride_seconds: float = 5.0
    checkpoint_base_dir: str = "/workspace/openpi0.5-rtc/checkpoints"
    assets_base_dir: str = "/workspace/openpi0.5-rtc/assets"


def _time_loader(iterator, batches: int) -> list[float]:
    times = []
    for _ in range(batches):
        start = time.perf_counter()
        batch = next(iterator)
        jax.block_until_ready(batch)
        times.append(time.perf_counter() - start)
    return times


def _summary(times: list[float]) -> dict[str, float]:
    return {
        "count": len(times),
        "mean": float(statistics.mean(times)),
        "median": float(statistics.median(times)),
        "min": float(min(times)),
        "max": float(max(times)),
    }


def main() -> None:
    args = tyro.cli(Args)
    cfg = run_rinse_fullft_jax.build_config(
        run_rinse_fullft_jax.Args(
            exp_name="debug_single_repo_dataloader",
            batch_size=args.batch_size,
            num_train_steps=1,
            num_workers=args.num_workers,
            checkpoint_base_dir=args.checkpoint_base_dir,
            assets_base_dir=args.assets_base_dir,
            wandb_enabled=False,
            overwrite=True,
            resume=False,
            fsdp_devices=args.fsdp_devices,
            video_memory_num_frames=args.video_memory_num_frames,
            video_memory_stride_seconds=args.video_memory_stride_seconds,
        )
    )
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, repo_ids=[args.repo_id]))

    total_batches = args.warmup_batches + args.timed_batches
    mesh = sharding.make_mesh(cfg.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    raw_loader = _data_loader.create_data_loader(
        cfg,
        sharding=None,
        shuffle=True,
        num_batches=total_batches,
        framework="pytorch",
    )

    raw_times = _time_loader(iter(raw_loader._data_loader), total_batches)
    timed_raw = raw_times[args.warmup_batches :]
    print("[dataloader] config", dataclasses.asdict(args))
    print("[dataloader] raw_batch_times_seconds", raw_times)
    print("[dataloader] raw_batch_summary", _summary(timed_raw))

    try:
        sharded_loader = _data_loader.create_data_loader(
            cfg,
            sharding=data_sharding,
            shuffle=True,
            num_batches=total_batches,
            framework="jax",
        )
        sharded_times = _time_loader(iter(sharded_loader._data_loader), total_batches)
        timed_sharded = sharded_times[args.warmup_batches :]
        print("[dataloader] sharded_batch_times_seconds", sharded_times)
        print("[dataloader] sharded_batch_summary", _summary(timed_sharded))
    except Exception as exc:
        print("[dataloader] sharded_error", repr(exc))
        traceback.print_exc()


if __name__ == "__main__":
    main()
