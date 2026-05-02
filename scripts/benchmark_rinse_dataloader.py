import dataclasses
import time

import jax
import tyro

import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import run_rinse_fullft_jax
import train as train_script


@dataclasses.dataclass
class Args:
    batch_size: int = 64
    num_workers: int = 4
    fsdp_devices: int = 8
    video_memory_num_frames: int = 6
    video_memory_stride_seconds: float = 5.0
    timed_batches: int = 10


def main() -> None:
    args = tyro.cli(Args)
    train_script.init_logging()

    cfg = run_rinse_fullft_jax.build_config(
        run_rinse_fullft_jax.Args(
            exp_name=f"dataloader_bs{args.batch_size}_nw{args.num_workers}",
            batch_size=args.batch_size,
            num_train_steps=1,
            num_workers=args.num_workers,
            wandb_enabled=False,
            overwrite=True,
            resume=False,
            fsdp_devices=args.fsdp_devices,
            video_memory_num_frames=args.video_memory_num_frames,
            video_memory_stride_seconds=args.video_memory_stride_seconds,
        )
    )

    mesh = sharding.make_mesh(cfg.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    t0 = time.perf_counter()
    data_loader = _data_loader.create_data_loader(cfg, sharding=data_sharding, shuffle=True)
    data_iter = iter(data_loader)
    first_batch = next(data_iter)
    jax.block_until_ready(first_batch)
    first_batch_seconds = time.perf_counter() - t0

    batch_times = []
    for _ in range(args.timed_batches):
        start = time.perf_counter()
        batch = next(data_iter)
        jax.block_until_ready(batch)
        batch_times.append(time.perf_counter() - start)

    steady_mean = sum(batch_times) / len(batch_times)
    steady_min = min(batch_times)
    steady_max = max(batch_times)

    summary = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "video_memory_num_frames": args.video_memory_num_frames,
        "video_memory_stride_seconds": args.video_memory_stride_seconds,
        "first_batch_seconds": first_batch_seconds,
        "steady_mean_batch_seconds": steady_mean,
        "steady_min_batch_seconds": steady_min,
        "steady_max_batch_seconds": steady_max,
        "timed_batches": args.timed_batches,
        "batches_per_second": 1.0 / steady_mean if steady_mean > 0 else float("inf"),
        "samples_per_second": args.batch_size / steady_mean if steady_mean > 0 else float("inf"),
    }

    print("[benchmark] dataloader", summary)
    print("[benchmark] batch_times_seconds", batch_times)


if __name__ == "__main__":
    main()
