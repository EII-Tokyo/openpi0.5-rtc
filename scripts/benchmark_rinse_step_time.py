import dataclasses
import functools
import time

import jax
import tyro
import wandb

import openpi.training.checkpoints as _checkpoints
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import run_rinse_fullft_jax
import train as train_script


RINSE_REPO_IDS = [
    "lyl472324464/2026-04-21_direction-lerobot-with-rinse",
    "lyl472324464/2026-04-21_direction_2-lerobot-with-rinse",
    "lyl472324464/2026-04-21_direction_havent_cap-lerobot-with-rinse",
    "lyl472324464/2026-04-21_direction_havent_cap_water-lerobot-with-rinse",
    "lyl472324464/2026-04-23_direction_have_cap_water-lerobot-with-rinse",
    "lyl472324464/2026-04-23_direction_havent_cap_water-lerobot-with-rinse",
    "lyl472324464/2026-04-27_direction_have_cap_water2-lerobot-with-rinse",
    "lyl472324464/2026-04-27direction_have_cap_water-lerobot-with-rinse",
    "lyl472324464/2026-04-28_direction_have_cap_water-lerobot-with-rinse",
    "lyl472324464/2026-04-28_direction_have_cap_water2-lerobot-with-rinse",
    "lyl472324464/2026-04-28_water1-lerobot-with-rinse",
    "lyl472324464/2026.03.18_twist-and-water_one_no_cap-with-rinse",
    "lyl472324464/2026.03.30_twist-and-water_two_have_cap-with-rinse",
]


@dataclasses.dataclass
class Args:
    exp_name: str = "bench_rinse_cam4_hist6_stride5_bs64"
    batch_size: int = 64
    num_workers: int = 4
    fsdp_devices: int = 8
    video_memory_num_frames: int = 6
    video_memory_stride_seconds: float = 5.0
    timed_steps: int = 5
    checkpoint_base_dir: str = "/workspace/openpi0.5-rtc/checkpoints"
    assets_base_dir: str = "/workspace/openpi0.5-rtc/assets"
    wandb_enabled: bool = True


def main() -> None:
    args = tyro.cli(Args)
    train_script.init_logging()

    cfg = run_rinse_fullft_jax.build_config(
        run_rinse_fullft_jax.Args(
            exp_name=args.exp_name,
            batch_size=args.batch_size,
            num_train_steps=args.timed_steps + 1,
            num_workers=args.num_workers,
            checkpoint_base_dir=args.checkpoint_base_dir,
            assets_base_dir=args.assets_base_dir,
            wandb_enabled=args.wandb_enabled,
            overwrite=True,
            resume=False,
            fsdp_devices=args.fsdp_devices,
            video_memory_num_frames=args.video_memory_num_frames,
            video_memory_stride_seconds=args.video_memory_stride_seconds,
        )
    )
    cfg = dataclasses.replace(
        cfg,
        data=dataclasses.replace(cfg.data, repo_ids=RINSE_REPO_IDS),
    )

    rng = jax.random.key(cfg.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(cfg.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    _checkpoints.initialize_checkpoint_dir(
        cfg.checkpoint_dir,
        keep_period=cfg.keep_period,
        overwrite=True,
        resume=False,
    )
    train_script.init_wandb(cfg, resuming=False, enabled=cfg.wandb_enabled)

    t0 = time.perf_counter()
    data_loader = _data_loader.create_data_loader(cfg, sharding=data_sharding, shuffle=True)
    data_iter = iter(data_loader)
    batch = next(data_iter)
    jax.block_until_ready(batch)
    first_batch_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    train_state, train_state_sharding = train_script.init_train_state(cfg, init_rng, mesh, resume=False)
    jax.block_until_ready(train_state)
    init_state_seconds = time.perf_counter() - t1

    ptrain_step = jax.jit(
        functools.partial(train_script.train_step, cfg),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    step_times = []
    info_history = []
    for step_idx in range(args.timed_steps):
        start = time.perf_counter()
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        jax.block_until_ready(train_state)
        info = jax.device_get(info)
        elapsed = time.perf_counter() - start
        step_times.append(elapsed)
        info_history.append(info)
        if step_idx != args.timed_steps - 1:
            batch = next(data_iter)

    warmup_seconds = step_times[0]
    steady_step_seconds = step_times[1:] if len(step_times) > 1 else []
    steady_mean_seconds = sum(steady_step_seconds) / len(steady_step_seconds) if steady_step_seconds else warmup_seconds

    summary = {
        "first_batch_seconds": first_batch_seconds,
        "init_state_seconds": init_state_seconds,
        "warmup_step_seconds": warmup_seconds,
        "steady_mean_step_seconds": steady_mean_seconds,
        "steady_min_step_seconds": min(steady_step_seconds) if steady_step_seconds else warmup_seconds,
        "steady_max_step_seconds": max(steady_step_seconds) if steady_step_seconds else warmup_seconds,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "video_memory_num_frames": args.video_memory_num_frames,
        "video_memory_stride_seconds": args.video_memory_stride_seconds,
        "repo_count": len(RINSE_REPO_IDS),
    }

    print("[benchmark] config", summary)
    print("[benchmark] step_times_seconds", step_times)
    print("[benchmark] last_info", info_history[-1])
    if cfg.wandb_enabled:
        wandb.log(summary, step=0)
        wandb.finish()


if __name__ == "__main__":
    main()
