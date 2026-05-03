#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import functools
import json
from pathlib import Path
import sys
import time

import jax
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))

import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import train as train_lib


@dataclasses.dataclass(frozen=True)
class Args:
    config_name: str
    exp_name: str = "benchmark_train_step"
    batch_size: int | None = None
    fsdp_devices: int | None = None
    num_workers: int | None = None
    gradient_accumulation_steps: int | None = None
    repo_ids: list[str] | None = None
    overwrite: bool = True


def _measure_next(data_iter) -> tuple[object, float]:
    start = time.perf_counter()
    batch = next(data_iter)
    jax.tree.map(lambda x: x, batch)
    return batch, time.perf_counter() - start


def main(args: Args) -> None:
    config = _config.get_config(args.config_name)
    data_config = config.data
    if args.repo_ids is not None:
        if not hasattr(data_config, "repo_ids"):
            raise ValueError(f"Config {config.name} does not support overriding repo_ids.")
        data_config = dataclasses.replace(data_config, repo_ids=args.repo_ids)
    config = dataclasses.replace(
        config,
        exp_name=args.exp_name,
        wandb_enabled=False,
        overwrite=args.overwrite,
        resume=False,
        batch_size=args.batch_size if args.batch_size is not None else config.batch_size,
        fsdp_devices=args.fsdp_devices if args.fsdp_devices is not None else config.fsdp_devices,
        num_workers=args.num_workers if args.num_workers is not None else config.num_workers,
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
            if args.gradient_accumulation_steps is not None
            else config.gradient_accumulation_steps
        ),
        data=data_config,
    )

    train_lib.init_logging()
    jax.config.update("jax_compilation_cache_dir", str(config.assets_dirs / ".." / ".cache" / "jax"))
    print(
        json.dumps(
            {
                "jax_devices": [str(device) for device in jax.devices()],
                "device_count": jax.device_count(),
            },
            ensure_ascii=False,
        )
    )

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=False,
    )

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    t0 = time.perf_counter()
    data_loader = _data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
    create_loader_s = time.perf_counter() - t0
    data_iter = iter(data_loader)

    batch, first_iter_s = _measure_next(data_iter)
    _, second_iter_s = _measure_next(data_iter)

    t1 = time.perf_counter()
    train_state, train_state_sharding = train_lib.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(train_state)
    init_state_s = time.perf_counter() - t1

    ptrain_step = jax.jit(
        functools.partial(train_lib.train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    with sharding.set_mesh(mesh):
        start = time.perf_counter()
        train_state, info = ptrain_step(train_rng, train_state, batch)
        jax.block_until_ready((train_state, info))
        first_step_s = time.perf_counter() - start

        start = time.perf_counter()
        train_state, info = ptrain_step(train_rng, train_state, batch)
        jax.block_until_ready((train_state, info))
        second_step_s = time.perf_counter() - start

    metrics = jax.device_get(info)
    result = {
        "config_name": config.name,
        "batch_size": config.batch_size,
        "fsdp_devices": config.fsdp_devices,
        "num_workers": config.num_workers,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "create_data_loader_s": create_loader_s,
        "first_dataloader_iter_s": first_iter_s,
        "second_dataloader_iter_s": second_iter_s,
        "init_train_state_s": init_state_s,
        "first_train_step_s": first_step_s,
        "second_train_step_s": second_step_s,
        "last_step_metrics": {k: float(v) for k, v in metrics.items()},
        "batch_info": training_utils.array_tree_to_info(batch),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
