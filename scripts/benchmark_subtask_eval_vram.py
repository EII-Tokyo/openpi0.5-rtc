#!/usr/bin/env python3
"""Measure GPU memory around the periodic subtask val block (train.py ~345–363).

Runs: load TrainConfig + dataloader (subtask split) → init_train_state → build Policy →
`run_subtask_val_eval`, printing `nvidia-smi` memory at each stage.

Usage (from repo root, same flags as train.py for config):

  uv run python scripts/benchmark_subtask_eval_vram.py twist_and_static_mixture_lora \\
    --exp-name bench_subtask_vram --batch-size 64 --overwrite --no-wandb-enabled \\
    --fsdp-devices 1

Optional: cap how many val frames run through infer (faster smoke; peak VRAM may be lower):

  BENCHMARK_SUBTASK_EVAL_CAP=32 uv run python scripts/benchmark_subtask_eval_vram.py ...
"""

from __future__ import annotations

import dataclasses
import importlib.util
import logging
import os
import platform
import subprocess
import time
from pathlib import Path

import etils.epath as epath
import jax
import numpy as np

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.subtask_eval as _subtask_eval


def _load_train_helpers():
    train_py = Path(__file__).resolve().parent / "train.py"
    spec = importlib.util.spec_from_file_location("_openpi_train_for_bench", train_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {train_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.init_logging, mod.init_train_state


def _gpu_mem_mib() -> float | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        line = out.splitlines()[0]
        used, total = (x.strip() for x in line.split(","))
        return float(used)
    except Exception:
        return None


def _log_mem(tag: str) -> None:
    jax.block_until_ready(())
    time.sleep(0.3)
    mib = _gpu_mem_mib()
    if mib is None:
        logging.info("benchmark_subtask_vram: [%s] nvidia-smi unavailable", tag)
    else:
        logging.info("benchmark_subtask_vram: [%s] GPU memory.used ≈ %.0f MiB", tag, mib)


def main(config: _config.TrainConfig) -> None:
    init_logging, init_train_state = _load_train_helpers()
    init_logging()
    logging.info("Running on: %s", platform.node())

    if config.fsdp_devices != 1:
        config = dataclasses.replace(config, fsdp_devices=1)
        logging.warning("benchmark_subtask_vram: forced fsdp_devices=1 (subtask eval path matches train.py)")

    if not config.subtask_eval_enabled:
        logging.warning(
            "benchmark_subtask_vram: subtask_eval_enabled is False; dataloader will not build val split. "
            "Use a config with subtask_eval_enabled=True."
        )

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"batch_size {config.batch_size} must be divisible by device count {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    logging.info("benchmark_subtask_vram: creating data loader (includes subtask split when enabled)...")
    t0 = time.perf_counter()
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=False,
    )
    logging.info(
        "benchmark_subtask_vram: create_data_loader done in %.2fs",
        time.perf_counter() - t0,
    )

    se_split = getattr(data_loader, "subtask_eval_split", None)
    se_outer = getattr(data_loader, "subtask_eval_outer_dataset", None)
    se_i2c: dict[int, int] = getattr(data_loader, "subtask_eval_index_to_class", None) or {}

    if se_split is None or se_outer is None or se_split.val_indices.size == 0:
        raise RuntimeError(
            "No subtask eval split on this loader (subtask_eval_enabled or val_indices empty). "
            "Pick a config with subtask_eval_enabled=True and canonical labels in data."
        )

    _log_mem("after_data_loader")

    logging.info("benchmark_subtask_vram: init_train_state (loads base weights onto device)...")
    t1 = time.perf_counter()
    with sharding.set_mesh(mesh):
        train_state, _ = init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(train_state)
    logging.info("benchmark_subtask_vram: init_train_state done in %.2fs", time.perf_counter() - t1)
    _log_mem("after_train_state")

    step = int(os.environ.get("BENCHMARK_SUBTASK_STEP", "1000"))
    rng_eval = jax.random.fold_in(train_rng, 424242 + step)
    logging.info("benchmark_subtask_vram: eval_policy_from_train_state (same as train loop)...")
    t2 = time.perf_counter()
    with sharding.set_mesh(mesh):
        policy = _subtask_eval.eval_policy_from_train_state(config, train_state, policy_rng=rng_eval)
    jax.block_until_ready(())
    logging.info("benchmark_subtask_vram: policy built in %.2fs", time.perf_counter() - t2)
    _log_mem("after_policy")

    rng_sub = np.random.default_rng(int(config.seed) + step)
    vix = _subtask_eval.subsample_val_indices(
        se_split.val_indices,
        index_to_class=se_i2c,
        max_per_class=config.subtask_eval_max_samples_per_class,
        rng=rng_sub,
    )
    cap = int(os.environ.get("BENCHMARK_SUBTASK_EVAL_CAP", "0"))
    if cap > 0:
        vix = np.asarray(vix[:cap], dtype=np.int64)
        logging.info("benchmark_subtask_vram: capped val frames to BENCHMARK_SUBTASK_EVAL_CAP=%d", cap)

    logging.info(
        "benchmark_subtask_vram: run_subtask_val_eval (infer_batch_size=%d, val_frames=%d)...",
        config.subtask_eval_batch_size,
        int(vix.size),
    )
    t3 = time.perf_counter()
    metrics, _ = _subtask_eval.run_subtask_val_eval(
        policy=policy,
        outer_torch_dataset=se_outer,
        val_indices=vix,
        canonical_pairs=se_split.canonical_pairs,
        index_to_class=se_i2c,
        action_horizon=config.model.action_horizon,
        infer_batch_size=config.subtask_eval_batch_size,
    )
    jax.block_until_ready(())
    logging.info("benchmark_subtask_vram: run_subtask_val_eval done in %.2fs", time.perf_counter() - t3)
    _log_mem("after_run_subtask_val_eval")

    acc = metrics.get("subtask_val/accuracy", 0.0)
    logging.info(
        "benchmark_subtask_vram: subtask_val/accuracy=%.4f total=%s (sanity)",
        acc,
        metrics.get("subtask_val/total"),
    )


if __name__ == "__main__":
    main(_config.cli())
