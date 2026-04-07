#!/usr/bin/env python3
"""Probe max batch size and latency for Policy.infer_subtask_batch (JAX PI05 + twist Aloha repack)."""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def _make_twist_raw_frame(*, action_horizon: int, rng: np.random.Generator) -> dict:
    """Flat LeRobot-style keys matching twist_and_static repack (no subtask)."""
    d: dict = {}
    for cam in ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"):
        d[f"observation.images.{cam}"] = rng.integers(0, 256, size=(3, 224, 224), dtype=np.uint8)
    d["observation.state"] = rng.normal(size=(14,)).astype(np.float32)
    a = rng.normal(size=(14,)).astype(np.float32)
    d["action"] = np.tile(a, (action_horizon, 1))
    d["prompt"] = "pick up the bottle"
    return d


@dataclass
class Args:
    config_name: str = "twist_and_static_mixture_full_finetune"
    checkpoint: Path = Path(
        "checkpoints/twist_and_static_mixture_full_finetune/"
        "twist_and_static_mixture_full_finetune_vast_20260405_100600/39999"
    )
    seed: int = 0
    repeats: int = 3
    """Timed runs per batch size (after one warmup call)."""
    max_batch: int = 256
    """Stop increasing batch size after this (inclusive). Use e.g. 32 for a quick probe."""


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = np.random.default_rng(args.seed)

    train_cfg = _config.get_config(args.config_name)
    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    ah = train_cfg.model.action_horizon

    logging.info("Loading checkpoint (JAX) …")
    policy = _policy_config.create_trained_policy(
        train_cfg,
        args.checkpoint,
        repack_transforms=data_config.repack_transforms,
    )

    base = _make_twist_raw_frame(action_horizon=ah, rng=rng)

    print("\n=== infer_subtask_batch: find max batch + latency ===\n", flush=True)
    print(
        f"config={args.config_name}\ncheckpoint={args.checkpoint}\n"
        f"repeats={args.repeats} (after 1 warmup)\nmax_batch={args.max_batch}\n",
        flush=True,
    )

    bs = 1
    max_ok = 0
    while bs <= args.max_batch:
        obs_list = [copy.deepcopy(base) for _ in range(bs)]
        try:
            policy.infer_subtask_batch(obs_list, batch_size=bs, temperature=0.0)
        except Exception as e:
            print(f"batch_size={bs:3d}  FAIL  {type(e).__name__}: {e}", flush=True)
            break

        times_ms: list[float] = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            policy.infer_subtask_batch(obs_list, batch_size=bs, temperature=0.0)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

        wall = float(np.mean(times_ms))
        per_sample = wall / bs
        print(
            f"batch_size={bs:3d}  wall={wall:8.1f} ms  per_sample={per_sample:7.2f} ms  "
            f"throughput={1000.0 * bs / wall:6.1f} samples/s",
            flush=True,
        )
        max_ok = bs

        if bs < 8:
            bs += 1
        elif bs < 32:
            bs += 4
        elif bs < 64:
            bs += 8
        else:
            bs += 16

    print(f"\nMax batch_size OK (within max_batch={args.max_batch}): {max_ok}\n", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
