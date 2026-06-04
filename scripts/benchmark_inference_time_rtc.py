#!/usr/bin/env python3
"""Benchmark Pi0.sample_action_chunk_with_inference_time_rtc for several denoising_steps values."""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from openpi.models import pi0_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--denoising-steps", type=int, nargs="+", default=[2, 5, 10])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jit", action="store_true")
    args = parser.parse_args()

    cfg = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=32,
        action_horizon=50,
        max_token_len=16,
    )
    model = cfg.create(jax.random.key(args.seed))
    obs = cfg.fake_obs(batch_size=1)
    prev_action = jax.random.normal(jax.random.key(args.seed + 1), (1, 50, 32))

    sample_fn = model.sample_action_chunk_with_inference_time_rtc
    if args.jit:
        from openpi.shared import nnx_utils
        sample_fn = nnx_utils.module_jit(sample_fn)

    # Warm one tiny call so model init side effects are not mixed into every row equally.
    sample_fn(jax.random.key(args.seed + 100), prev_action, obs, denoising_steps=1, replan_start_step=25, handoff_delay_steps=10, guidance_scale=8.0)

    for steps in args.denoising_steps:
        times = []
        output_shape = None
        for i in range(args.repeat):
            start = time.perf_counter()
            out = sample_fn(
                jax.random.key(args.seed + 1000 + steps * 17 + i),
                prev_action,
                obs,
                denoising_steps=steps,
                replan_start_step=25,
                handoff_delay_steps=10,
                guidance_scale=8.0,
            )
            jax.block_until_ready(out)
            elapsed_ms = (time.perf_counter() - start) * 1000
            output_shape = tuple(out.shape)
            times.append(elapsed_ms)
        print(
            f"denoising_steps={steps} jit={args.jit} repeat={args.repeat} "
            f"shape={output_shape} avg_ms={sum(times) / len(times):.2f} "
            f"min_ms={min(times):.2f} max_ms={max(times):.2f}"
        )


if __name__ == "__main__":
    main()
