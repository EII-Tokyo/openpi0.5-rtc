#!/usr/bin/env python3
"""Smoke-test Pi0.sample_action_chunk_with_training_time_rtc."""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from openpi.models import pi0_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff-delay-steps", type=int, default=10)
    parser.add_argument("--denoising-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--max-token-len", type=int, default=16)
    parser.add_argument("--jit", action="store_true")
    args = parser.parse_args()

    cfg = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
        max_token_len=args.max_token_len,
        training_time_rtc=True,
    )
    model = cfg.create(jax.random.key(args.seed))
    obs = cfg.fake_obs(batch_size=args.batch_size)

    noise = jax.random.normal(
        jax.random.key(args.seed + 1),
        (args.batch_size, args.action_horizon, args.action_dim),
    )
    action_prefix = jax.random.normal(
        jax.random.key(args.seed + 2),
        (args.batch_size, args.action_horizon, args.action_dim),
    )

    sample_fn = model.sample_action_chunk_with_training_time_rtc
    if args.jit:
        from openpi.shared import nnx_utils
        sample_fn = nnx_utils.module_jit(sample_fn)

    start = time.perf_counter()
    actions = sample_fn(
        jax.random.key(args.seed + 3),
        obs,
        action_prefix=action_prefix,
        handoff_delay_steps=args.handoff_delay_steps,
        denoising_steps=args.denoising_steps,
        noise=noise,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    prefix = actions[:, : args.handoff_delay_steps]
    expected_prefix = action_prefix[:, : args.handoff_delay_steps]
    prefix_max_abs_diff = float(jnp.max(jnp.abs(prefix - expected_prefix))) if args.handoff_delay_steps > 0 else 0.0

    suffix = actions[:, args.handoff_delay_steps :]
    suffix_noise = noise[:, args.handoff_delay_steps :]
    suffix_changed_from_noise = float(jnp.max(jnp.abs(suffix - suffix_noise))) if suffix.size else 0.0

    print(f"actions_shape={tuple(actions.shape)}")
    print(f"handoff_delay_steps={args.handoff_delay_steps}")
    print(f"denoising_steps={args.denoising_steps}")
    print(f"jit={args.jit}")
    print(f"elapsed_ms={elapsed_ms:.2f}")
    print(f"prefix_max_abs_diff={prefix_max_abs_diff}")
    print(f"suffix_max_abs_diff_from_initial_noise={suffix_changed_from_noise}")
    if prefix_max_abs_diff != 0.0:
        raise AssertionError("prefix was modified")
    print("training_time_rtc_sampler_ok")


if __name__ == "__main__":
    main()
