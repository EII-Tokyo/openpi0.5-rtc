#!/usr/bin/env python3
"""Minimal smoke test for Pi0.compute_loss.

Default uses the tiny dummy Pi0 variants so it runs quickly and does not load a
checkpoint. Pass --config to instantiate a real training config only when you
want to inspect real token/image/action shapes.
"""

from __future__ import annotations

import argparse
import dataclasses
import time

import jax
import jax.numpy as jnp

from openpi.models import pi0_config


def _make_config(args):
    if args.config:
        from openpi.training import config as train_config

        cfg = train_config.get_config(args.config).model
        if args.training_time_rtc is not None:
            cfg = dataclasses.replace(cfg, training_time_rtc=args.training_time_rtc)
        if args.rtc_max_delay is not None:
            cfg = dataclasses.replace(cfg, rtc_max_delay=args.rtc_max_delay)
        return cfg

    return pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
        max_token_len=args.max_token_len,
        image_resolution=(args.image_resolution, args.image_resolution),
        image_keys=tuple(args.image_keys.split(",")),
        training_time_rtc=bool(args.training_time_rtc),
        rtc_max_delay=args.rtc_max_delay if args.rtc_max_delay is not None else 10,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="", help="Optional training config name; empty uses tiny dummy Pi0.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--train", action="store_true", help="Pass train=True to compute_loss.")
    parser.add_argument("--training-time-rtc", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--rtc-max-delay", type=int, default=None)

    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--max-token-len", type=int, default=16)
    parser.add_argument("--image-resolution", type=int, default=224)
    parser.add_argument("--image-keys", default="cam_high")
    args = parser.parse_args()

    rng = jax.random.key(args.seed)
    cfg = _make_config(args)
    model = cfg.create(rng)
    obs = cfg.fake_obs(batch_size=args.batch_size)
    actions = jax.random.normal(
        jax.random.key(args.seed + 1),
        (args.batch_size, cfg.action_horizon, cfg.action_dim),
        dtype=jnp.float32,
    )
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
    suffix_tokens, suffix_mask, suffix_ar_mask, _ = model.embed_suffix(
        obs,
        actions,
        jnp.ones((args.batch_size,), dtype=jnp.float32) * 0.5,
    )

    def loss_fn(m, loss_rng):
        return m.compute_loss(loss_rng, obs, actions, train=args.train)

    run_loss = loss_fn
    if args.jit:
        from openpi.shared import nnx_utils

        run_loss = nnx_utils.module_jit(loss_fn)

    start = time.perf_counter()
    loss = run_loss(model, jax.random.key(args.seed + 2))
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    print(f"config={args.config or 'dummy'}")
    print(f"training_time_rtc={cfg.training_time_rtc} rtc_max_delay={cfg.rtc_max_delay} train={args.train} jit={args.jit}")
    print(f"obs.state.shape={obs.state.shape}")
    print(f"actions.shape={actions.shape}")
    print(f"prefix_tokens.shape={prefix_tokens.shape}")
    print(f"prefix_mask.shape={prefix_mask.shape}")
    print(f"prefix_ar_mask.shape={prefix_ar_mask.shape}")
    print(f"suffix_tokens.shape={suffix_tokens.shape}")
    print(f"suffix_mask.shape={suffix_mask.shape}")
    print(f"suffix_ar_mask.shape={suffix_ar_mask.shape}")
    print(f"combined_ar_mask.shape={(prefix_ar_mask.shape[0] + suffix_ar_mask.shape[0],)}")
    print(f"loss.shape={loss.shape}")
    print(f"loss.mean={float(jnp.mean(loss)):.6f}")
    print(f"loss.min={float(jnp.min(loss)):.6f} loss.max={float(jnp.max(loss)):.6f}")
    print(f"elapsed_ms={elapsed_ms:.2f}")


if __name__ == "__main__":
    main()
