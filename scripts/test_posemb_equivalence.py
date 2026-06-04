#!/usr/bin/env python3
"""Compare equivalent-looking timestep sinusoid inputs in JAX."""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp


def make_scale(embedding_dim: int, min_period: float, max_period: float):
    if embedding_dim % 2 != 0:
        raise ValueError("embedding_dim must be divisible by 2")
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    return 1.0 / period * 2 * jnp.pi


def compare_1d(pos, scale):
    einsum_highest = jnp.einsum("i,j->ij", pos, scale, precision=jax.lax.Precision.HIGHEST)
    broadcast_explicit = pos[:, None] * scale[None, :]
    broadcast_ellipsis = pos[..., None] * scale
    print("1D shapes:", einsum_highest.shape, broadcast_explicit.shape, broadcast_ellipsis.shape)
    print("einsum_highest vs pos[:, None] * scale[None, :] max_abs_diff:", float(jnp.max(jnp.abs(einsum_highest - broadcast_explicit))))
    print("einsum_highest vs pos[..., None] * scale max_abs_diff:", float(jnp.max(jnp.abs(einsum_highest - broadcast_ellipsis))))
    print("pos[:, None] * scale[None, :] vs pos[..., None] * scale max_abs_diff:", float(jnp.max(jnp.abs(broadcast_explicit - broadcast_ellipsis))))
    print("allclose(einsum_highest, ellipsis, atol=0, rtol=0):", bool(jnp.allclose(einsum_highest, broadcast_ellipsis, atol=0, rtol=0)))


def compare_2d(pos, scale):
    broadcast_ellipsis = pos[..., None] * scale
    broadcast_explicit = pos[:, :, None] * scale[None, None, :]
    print("2D shapes:", broadcast_ellipsis.shape, broadcast_explicit.shape)
    print("pos[..., None] * scale vs pos[:, :, None] * scale[None, None, :] max_abs_diff:", float(jnp.max(jnp.abs(broadcast_ellipsis - broadcast_explicit))))
    print("allclose(ellipsis, explicit, atol=0, rtol=0):", bool(jnp.allclose(broadcast_ellipsis, broadcast_explicit, atol=0, rtol=0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    args = parser.parse_args()

    dtype = jnp.float32 if args.dtype == "float32" else jnp.bfloat16
    scale = make_scale(args.embedding_dim, min_period=4e-3, max_period=4.0).astype(dtype)
    key = jax.random.key(args.seed)
    pos_1d = jax.random.uniform(key, (args.batch_size,), minval=0.001, maxval=1.0, dtype=dtype)
    pos_2d = jax.random.uniform(key, (args.batch_size, args.horizon), minval=0.001, maxval=1.0, dtype=dtype)

    print("dtype:", args.dtype)
    compare_1d(pos_1d, scale)
    compare_2d(pos_2d, scale)


if __name__ == "__main__":
    main()
