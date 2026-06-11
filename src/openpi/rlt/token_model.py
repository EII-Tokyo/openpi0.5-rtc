from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


Params = dict[str, Any]


@dataclasses.dataclass(frozen=True)
class RLTTokenConfig:
    input_dim: int
    token_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2


def _linear_params(rng: jax.Array, in_dim: int, out_dim: int) -> Params:
    scale = np.sqrt(2.0 / float(in_dim + out_dim))
    return {
        "w": jax.random.normal(rng, (in_dim, out_dim), dtype=jnp.float32) * scale,
        "b": jnp.zeros((out_dim,), dtype=jnp.float32),
    }


def _linear(params: Params, x: jax.Array) -> jax.Array:
    return x @ params["w"] + params["b"]


def _mlp_params(rng: jax.Array, dims: list[int]) -> list[Params]:
    keys = jax.random.split(rng, len(dims) - 1)
    return [_linear_params(key, dims[i], dims[i + 1]) for i, key in enumerate(keys)]


def _mlp(params: list[Params], x: jax.Array, *, final_activation: bool = False) -> jax.Array:
    for index, layer in enumerate(params):
        x = _linear(layer, x)
        if index != len(params) - 1 or final_activation:
            x = jax.nn.gelu(x)
    return x


def init_token_params(rng: jax.Array, config: RLTTokenConfig) -> Params:
    enc_rng, dec_rng = jax.random.split(rng)
    hidden = [config.hidden_dim] * max(1, config.num_layers)
    return {
        "encoder": _mlp_params(enc_rng, [config.input_dim, *hidden, config.token_dim]),
        "decoder": _mlp_params(dec_rng, [config.token_dim, *hidden, config.input_dim]),
    }


def encode(params: Params, embeddings: jax.Array, mask: jax.Array) -> jax.Array:
    mask = mask.astype(jnp.float32)
    denom = jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), 1.0)
    pooled = jnp.sum(embeddings * mask[..., None], axis=1) / denom
    return _mlp(params["encoder"], pooled)


def decode(params: Params, token: jax.Array, seq_len: int) -> jax.Array:
    recon = _mlp(params["decoder"], token)
    return jnp.repeat(recon[:, None, :], seq_len, axis=1)


def reconstruction_loss(params: Params, embeddings: jax.Array, mask: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
    embeddings = jax.lax.stop_gradient(embeddings.astype(jnp.float32))
    mask = mask.astype(jnp.float32)
    token = encode(params, embeddings, mask)
    recon = decode(params, token, embeddings.shape[1])
    sq = jnp.mean(jnp.square(recon - embeddings), axis=-1)
    loss = jnp.sum(sq * mask) / jnp.maximum(jnp.sum(mask), 1.0)
    return loss, {"token_norm": jnp.mean(jnp.linalg.norm(token, axis=-1)), "reconstruction_loss": loss}
