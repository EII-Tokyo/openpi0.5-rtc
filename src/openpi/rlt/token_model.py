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
    num_heads: int = 4


def _linear_params(rng: jax.Array, in_dim: int, out_dim: int) -> Params:
    scale = np.sqrt(2.0 / float(in_dim + out_dim))
    return {
        "w": jax.random.normal(rng, (in_dim, out_dim), dtype=jnp.float32) * scale,
        "b": jnp.zeros((out_dim,), dtype=jnp.float32),
    }


def _linear(params: Params, x: jax.Array) -> jax.Array:
    return x @ params["w"] + params["b"]


def _layer_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + eps)


def _transformer_layer_params(rng: jax.Array, dim: int, hidden_dim: int) -> Params:
    qkv_rng, out_rng, mlp0_rng, mlp1_rng = jax.random.split(rng, 4)
    return {
        "qkv": _linear_params(qkv_rng, dim, dim * 3),
        "out": _linear_params(out_rng, dim, dim),
        "mlp0": _linear_params(mlp0_rng, dim, hidden_dim),
        "mlp1": _linear_params(mlp1_rng, hidden_dim, dim),
    }


def _attention(params: Params, x: jax.Array, mask: jax.Array | None, num_heads: int) -> jax.Array:
    batch, seq_len, dim = x.shape
    if dim % num_heads != 0:
        raise ValueError(f"token_dim={dim} must be divisible by num_heads={num_heads}")
    head_dim = dim // num_heads
    qkv = _linear(params["qkv"], x)
    q, k, v = jnp.split(qkv, 3, axis=-1)
    q = q.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(jnp.asarray(head_dim, dtype=jnp.float32))
    if mask is not None:
        logits = jnp.where(mask[:, None, None, :], logits, -1e9)
    weights = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum("bhqk,bhkd->bhqd", weights, v).transpose(0, 2, 1, 3).reshape(batch, seq_len, dim)
    return _linear(params["out"], out)


def _transformer(params: list[Params], x: jax.Array, mask: jax.Array | None, config: RLTTokenConfig) -> jax.Array:
    for layer in params:
        x = x + _attention(layer, _layer_norm(x), mask, config.num_heads)
        x = x + _linear(layer["mlp1"], jax.nn.gelu(_linear(layer["mlp0"], _layer_norm(x))))
    return _layer_norm(x)


def init_token_params(rng: jax.Array, config: RLTTokenConfig) -> Params:
    keys = jax.random.split(rng, 4 + config.num_layers * 2)
    enc_layer_keys = keys[4 : 4 + config.num_layers]
    dec_layer_keys = keys[4 + config.num_layers :]
    return {
        "input_proj": _linear_params(keys[0], config.input_dim, config.token_dim),
        "encoder_query": jax.random.normal(keys[1], (config.token_dim,), dtype=jnp.float32) * 0.02,
        "decoder_query": jax.random.normal(keys[2], (config.token_dim,), dtype=jnp.float32) * 0.02,
        "output_proj": _linear_params(keys[3], config.token_dim, config.input_dim),
        "encoder_layers": [_transformer_layer_params(key, config.token_dim, config.hidden_dim) for key in enc_layer_keys],
        "decoder_layers": [_transformer_layer_params(key, config.token_dim, config.hidden_dim) for key in dec_layer_keys],
    }


def encode(params: Params, embeddings: jax.Array, mask: jax.Array, config: RLTTokenConfig | None = None) -> jax.Array:
    if config is None:
        config = RLTTokenConfig(input_dim=embeddings.shape[-1], token_dim=params["encoder_query"].shape[-1])
    mask = mask.astype(jnp.bool_)
    x = _linear(params["input_proj"], embeddings.astype(jnp.float32))
    query = jnp.broadcast_to(params["encoder_query"], (x.shape[0], 1, x.shape[-1]))
    x = jnp.concatenate([x, query], axis=1)
    mask = jnp.concatenate([mask, jnp.ones((mask.shape[0], 1), dtype=jnp.bool_)], axis=1)
    x = _transformer(params["encoder_layers"], x, mask, config)
    return x[:, -1]


def decode(params: Params, token: jax.Array, seq_len: int, config: RLTTokenConfig | None = None) -> jax.Array:
    if config is None:
        config = RLTTokenConfig(input_dim=params["output_proj"]["b"].shape[-1], token_dim=token.shape[-1])
    query = jnp.broadcast_to(params["decoder_query"], (token.shape[0], seq_len, token.shape[-1]))
    x = query + token[:, None, :]
    x = _transformer(params["decoder_layers"], x, None, config)
    return _linear(params["output_proj"], x)


def reconstruction_loss(params: Params, embeddings: jax.Array, mask: jax.Array, config: RLTTokenConfig | None = None) -> tuple[jax.Array, dict[str, jax.Array]]:
    if config is None:
        config = RLTTokenConfig(input_dim=embeddings.shape[-1], token_dim=params["encoder_query"].shape[-1])
    embeddings = jax.lax.stop_gradient(embeddings.astype(jnp.float32))
    mask_f = mask.astype(jnp.float32)
    token = encode(params, embeddings, mask, config)
    recon = decode(params, token, embeddings.shape[1], config)
    sq = jnp.mean(jnp.square(recon - embeddings), axis=-1)
    loss = jnp.sum(sq * mask_f) / jnp.maximum(jnp.sum(mask_f), 1.0)
    return loss, {"token_norm": jnp.mean(jnp.linalg.norm(token, axis=-1)), "reconstruction_loss": loss}
