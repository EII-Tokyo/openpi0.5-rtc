from __future__ import annotations

import dataclasses
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp


Params = dict[str, Any]


@dataclasses.dataclass(frozen=True)
class RLTTokenConfig:
    input_dim: int
    token_dim: int | None = None
    hidden_dim: int | None = None
    num_layers: int = 2
    num_heads: int = 8

    def __post_init__(self) -> None:
        if self.token_dim is None:
            object.__setattr__(self, "token_dim", self.input_dim)
        if self.hidden_dim is None:
            object.__setattr__(self, "hidden_dim", self.input_dim * 4)
        if self.token_dim != self.input_dim:
            raise ValueError(
                f"RLT encoder/decoder dim must match VLA embedding dim. "
                f"Got token_dim={self.token_dim}, input_dim={self.input_dim}."
            )
        if self.token_dim % self.num_heads != 0:
            raise ValueError(f"token_dim={self.token_dim} must be divisible by num_heads={self.num_heads}")

    @property
    def dim(self) -> int:
        assert self.token_dim is not None
        return self.token_dim

    @property
    def mlp_dim(self) -> int:
        assert self.hidden_dim is not None
        return self.hidden_dim


def _linear_params(rng: jax.Array, in_dim: int, out_dim: int) -> Params:
    return {
        "w": nn.initializers.lecun_normal(in_axis=-2, out_axis=-1)(rng, (in_dim, out_dim), jnp.float32),
    }


def _linear(params: Params, x: jax.Array) -> jax.Array:
    return x @ params["w"]


def _rms_norm_params(dim: int) -> Params:
    return {"scale": jnp.zeros((dim,), dtype=jnp.float32)}


def _rms_norm(params: Params, x: jax.Array, eps: float = 1e-6) -> jax.Array:
    dtype = x.dtype
    x_f32 = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    x_norm = x_f32 * jax.lax.rsqrt(var + eps)
    return (x_norm * (1.0 + params["scale"])).astype(dtype)


def _transformer_layer_params(rng: jax.Array, dim: int, hidden_dim: int) -> Params:
    qkv_rng, out_rng, gate_rng, up_rng, down_rng = jax.random.split(rng, 5)
    return {
        "pre_attention_norm": _rms_norm_params(dim),
        "qkv": _linear_params(qkv_rng, dim, dim * 3),
        "out": _linear_params(out_rng, dim, dim),
        "pre_ffw_norm": _rms_norm_params(dim),
        "gate": _linear_params(gate_rng, dim, hidden_dim),
        "up": _linear_params(up_rng, dim, hidden_dim),
        "down": _linear_params(down_rng, hidden_dim, dim),
    }


def _apply_rope(x: jax.Array, *, positions: jax.Array, max_wavelength: int = 10_000) -> jax.Array:
    """Applies RoPE positions [B, L] to x [B, L, H, D], matching Gemma."""
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    return res.astype(x.dtype)


def _attention(params: Params, x: jax.Array, attn_mask: jax.Array, positions: jax.Array, num_heads: int) -> jax.Array:
    batch, seq_len, dim = x.shape
    head_dim = dim // num_heads
    qkv = _linear(params["qkv"], x)
    q, k, v = jnp.split(qkv, 3, axis=-1)
    q = q.reshape(batch, seq_len, num_heads, head_dim)
    k = k.reshape(batch, seq_len, num_heads, head_dim)
    v = v.reshape(batch, seq_len, num_heads, head_dim)
    q = _apply_rope(q, positions=positions)
    k = _apply_rope(k, positions=positions)
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    logits = jnp.einsum("bhqd,bhkd->bhqk", q, k, preferred_element_type=jnp.float32) / jnp.sqrt(
        jnp.asarray(head_dim, dtype=jnp.float32)
    )
    logits = jnp.where(attn_mask[:, None, :, :], logits, -2.3819763e38)
    weights = jax.nn.softmax(logits, axis=-1).astype(x.dtype)
    out = jnp.einsum("bhqk,bhkd->bhqd", weights, v).transpose(0, 2, 1, 3).reshape(batch, seq_len, dim)
    return _linear(params["out"], out)


def _feed_forward(params: Params, x: jax.Array) -> jax.Array:
    gate = jax.nn.gelu(_linear(params["gate"], x))
    up = _linear(params["up"], x)
    return _linear(params["down"], gate * up)


def _transformer(
    params: list[Params], x: jax.Array, attn_mask: jax.Array, positions: jax.Array, config: RLTTokenConfig
) -> jax.Array:
    for layer in params:
        x = x + _attention(layer, _rms_norm(layer["pre_attention_norm"], x), attn_mask, positions, config.num_heads)
        x = x + _feed_forward(layer, _rms_norm(layer["pre_ffw_norm"], x))
    return x


def _token_pool_mask(input_mask: jax.Array) -> jax.Array:
    batch, seq_len = input_mask.shape
    total_len = seq_len + 1
    key_mask = jnp.concatenate([input_mask.astype(jnp.bool_), jnp.ones((batch, 1), dtype=jnp.bool_)], axis=1)
    attn_mask = jnp.broadcast_to(key_mask[:, None, :], (batch, total_len, total_len))
    # VLA tokens do not need to read the final RLT token during encoding; the
    # final RLT token pools all valid VLA tokens and itself.
    attn_mask = attn_mask.at[:, :seq_len, -1].set(False)
    return attn_mask


def _causal_mask(input_mask: jax.Array) -> jax.Array:
    batch, seq_len = input_mask.shape
    total_len = seq_len + 1
    valid = jnp.concatenate([jnp.ones((batch, 1), dtype=jnp.bool_), input_mask.astype(jnp.bool_)], axis=1)
    causal = jnp.tril(jnp.ones((total_len, total_len), dtype=jnp.bool_))[None, :, :]
    return causal & valid[:, None, :] & valid[:, :, None]


def _positions_from_mask(mask: jax.Array) -> jax.Array:
    return (jnp.cumsum(mask.astype(jnp.int32), axis=1) - 1).astype(jnp.int32)


def make_debug_masks(mask: jax.Array) -> dict[str, jax.Array]:
    mask = mask.astype(jnp.bool_)
    encoder_valid = jnp.concatenate([mask, jnp.ones((mask.shape[0], 1), dtype=jnp.bool_)], axis=1)
    decoder_valid = jnp.concatenate([jnp.ones((mask.shape[0], 1), dtype=jnp.bool_), mask[:, :-1]], axis=1)
    return {
        "prefix_mask": mask,
        "encoder_valid": encoder_valid,
        "decoder_valid": decoder_valid,
        "encoder_positions": _positions_from_mask(encoder_valid),
        "decoder_positions": _positions_from_mask(decoder_valid),
        "encoder_attn_mask": _token_pool_mask(mask),
        "decoder_attn_mask": _causal_mask(mask)[:, :-1, :-1],
    }


def init_token_params(rng: jax.Array, config: RLTTokenConfig) -> Params:
    keys = jax.random.split(rng, 2 + config.num_layers * 2)
    enc_layer_keys = keys[2 : 2 + config.num_layers]
    dec_layer_keys = keys[2 + config.num_layers :]
    dim = config.dim
    return {
        "encoder_rlt_token": nn.initializers.normal()(keys[0], (dim,), jnp.float32),
        "output_proj": _linear_params(keys[1], dim, config.input_dim),
        "encoder_layers": [_transformer_layer_params(key, dim, config.mlp_dim) for key in enc_layer_keys],
        "decoder_layers": [_transformer_layer_params(key, dim, config.mlp_dim) for key in dec_layer_keys],
    }


def encode(params: Params, embeddings: jax.Array, mask: jax.Array, config: RLTTokenConfig | None = None) -> jax.Array:
    if config is None:
        config = RLTTokenConfig(input_dim=embeddings.shape[-1], token_dim=embeddings.shape[-1])
    mask = mask.astype(jnp.bool_)
    x = embeddings.astype(jnp.float32)
    rlt_token = jnp.broadcast_to(params["encoder_rlt_token"], (x.shape[0], 1, x.shape[-1]))
    x = jnp.concatenate([x, rlt_token], axis=1)
    valid = jnp.concatenate([mask, jnp.ones((mask.shape[0], 1), dtype=jnp.bool_)], axis=1)
    x = _transformer(params["encoder_layers"], x, _token_pool_mask(mask), _positions_from_mask(valid), config)
    return x[:, -1]


def decode(
    params: Params,
    rlt_token: jax.Array,
    embeddings: jax.Array,
    mask: jax.Array,
    config: RLTTokenConfig | None = None,
) -> jax.Array:
    if config is None:
        config = RLTTokenConfig(input_dim=embeddings.shape[-1], token_dim=embeddings.shape[-1])
    mask = mask.astype(jnp.bool_)
    embeddings = embeddings.astype(jnp.float32)
    rlt_token = rlt_token[:, None, :]
    # Autoregressive teacher forcing: predict embedding[t] from the RLT token and
    # previous VLA embeddings, not from a learned decode query or the current target.
    decoder_inputs = jnp.concatenate([rlt_token, embeddings[:, :-1]], axis=1)
    decoder_valid = jnp.concatenate([jnp.ones((mask.shape[0], 1), dtype=jnp.bool_), mask[:, :-1]], axis=1)
    decoder_mask = _causal_mask(mask)[:, :-1, :-1]
    decoded = _transformer(
        params["decoder_layers"], decoder_inputs, decoder_mask, _positions_from_mask(decoder_valid), config
    )
    output = _linear(params["output_proj"], decoded)
    return output * mask.astype(output.dtype)[..., None]


def reconstruction_loss(
    params: Params,
    embeddings: jax.Array,
    mask: jax.Array,
    config: RLTTokenConfig | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    if config is None:
        config = RLTTokenConfig(input_dim=embeddings.shape[-1], token_dim=embeddings.shape[-1])
    embeddings = jax.lax.stop_gradient(embeddings.astype(jnp.float32))
    mask_f = mask.astype(jnp.float32)
    rlt_token = encode(params, embeddings, mask, config)
    recon = decode(params, rlt_token, embeddings, mask, config)
    sq = jnp.mean(jnp.square(recon - embeddings), axis=-1)
    loss = jnp.sum(sq * mask_f) / jnp.maximum(jnp.sum(mask_f), 1.0)
    padding_f = 1.0 - mask_f
    padding_abs = jnp.sum(jnp.abs(recon) * padding_f[..., None]) / jnp.maximum(
        jnp.sum(padding_f) * recon.shape[-1], 1.0
    )
    return loss, {
        "token_norm": jnp.mean(jnp.linalg.norm(rlt_token, axis=-1)),
        "reconstruction_loss": loss,
        "valid_token_count": jnp.sum(mask_f),
        "padding_output_abs_mean": padding_abs,
    }
