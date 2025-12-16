"""Gemma3 wrapper for value function, adapted from pi0's gemma.py pattern.

This module provides a Gemma3 implementation with separate embed() and __call__() methods,
allowing pre-computed embeddings (e.g., from SigLIP) to be passed directly to the transformer.

We follow this einsum axis naming convention:
  B: batch
  T: query length
  S: k/v length
  N: num query heads
  K: num k/v heads
  G: num query heads per k/v head
  H: head dim
  D: d_model ("features")
"""

from collections.abc import Sequence
import dataclasses
from typing import Literal

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at


# Gemma3 vocab size (from gemma.gm.nn._gemma)
GEMMA3_VOCAB_SIZE = 262144


@dataclasses.dataclass
class Gemma3Config:
    """Configuration for Gemma3 model."""
    width: int  # embed_dim
    depth: int  # num_layers
    mlp_dim: int  # hidden_dim
    num_heads: int
    num_kv_heads: int
    head_dim: int
    use_post_attn_norm: bool = True
    use_post_ffw_norm: bool = True
    use_qk_norm: bool = True
    sliding_window_size: int = 512
    local_base_frequency: int = 10_000
    global_base_frequency: int = 1_000_000


Variant = Literal["gemma3_270m"]


def get_config(variant: Variant) -> Gemma3Config:
    """Returns config for specified Gemma3 variant."""
    if variant == "gemma3_270m":
        # Matches Gemma3_270M from gemma.gm.nn._gemma
        return Gemma3Config(
            width=640,
            depth=12,  # _NUM_LAYERS_GEMMA3_270M
            mlp_dim=2048,
            num_heads=4,
            num_kv_heads=1,
            head_dim=256,
            use_post_attn_norm=True,
            use_post_ffw_norm=True,
            use_qk_norm=True,
            sliding_window_size=512,
            local_base_frequency=10_000,
            global_base_frequency=1_000_000,
        )
    raise ValueError(f"Unknown variant: {variant}")


class RMSNorm(nn.Module):
    """RMS normalization layer."""

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed_inputs = x * jnp.reciprocal(jnp.sqrt(var + 1e-06))
        scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1],))
        normed_inputs = normed_inputs * (1 + scale)
        return normed_inputs.astype(dtype)


class Embedder(nn.Module):
    """Embedder module for text tokens."""

    vocab_size: int
    embed_dim: int

    def setup(self):
        self.input_embedding_table = self.param(
            "input_embedding",
            nn.initializers.normal(),
            (self.vocab_size, self.embed_dim),
        )

    def encode(self, x):
        """Encode tokens to embeddings."""
        x = self.input_embedding_table[(x,)]
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def decode(self, x):
        """Decode embeddings back to logits over vocabulary."""
        return jnp.dot(x, self.input_embedding_table.T)


class Attention(nn.Module):
    """Multi-head attention with RoPE and optional QK normalization."""

    config: Gemma3Config
    layer_idx: int

    @nn.compact
    def __call__(self, x, positions, attn_mask, kv_cache=None):
        config = self.config
        dtype = x.dtype

        # Determine if this is local or global attention based on layer pattern
        # Gemma3 pattern: (GLOBAL, LOCAL_SLIDING) * 5 + (GLOBAL, GLOBAL) for 12 layers
        # Simplified: even layers are global, odd layers are local sliding (except last two)
        is_local = (self.layer_idx % 2 == 1) and (self.layer_idx < config.depth - 2)

        rope_base_freq = config.local_base_frequency if is_local else config.global_base_frequency

        # Q, K, V projections
        if config.num_kv_heads == config.num_heads:
            qkv_einsum = nn.Dense(
                3 * config.num_heads * config.head_dim,
                use_bias=False,
                name="qkv_proj",
            )
            qkv = qkv_einsum(x)
            qkv = qkv.reshape(x.shape[0], x.shape[1], 3, config.num_heads, config.head_dim)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        else:
            q_proj = nn.Dense(config.num_heads * config.head_dim, use_bias=False, name="q_proj")
            kv_proj = nn.Dense(2 * config.num_kv_heads * config.head_dim, use_bias=False, name="kv_proj")
            q = q_proj(x).reshape(x.shape[0], x.shape[1], config.num_heads, config.head_dim)
            kv = kv_proj(x).reshape(x.shape[0], x.shape[1], 2, config.num_kv_heads, config.head_dim)
            k, v = kv[:, :, 0], kv[:, :, 1]

        # QK normalization (Gemma3 specific)
        if config.use_qk_norm:
            q = RMSNorm(name="q_norm")(q)
            k = RMSNorm(name="k_norm")(k)

        # Apply RoPE
        q = _apply_rope(q, positions, base_frequency=rope_base_freq)
        k = _apply_rope(k, positions, base_frequency=rope_base_freq)

        # Scale query
        q = q * (config.head_dim ** -0.5)

        # Handle GQA (grouped query attention)
        if config.num_kv_heads != config.num_heads:
            # Reshape for grouped query attention
            groups = config.num_heads // config.num_kv_heads
            q = q.reshape(x.shape[0], x.shape[1], config.num_kv_heads, groups, config.head_dim)
            logits = jnp.einsum("BTKGH,BSKH->BTKGS", q, k)
            logits = logits.reshape(x.shape[0], x.shape[1], config.num_heads, -1)
        else:
            logits = jnp.einsum("BTNH,BSNH->BTNS", q, k)

        # Apply sliding window mask for local attention
        if is_local and config.sliding_window_size is not None:
            sliding_mask = _create_sliding_mask(positions, config.sliding_window_size)
            attn_mask = attn_mask & sliding_mask

        # Apply attention mask
        # logits shape: (batch, seq, heads, seq) -> need mask shape (batch, 1, 1, seq) or (batch, seq, 1, seq)
        big_neg = -2.3819763e38
        # Expand mask: (batch, seq, seq) -> (batch, seq, 1, seq) to broadcast with (batch, seq, heads, seq)
        masked_logits = jnp.where(attn_mask[:, :, None, :], logits, big_neg)

        # Softmax and weighted sum
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        if config.num_kv_heads != config.num_heads:
            groups = config.num_heads // config.num_kv_heads
            probs = probs.reshape(x.shape[0], x.shape[1], config.num_kv_heads, groups, -1)
            encoded = jnp.einsum("BTKGS,BSKH->BTKGH", probs, v)
            encoded = encoded.reshape(x.shape[0], x.shape[1], config.num_heads, config.head_dim)
        else:
            encoded = jnp.einsum("BTNS,BSNH->BTNH", probs, v)

        # Output projection
        out_proj = nn.Dense(config.width, use_bias=False, name="out_proj")
        output = out_proj(encoded.reshape(x.shape[0], x.shape[1], -1))

        return output


class FeedForward(nn.Module):
    """Feed-forward network with gated activation (GeGLU)."""

    config: Gemma3Config

    @nn.compact
    def __call__(self, x):
        config = self.config
        dtype = x.dtype

        # Gated linear unit with GELU activation
        gate_proj = nn.Dense(config.mlp_dim, use_bias=False, name="gate_proj")
        up_proj = nn.Dense(config.mlp_dim, use_bias=False, name="up_proj")
        down_proj = nn.Dense(config.width, use_bias=False, name="down_proj")

        gate = nn.gelu(gate_proj(x))
        up = up_proj(x)
        hidden = gate * up
        output = down_proj(hidden)

        return output


class Block(nn.Module):
    """Transformer block with pre-norm architecture."""

    config: Gemma3Config
    layer_idx: int

    @nn.compact
    def __call__(self, x, positions, attn_mask):
        config = self.config

        # Pre-attention norm
        normed = RMSNorm(name="pre_attention_norm")(x)

        # Self-attention
        attn_output = Attention(config=config, layer_idx=self.layer_idx, name="attn")(
            normed, positions, attn_mask
        )

        # Post-attention norm (Gemma3 specific)
        if config.use_post_attn_norm:
            attn_output = RMSNorm(name="post_attention_norm")(attn_output)

        # Residual connection
        x = x + attn_output

        # Pre-FFN norm
        normed = RMSNorm(name="pre_ffw_norm")(x)

        # Feed-forward
        ffn_output = FeedForward(config=config, name="mlp")(normed)

        # Post-FFN norm (Gemma3 specific)
        if config.use_post_ffw_norm:
            ffn_output = RMSNorm(name="post_ffw_norm")(ffn_output)

        # Residual connection
        x = x + ffn_output

        return x


class Gemma3Module(nn.Module):
    """Gemma3 transformer with separate embed() and __call__() methods.

    This design allows:
    1. embed(tokens) - Convert token IDs to embeddings
    2. __call__(embeddings, ...) - Process pre-computed embeddings through transformer

    This separation enables mixing embeddings from different sources (e.g., SigLIP image
    embeddings + text embeddings) before passing through the transformer.
    """

    config: Gemma3Config
    embed_dtype: str = "bfloat16"

    def setup(self):
        self.embedder = Embedder(
            vocab_size=GEMMA3_VOCAB_SIZE,
            embed_dim=self.config.width,
            name="embedder",
        )
        self.layers = [
            Block(config=self.config, layer_idx=i, name=f"layer_{i}")
            for i in range(self.config.depth)
        ]
        self.final_norm = RMSNorm(name="final_norm")

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        """Embed tokens to continuous representations.

        Args:
            tokens: Integer token IDs of shape [batch, seq_len]

        Returns:
            Embeddings of shape [batch, seq_len, embed_dim]
        """
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    @at.typecheck
    def __call__(
        self,
        embedded: at.Float[at.Array, "b t d"],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        *,
        deterministic: bool = True,
    ) -> at.Float[at.Array, "b t d"]:
        """Forward pass through transformer blocks.

        Args:
            embedded: Pre-computed embeddings of shape [batch, seq_len, embed_dim]
            positions: Position indices of shape [batch, seq_len]
            mask: Attention mask of shape [batch, seq_len, seq_len]
            deterministic: Whether to disable dropout (unused, for API compatibility)

        Returns:
            Output embeddings of shape [batch, seq_len, embed_dim]
        """
        x = embedded.astype(self.embed_dtype)

        for layer in self.layers:
            x = layer(x, positions, mask)

        x = self.final_norm(x)
        return x

    def initialize_params(self):
        """Initialize all parameters by running dummy forward passes."""
        # Initialize embedder
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        # Initialize transformer
        self(
            jnp.zeros((1, 1, self.config.width), dtype=jnp.float32),
            jnp.zeros((1, 1), dtype=jnp.int32),
            jnp.ones((1, 1, 1), dtype=bool),
        )


def _apply_rope(x, positions, base_frequency=10_000):
    """Apply Rotary Position Embedding (RoPE).

    Args:
        x: Input tensor of shape [batch, seq_len, num_heads, head_dim]
        positions: Position indices of shape [batch, seq_len]
        base_frequency: Base frequency for RoPE

    Returns:
        Tensor with RoPE applied, same shape as input
    """
    head_dim = x.shape[-1]
    freq_exponents = (2.0 / head_dim) * jnp.arange(head_dim // 2, dtype=jnp.float32)
    timescale = base_frequency ** freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]  # [B, T, 1, head_dim//2]

    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    return res.astype(x.dtype)


def _create_sliding_mask(positions, window_size):
    """Create sliding window attention mask.

    Args:
        positions: Position indices of shape [batch, seq_len]
        window_size: Size of the sliding window

    Returns:
        Boolean mask of shape [batch, seq_len, seq_len]
    """
    # positions[:, :, None] - positions[:, None, :] gives relative positions
    pos_diff = positions[:, :, None] - positions[:, None, :]
    mask = jnp.abs(pos_diff) < window_size
    return mask
