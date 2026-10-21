# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A refactored and simplified ViT adoptation for Pi, taken from big_vision."""

from collections.abc import Sequence
import math

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import openpi.training.sharding as sharding


def spatial_average_pool_tokens(tokens, grid_shape, target_tokens):
    """Pool the spatial grid, never the temporal or channel axes, before projection.

    Like Gemma3, divisible grids use a non-overlapping mean with kernel/stride
    derived from the image patch grid. Other grids use adaptive average bins.
    """
    height, width = grid_shape
    side = math.isqrt(target_tokens)
    if side * side != target_tokens or side < 1:
        raise ValueError("image_tokens_per_image must be a positive perfect square")
    if min(height, width) < side or tokens.shape[-2] != height * width:
        raise ValueError(f"Cannot average-pool grid {grid_shape} to {side}x{side}")
    grid = tokens.reshape(*tokens.shape[:-2], height, width, tokens.shape[-1])
    if height % side == 0 and width % side == 0:
        grid = grid.reshape(*tokens.shape[:-2], side, height // side, side, width // side, tokens.shape[-1])
        pooled = grid.mean(axis=(-4, -2))
    else:
        def weights(length):
            starts = np.floor(np.arange(side) * length / side).astype(int)
            ends = np.ceil((np.arange(side) + 1) * length / side).astype(int)
            positions = np.arange(length)[None, :]
            mask = (positions >= starts[:, None]) & (positions < ends[:, None])
            return jnp.asarray(mask / (ends - starts)[:, None], dtype=tokens.dtype)
        pooled = jnp.einsum('...hwc,ih,jw->...ijc', grid, weights(height), weights(width))
    return pooled.reshape(*tokens.shape[:-2], target_tokens, tokens.shape[-1])


def posemb_sincos_2d(h, w, width, temperature=10_000.0, dtype=jnp.float32):
    """Follows the MoCo v3 logic."""
    y, x = jnp.mgrid[:h, :w]

    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]


def posemb_sincos_1d(length, width, temperature=10_000.0, dtype=jnp.float32):
    positions = jnp.arange(length)
    assert width % 2 == 0, "Width must be even for 1D sincos posemb"
    omega = jnp.arange(width // 2) / jnp.maximum(width // 2 - 1, 1)
    omega = 1.0 / (temperature**omega)
    pos = jnp.einsum("m,d->md", positions, omega)
    pe = jnp.concatenate([jnp.sin(pos), jnp.cos(pos)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, None, :]


def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
    if typ == "learn":
        return self.param(
            name,
            nn.initializers.normal(stddev=1 / np.sqrt(width)),
            (1, np.prod(seqshape), width),
            dtype,
        )
    if typ == "sincos2d":
        return posemb_sincos_2d(*seqshape, width, dtype=dtype)
    raise ValueError(f"Unknown posemb type: {typ}")


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int | None = None
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        inits = {
            "kernel_init": nn.initializers.xavier_uniform(),
            "bias_init": nn.initializers.normal(stddev=1e-6),
        }

        d = x.shape[-1]
        x = nn.Dense(self.mlp_dim or 4 * d, dtype=self.dtype_mm, **inits)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        return nn.Dense(d, dtype=self.dtype_mm, **inits)(x)


class Encoder1DBlock(nn.Module):
    """Single transformer encoder block (MHSA + MLP)."""

    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"

    def setup(self):
        # Keep parameter paths compatible with existing image/video checkpoints.
        self.ln0 = nn.LayerNorm(name="LayerNorm_0", dtype=self.dtype_mm)
        self.attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            dtype=self.dtype_mm,
            name="MultiHeadDotProductAttention_0",
        )
        self.ln1 = nn.LayerNorm(name="LayerNorm_1", dtype=self.dtype_mm)
        self.mlp = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
            name="MlpBlock_0",
        )
        self.attention_dropout = nn.Dropout(rate=self.dropout)
        self.mlp_dropout = nn.Dropout(rate=self.dropout)

    def _apply_attention(self, seq, mask, deterministic):
        seq = sharding.activation_sharding_constraint(seq)
        y = self.ln0(seq)
        y = self.attn(y, y, mask=mask, deterministic=deterministic)
        y = sharding.activation_sharding_constraint(y)
        y = self.attention_dropout(y, deterministic=deterministic)
        return seq + y, y

    def _apply_temporal_attention(self, x, mask, deterministic):
        batch_size, time_size, seq_len, width = x.shape
        temporal_x = jnp.swapaxes(x, 1, 2).reshape(batch_size * seq_len, time_size, width)
        temporal_x, temporal_y = self._apply_attention(temporal_x, mask, deterministic)
        temporal_x = temporal_x.reshape(batch_size, seq_len, time_size, width).swapaxes(1, 2)
        temporal_y = temporal_y.reshape(batch_size, seq_len, time_size, width).swapaxes(1, 2)
        return temporal_x, temporal_y

    def __call__(
        self,
        x,
        deterministic=True,
        temporal_first=False,
        attn_mask=None,
        temporal_attn_mask=None,
    ):  # noqa: FBT002
        out = {}

        if x.ndim == 4:
            batch_size, time_size, seq_len, width = x.shape
            if time_size > 1:

                def temporal(block, value):
                    temporal_x, temporal_y = block._apply_temporal_attention(value, temporal_attn_mask, deterministic)
                    # Preserve the baseline's BF16 residual operation order.
                    return value + (temporal_x - value), temporal_y

                def skip_temporal(block, value):
                    return value, jnp.zeros_like(value)

                if self.is_initializing():
                    # Both cond branches must see the same parameter tree. Create
                    # shared attention parameters before entering lifted control flow.
                    temporal_x, temporal_y = self._apply_temporal_attention(x, temporal_attn_mask, deterministic)
                    temporal_gate = jnp.asarray(temporal_first, dtype=x.dtype)
                    x = x + temporal_gate * (temporal_x - x)
                    out["temporal_sa"] = temporal_gate * temporal_y
                else:
                    x, out["temporal_sa"] = nn.cond(temporal_first, temporal, skip_temporal, self, x)
            else:
                out["temporal_sa"] = jnp.zeros_like(x)

            spatial_x = jnp.reshape(x, [batch_size * time_size, seq_len, width])
            spatial_x, spatial_y = self._apply_attention(spatial_x, attn_mask, deterministic)
            x = jnp.reshape(spatial_x, [batch_size, time_size, seq_len, width])
            out["sa"] = jnp.reshape(spatial_y, [batch_size, time_size, seq_len, width])
        else:
            x, sa_y = self._apply_attention(x, attn_mask, deterministic)
            out["temporal_sa"] = jnp.zeros_like(x)
            out["sa"] = sa_y

        y = self.ln1(x)
        y = out["mlp"] = self.mlp(y, deterministic)
        y = sharding.activation_sharding_constraint(y)
        y = self.mlp_dropout(y, deterministic=deterministic)
        x = out["+mlp"] = x + y
        x = sharding.activation_sharding_constraint(x)
        return x, out


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    depth: int
    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"
    temporal_every_n_layers: int = 4

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}

        if self.scan:
            block = nn.remat(
                Encoder1DBlock,
                prevent_cse=False,
                static_argnums=(2,),
                policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
            )
            temporal_flags = jnp.asarray(
                [(lyr + 1) % self.temporal_every_n_layers == 0 for lyr in range(self.depth)],
                dtype=jnp.bool_,
            )
            temporal_attn_mask = None
            if x.ndim == 4 and x.shape[1] > 1:
                time_size = x.shape[1]
                temporal_attn_mask = jnp.tril(jnp.ones((time_size, time_size), dtype=jnp.bool_))[None, None, :, :]
            x, scan_out = nn.scan(
                block,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=(nn.broadcast, 0, nn.broadcast, nn.broadcast),
                length=self.depth,
            )(
                name="encoderblock",
                dtype_mm=self.dtype_mm,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )(x, deterministic, temporal_flags, None, temporal_attn_mask)
            for lyr in range(self.depth):
                out[f"block{lyr:02d}"] = jax.tree.map(lambda o, lyr=lyr: o[lyr], scan_out)
        else:
            for lyr in range(self.depth):
                block_cur = Encoder1DBlock(
                    name=f"encoderblock_{lyr}",
                    dtype_mm=self.dtype_mm,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
                if x.ndim == 4:
                    time_size = x.shape[1]
                    if time_size > 1 and (lyr + 1) % self.temporal_every_n_layers == 0:
                        causal_mask = jnp.tril(jnp.ones((time_size, time_size), dtype=jnp.bool_))[None, None, :, :]
                        x, out[f"block{lyr:02d}"] = block_cur(
                            x,
                            deterministic,
                            temporal_first=True,
                            temporal_attn_mask=causal_mask,
                        )
                    else:
                        x, out[f"block{lyr:02d}"] = block_cur(x, deterministic)
                else:
                    x, out[f"block{lyr:02d}"] = block_cur(x, deterministic)
            out["pre_ln"] = x

        return nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x), out


class MAPHead(nn.Module):
    """Multihead Attention Pooling."""

    mlp_dim: int | None = None
    num_heads: int = 12
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x):
        n, _, d = x.shape
        probe = self.param("probe", nn.initializers.xavier_uniform(), (1, 1, d), x.dtype)
        probe = jnp.tile(probe, [n, 1, 1])

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype_mm,
            kernel_init=nn.initializers.xavier_uniform(),
        )(probe, x)

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        x = x + MlpBlock(mlp_dim=self.mlp_dim, dtype=self.dtype_mm)(y)
        return x[:, 0]


class _Module(nn.Module):
    """ViT model."""

    num_classes: int | None = None
    patch_size: Sequence[int] = (16, 16)
    width: int = 768
    depth: int = 12
    mlp_dim: int | None = None
    num_heads: int = 12
    posemb: str = "learn"
    rep_size: int | bool = False
    dropout: float = 0.0
    pool_type: str = "gap"
    head_zeroinit: bool = True
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"
    temporal_every_n_layers: int = 4
    image_tokens_per_image: int | None = None

    @nn.compact
    def __call__(self, image, *, train=False):
        out = {}
        image = jnp.asarray(image, jnp.float32)

        is_video = image.ndim == 5
        if is_video:
            batch_size, time_size, image_h, image_w, image_c = image.shape
            image = jnp.reshape(image, [batch_size * time_size, image_h, image_w, image_c])
        else:
            batch_size = image.shape[0]
            time_size = 1

        x = out["stem"] = nn.Conv(
            self.width,
            self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            name="embedding",
            dtype=jnp.float32,
        )(image)

        _, h, w, c = x.shape
        x = jnp.reshape(x, [batch_size, time_size, h * w, c])

        spatial_posemb = get_posemb(self, self.posemb, (h, w), c, "pos_embedding", jnp.float32)
        x = x + spatial_posemb[:, None, :, :]
        if time_size > 1:
            temporal_posemb = posemb_sincos_1d(time_size, c, dtype=jnp.float32)
            temporal_posemb = temporal_posemb - temporal_posemb[:, -1:, :, :]
            x = x + temporal_posemb
        x = out["with_posemb"] = x

        if self.pool_type == "tok":
            cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
            cls = jnp.tile(cls[:, None, :, :], [batch_size, time_size, 1, 1])
            x = jnp.concatenate([cls, x], axis=2)

        x = nn.Dropout(rate=self.dropout)(x, not train)
        x = x.astype(self.dtype_mm)

        x, out["encoder"] = Encoder(
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            scan=self.scan,
            remat_policy=self.remat_policy,
            dtype_mm=self.dtype_mm,
            temporal_every_n_layers=self.temporal_every_n_layers,
            name="Transformer",
        )(x, deterministic=not train)
        encoded = out["encoded"] = x[:, -1, :, :]
        if self.image_tokens_per_image is not None:
            if self.pool_type != "none":
                raise ValueError("Spatial token pooling requires pool_type='none'")
            encoded = spatial_average_pool_tokens(encoded, (h, w), self.image_tokens_per_image)
            h = w = math.isqrt(self.image_tokens_per_image)
            out["spatially_pooled"] = encoded

        if self.pool_type == "map":
            x = out["head_input"] = MAPHead(
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dtype_mm=self.dtype_mm,
            )(encoded)
        elif self.pool_type == "gap":
            x = out["head_input"] = jnp.mean(encoded, axis=1)
        elif self.pool_type == "0":
            x = out["head_input"] = encoded[:, 0]
        elif self.pool_type == "tok":
            x = out["head_input"] = encoded[:, 0]
            encoded = encoded[:, 1:]
        elif self.pool_type == "none":
            x = encoded
        else:
            raise ValueError(f"Unknown pool type: '{self.pool_type}'")

        x_2d = jnp.reshape(encoded, [batch_size, h, w, -1])

        if self.rep_size:
            rep_size = self.width if self.rep_size is True else self.rep_size
            hid = nn.Dense(rep_size, dtype=self.dtype_mm, name="pre_logits")
            x_2d = nn.tanh(hid(x_2d))
            x = nn.tanh(hid(x))

        out["pre_logits_2d"] = x_2d
        out["pre_logits"] = x

        if self.num_classes:
            kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
            head = nn.Dense(self.num_classes, dtype=self.dtype_mm, name="head", **kw)
            x_2d = out["logits_2d"] = head(x_2d)
            x = out["logits"] = head(x)

        return x, out


def Module(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name  # noqa: N802
    """Factory function, because linen really don't like what I'm doing!"""
    return _Module(num_classes, **{**decode_variant(variant), **kw})


def decode_variant(variant):
    """Converts a string like "B" or "B/32" into a params dict."""
    if variant is None:
        return {}

    v, patch = variant, {}
    if "/" in variant:
        v, patch = variant.split("/")
        patch = {"patch_size": (int(patch), int(patch))}

    return {
        "width": {
            "mu": 32,
            "Ti": 192,
            "S": 384,
            "M": 512,
            "B": 768,
            "L": 1024,
            "So400m": 1152,
            "H": 1280,
            "g": 1408,
            "g-opt": 1536,
            "G": 1664,
            "G-opt": 1536,
            "e": 1792,
        }[v],
        "depth": {
            "mu": 1,
            "Ti": 12,
            "S": 12,
            "M": 12,
            "B": 12,
            "L": 24,
            "So400m": 27,
            "H": 32,
            "g": 40,
            "g-opt": 40,
            "G": 48,
            "G-opt": 48,
            "e": 56,
        }[v],
        "mlp_dim": {
            "mu": 128,
            "Ti": 768,
            "S": 1536,
            "M": 2048,
            "B": 3072,
            "L": 4096,
            "So400m": 4304,
            "H": 5120,
            "g": 6144,
            "g-opt": 6144,
            "G": 8192,
            "G-opt": 8192,
            "e": 15360,
        }[v],
        "num_heads": {
            "mu": 2,
            "Ti": 3,
            "S": 6,
            "M": 8,
            "B": 12,
            "L": 16,
            "So400m": 16,
            "H": 16,
            "g": 16,
            "g-opt": 16,
            "G": 16,
            "G-opt": 16,
            "e": 16,
        }[v],
        **patch,
    }
