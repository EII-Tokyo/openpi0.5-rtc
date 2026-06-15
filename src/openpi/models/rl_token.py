import dataclasses
from typing import Literal

from flax import nnx
import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class RLTokenConfig:
    hidden_dim: int = 2048
    token_hidden_dim: int | None = None
    z_dim: int | None = None
    encoder_layers: int = 4
    decoder_layers: int = 4
    num_heads: int = 8
    mlp_dim: int = 8192
    max_prefix_len: int = 1224
    decoder_mode: Literal["teacher_forced", "query"] = "teacher_forced"
    history_mask_ratio: float = 0.0
    zero_margin_weight: float = 0.0
    zero_margin: float = 0.5
    shuffled_margin_weight: float = 0.0
    shuffled_margin: float = 0.1

    def __post_init__(self):
        if self.token_hidden_dim is None:
            object.__setattr__(self, "token_hidden_dim", self.hidden_dim)
        if self.z_dim is None:
            object.__setattr__(self, "z_dim", self.hidden_dim)
        if self.decoder_layers != self.encoder_layers:
            raise ValueError(
                "RLT RL Token autoencoder uses matched encoder/decoder depth; "
                f"got encoder_layers={self.encoder_layers}, decoder_layers={self.decoder_layers}."
            )
        if self.token_hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"token_hidden_dim={self.token_hidden_dim} must be divisible by num_heads={self.num_heads}"
            )
        if not 0 <= self.history_mask_ratio < 1:
            raise ValueError(f"history_mask_ratio must be in [0, 1), got {self.history_mask_ratio}.")


class RLTokenBlock(nnx.Module):
    def __init__(self, config: RLTokenConfig, *, rngs: nnx.Rngs):
        self.config = config
        token_hidden_dim = config.token_hidden_dim
        self.pre_attn_norm = nnx.LayerNorm(token_hidden_dim, rngs=rngs)
        self.q_proj = nnx.Linear(token_hidden_dim, token_hidden_dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(token_hidden_dim, token_hidden_dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(token_hidden_dim, token_hidden_dim, use_bias=False, rngs=rngs)
        self.out_proj = nnx.Linear(token_hidden_dim, token_hidden_dim, use_bias=False, rngs=rngs)
        self.pre_mlp_norm = nnx.LayerNorm(token_hidden_dim, rngs=rngs)
        self.mlp_in = nnx.Linear(token_hidden_dim, config.mlp_dim, rngs=rngs)
        self.mlp_out = nnx.Linear(config.mlp_dim, token_hidden_dim, rngs=rngs)

    def __call__(
        self,
        x: at.Float[at.Array, "b s d"],
        query_mask: at.Bool[at.Array, "b s"],
        key_mask: at.Bool[at.Array, "b s"] | None = None,
        *,
        causal: bool = False,
    ) -> at.Float[at.Array, "b s d"]:
        if key_mask is None:
            key_mask = query_mask
        dtype = x.dtype
        residual = x
        x_norm = self.pre_attn_norm(x)

        batch_size, seq_len, _ = x_norm.shape
        head_dim = self.config.token_hidden_dim // self.config.num_heads

        def split_heads(y):
            y = y.reshape(batch_size, seq_len, self.config.num_heads, head_dim)
            return jnp.swapaxes(y, 1, 2)

        q = split_heads(self.q_proj(x_norm))
        k = split_heads(self.k_proj(x_norm))
        v = split_heads(self.v_proj(x_norm))
        logits = jnp.einsum("bhqd,bhkd->bhqk", q, k, preferred_element_type=jnp.float32)
        logits = logits * (head_dim**-0.5)

        key_attn_mask = key_mask[:, None, None, :]
        query_attn_mask = query_mask[:, None, :, None]
        attn_mask = key_attn_mask & query_attn_mask
        if causal:
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))[None, None, :, :]
            attn_mask = attn_mask & causal_mask
        big_neg = jnp.finfo(jnp.float32).min
        logits = jnp.where(attn_mask, logits, big_neg)
        probs = jax.nn.softmax(logits, axis=-1).astype(dtype)
        attended = jnp.einsum("bhqk,bhkd->bhqd", probs, v)
        attended = jnp.swapaxes(attended, 1, 2).reshape(batch_size, seq_len, self.config.token_hidden_dim)
        x = residual + self.out_proj(attended)

        residual = x
        x_norm = self.pre_mlp_norm(x)
        x = residual + self.mlp_out(jax.nn.gelu(self.mlp_in(x_norm)))
        return jnp.where(query_mask[..., None], x, 0)


class RLTokenAutoencoder(nnx.Module):
    def __init__(self, config: RLTokenConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.input_proj = (
            nnx.Linear(config.hidden_dim, config.token_hidden_dim, rngs=rngs)
            if config.token_hidden_dim != config.hidden_dim
            else None
        )
        self.z_proj = (
            nnx.Linear(config.token_hidden_dim, config.z_dim, rngs=rngs)
            if config.z_dim != config.token_hidden_dim
            else None
        )
        self.rl_query = nnx.Param(
            jax.random.normal(rngs.params(), (1, 1, config.token_hidden_dim), dtype=jnp.float32) * 0.02
        )
        self.encoder_pos_embedding = nnx.Param(
            jax.random.normal(
                rngs.params(),
                (1, config.max_prefix_len + 1, config.token_hidden_dim),
                dtype=jnp.float32,
            )
            * 0.02
        )
        self.decoder_pos_embedding = nnx.Param(
            jax.random.normal(
                rngs.params(),
                (1, config.max_prefix_len, config.token_hidden_dim),
                dtype=jnp.float32,
            )
            * 0.02
        )
        self.decoder_query_embedding = nnx.Param(
            jax.random.normal(
                rngs.params(),
                (1, config.max_prefix_len, config.token_hidden_dim),
                dtype=jnp.float32,
            )
            * 0.02
        )
        self.decoder_mask_embedding = nnx.Param(
            jax.random.normal(rngs.params(), (1, 1, config.token_hidden_dim), dtype=jnp.float32) * 0.02
        )
        self.encoder_blocks = [RLTokenBlock(config, rngs=rngs) for _ in range(config.encoder_layers)]
        self.decoder_blocks = [RLTokenBlock(config, rngs=rngs) for _ in range(config.decoder_layers)]
        self.output_norm = nnx.LayerNorm(config.token_hidden_dim, rngs=rngs)
        self.output_proj = nnx.Linear(config.token_hidden_dim, config.hidden_dim, rngs=rngs)
        self.z_to_decoder = nnx.Linear(config.z_dim, config.token_hidden_dim, rngs=rngs)

    def __call__(
        self,
        h_vla: at.Float[at.Array, "b n d"],
        prefix_mask: at.Bool[at.Array, "b n"],
        *,
        rng: at.KeyArrayLike | None = None,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "b d"], at.Float[at.Array, "b n d"]]:
        z_rl = self.encode(h_vla, prefix_mask)
        h_hat = self.decode(z_rl, h_vla, prefix_mask, rng=rng, train=train)
        return z_rl, h_hat

    def encode(
        self,
        h_vla: at.Float[at.Array, "b n d"],
        prefix_mask: at.Bool[at.Array, "b n"],
    ) -> at.Float[at.Array, "b d"]:
        if h_vla.shape[-1] != self.config.hidden_dim:
            raise ValueError(f"Expected h_vla dim {self.config.hidden_dim}, got {h_vla.shape[-1]}")
        seq_len = h_vla.shape[1]
        if seq_len > self.config.max_prefix_len:
            raise ValueError(f"prefix length {seq_len} exceeds max_prefix_len={self.config.max_prefix_len}")

        batch_size = h_vla.shape[0]
        h_token = self.input_proj(h_vla) if self.input_proj is not None else h_vla
        rl_query = jnp.broadcast_to(
            self.rl_query.value.astype(h_token.dtype),
            (batch_size, 1, self.config.token_hidden_dim),
        )
        encoder_input = jnp.concatenate([h_token, rl_query], axis=1)
        encoder_input = encoder_input + self.encoder_pos_embedding.value[:, : seq_len + 1].astype(h_vla.dtype)
        encoder_mask = jnp.concatenate(
            [prefix_mask, jnp.ones((batch_size, 1), dtype=prefix_mask.dtype)],
            axis=1,
        )

        x = encoder_input
        for block in self.encoder_blocks:
            x = block(x, encoder_mask)
        z_rl = x[:, -1, :]
        return self.z_proj(z_rl) if self.z_proj is not None else z_rl

    def decode(
        self,
        z_rl: at.Float[at.Array, "b d"],
        h_vla: at.Float[at.Array, "b n d"],
        prefix_mask: at.Bool[at.Array, "b n"],
        *,
        rng: at.KeyArrayLike | None = None,
        train: bool = False,
    ) -> at.Float[at.Array, "b n d"]:
        if z_rl.shape[-1] != self.config.z_dim:
            raise ValueError(f"Expected z_rl dim {self.config.z_dim}, got {z_rl.shape[-1]}")
        if h_vla.shape[-1] != self.config.hidden_dim:
            raise ValueError(f"Expected h_vla dim {self.config.hidden_dim}, got {h_vla.shape[-1]}")
        seq_len = h_vla.shape[1]
        if seq_len > self.config.max_prefix_len:
            raise ValueError(f"prefix length {seq_len} exceeds max_prefix_len={self.config.max_prefix_len}")

        batch_size = h_vla.shape[0]
        z_condition = self.z_to_decoder(z_rl)[:, None, :].astype(h_vla.dtype)
        if self.config.decoder_mode == "teacher_forced":
            target = jnp.where(prefix_mask[..., None], jax.lax.stop_gradient(h_vla), 0)
            history = self.input_proj(target) if self.input_proj is not None else target
            history = history[:, :-1, :]
            history_mask = prefix_mask[:, :-1]
            if train and self.config.history_mask_ratio > 0:
                if rng is None:
                    raise ValueError("rng is required when history_mask_ratio > 0 during training.")
                keep = jax.random.bernoulli(
                    rng,
                    p=1.0 - self.config.history_mask_ratio,
                    shape=history_mask.shape,
                )
                keep = keep & history_mask
                mask_embedding = self.decoder_mask_embedding.value.astype(h_vla.dtype)
                history = jnp.where(keep[..., None], history, mask_embedding)
            decoder_input = jnp.concatenate([z_condition, history], axis=1)
            decoder_key_mask = jnp.concatenate(
                [jnp.ones((batch_size, 1), dtype=prefix_mask.dtype), history_mask],
                axis=1,
            )
            causal = True
        elif self.config.decoder_mode == "query":
            queries = jnp.broadcast_to(
                self.decoder_query_embedding.value[:, :seq_len].astype(h_vla.dtype),
                (batch_size, seq_len, self.config.token_hidden_dim),
            )
            decoder_input = queries + z_condition
            decoder_key_mask = prefix_mask
            causal = False
        else:
            raise ValueError(f"Unknown decoder_mode={self.config.decoder_mode}.")
        decoder_input = decoder_input + self.decoder_pos_embedding.value[:, :seq_len].astype(h_vla.dtype)

        x = decoder_input
        for block in self.decoder_blocks:
            x = block(x, prefix_mask, decoder_key_mask, causal=causal)
        return self.output_proj(self.output_norm(x))

    def reconstruct_with_z(
        self,
        h_vla: at.Float[at.Array, "b n d"],
        prefix_mask: at.Bool[at.Array, "b n"],
        z_rl: at.Float[at.Array, "b d"],
    ) -> at.Float[at.Array, "b n d"]:
        return self.decode(z_rl, h_vla, prefix_mask)

    def compute_loss(
        self,
        h_vla: at.Float[at.Array, "b n d"],
        prefix_mask: at.Bool[at.Array, "b n"],
        *,
        rng: at.KeyArrayLike | None = None,
        train: bool = False,
    ) -> at.Float[at.Array, " b"]:
        z_rl, h_hat = self(h_vla, prefix_mask, rng=rng, train=train)
        real_loss = self.reconstruction_loss(h_hat, h_vla, prefix_mask)
        if self.config.zero_margin_weight == 0 and self.config.shuffled_margin_weight == 0:
            return real_loss

        zero_loss = self.compute_loss_with_z(h_vla, prefix_mask, jnp.zeros_like(z_rl))
        shuffled_loss = self.compute_loss_with_z(h_vla, prefix_mask, _batch_shuffle(z_rl))
        loss = real_loss
        if self.config.zero_margin_weight > 0:
            loss = loss + self.config.zero_margin_weight * jnp.maximum(
                0, self.config.zero_margin + real_loss - zero_loss
            )
        if self.config.shuffled_margin_weight > 0:
            loss = loss + self.config.shuffled_margin_weight * jnp.maximum(
                0, self.config.shuffled_margin + real_loss - shuffled_loss
            )
        return loss

    def compute_loss_with_z(
        self,
        h_vla: at.Float[at.Array, "b n d"],
        prefix_mask: at.Bool[at.Array, "b n"],
        z_rl: at.Float[at.Array, "b d"],
    ) -> at.Float[at.Array, " b"]:
        h_hat = self.reconstruct_with_z(h_vla, prefix_mask, z_rl)
        return self.reconstruction_loss(h_hat, h_vla, prefix_mask)

    def reconstruction_loss(
        self,
        h_hat: at.Float[at.Array, "b n d"],
        h_vla: at.Float[at.Array, "b n d"],
        prefix_mask: at.Bool[at.Array, "b n"],
    ) -> at.Float[at.Array, " b"]:
        target = jax.lax.stop_gradient(h_vla)
        token_loss = jnp.mean(jnp.square(h_hat - target), axis=-1)
        token_loss = jnp.where(prefix_mask, token_loss, 0)
        denom = jnp.maximum(jnp.sum(prefix_mask, axis=-1), 1)
        return jnp.sum(token_loss, axis=-1) / denom


def _batch_shuffle(z_rl: at.Float[at.Array, "b d"]) -> at.Float[at.Array, "b d"]:
    if z_rl.shape[0] <= 1:
        return jnp.zeros_like(z_rl)
    return jnp.roll(z_rl, shift=1, axis=0)
