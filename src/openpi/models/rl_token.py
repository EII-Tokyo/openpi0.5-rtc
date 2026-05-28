import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class RLTokenConfig:
    hidden_dim: int = 2048
    encoder_layers: int = 4
    decoder_layers: int = 4
    num_heads: int = 8
    mlp_dim: int = 8192
    max_prefix_len: int = 1224


class RLTokenBlock(nnx.Module):
    def __init__(self, config: RLTokenConfig, *, rngs: nnx.Rngs):
        if config.hidden_dim % config.num_heads != 0:
            raise ValueError(
                f"hidden_dim={config.hidden_dim} must be divisible by num_heads={config.num_heads}"
            )
        self.config = config
        self.pre_attn_norm = nnx.LayerNorm(config.hidden_dim, rngs=rngs)
        self.q_proj = nnx.Linear(config.hidden_dim, config.hidden_dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(config.hidden_dim, config.hidden_dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(config.hidden_dim, config.hidden_dim, use_bias=False, rngs=rngs)
        self.out_proj = nnx.Linear(config.hidden_dim, config.hidden_dim, use_bias=False, rngs=rngs)
        self.pre_mlp_norm = nnx.LayerNorm(config.hidden_dim, rngs=rngs)
        self.mlp_in = nnx.Linear(config.hidden_dim, config.mlp_dim, rngs=rngs)
        self.mlp_out = nnx.Linear(config.mlp_dim, config.hidden_dim, rngs=rngs)

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
        head_dim = self.config.hidden_dim // self.config.num_heads

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
        attended = jnp.swapaxes(attended, 1, 2).reshape(batch_size, seq_len, self.config.hidden_dim)
        x = residual + self.out_proj(attended)

        residual = x
        x_norm = self.pre_mlp_norm(x)
        x = residual + self.mlp_out(jax.nn.gelu(self.mlp_in(x_norm)))
        return jnp.where(query_mask[..., None], x, 0)


class RLTokenAutoencoder(nnx.Module):
    def __init__(self, config: RLTokenConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.rl_query = nnx.Param(
            jax.random.normal(rngs.params(), (1, 1, config.hidden_dim), dtype=jnp.float32) * 0.02
        )
        self.encoder_pos_embedding = nnx.Param(
            jax.random.normal(rngs.params(), (1, config.max_prefix_len + 1, config.hidden_dim), dtype=jnp.float32)
            * 0.02
        )
        self.decoder_pos_embedding = nnx.Param(
            jax.random.normal(rngs.params(), (1, config.max_prefix_len, config.hidden_dim), dtype=jnp.float32) * 0.02
        )
        self.encoder_blocks = [RLTokenBlock(config, rngs=rngs) for _ in range(config.encoder_layers)]
        self.decoder_blocks = [RLTokenBlock(config, rngs=rngs) for _ in range(config.decoder_layers)]
        self.output_norm = nnx.LayerNorm(config.hidden_dim, rngs=rngs)
        self.output_proj = nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs)
        self.z_to_decoder = nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs)

    def __call__(
        self,
        h_vla: at.Float[at.Array, "b n d"],
        prefix_mask: at.Bool[at.Array, "b n"],
    ) -> tuple[at.Float[at.Array, "b d"], at.Float[at.Array, "b n d"]]:
        if h_vla.shape[-1] != self.config.hidden_dim:
            raise ValueError(f"Expected h_vla dim {self.config.hidden_dim}, got {h_vla.shape[-1]}")
        seq_len = h_vla.shape[1]
        if seq_len > self.config.max_prefix_len:
            raise ValueError(f"prefix length {seq_len} exceeds max_prefix_len={self.config.max_prefix_len}")

        batch_size = h_vla.shape[0]
        rl_query = jnp.broadcast_to(self.rl_query.value.astype(h_vla.dtype), (batch_size, 1, self.config.hidden_dim))
        encoder_input = jnp.concatenate([h_vla, rl_query], axis=1)
        encoder_input = encoder_input + self.encoder_pos_embedding.value[:, : seq_len + 1].astype(h_vla.dtype)
        encoder_mask = jnp.concatenate(
            [prefix_mask, jnp.ones((batch_size, 1), dtype=prefix_mask.dtype)],
            axis=1,
        )

        x = encoder_input
        for block in self.encoder_blocks:
            x = block(x, encoder_mask)
        z_rl = x[:, -1, :]

        target = jnp.where(prefix_mask[..., None], jax.lax.stop_gradient(h_vla), 0)
        z_condition = self.z_to_decoder(z_rl)[:, None, :].astype(h_vla.dtype)
        decoder_input = jnp.concatenate([z_condition, target[:, :-1, :]], axis=1)
        decoder_input = decoder_input + self.decoder_pos_embedding.value[:, :seq_len].astype(h_vla.dtype)
        decoder_key_mask = jnp.concatenate(
            [jnp.ones((batch_size, 1), dtype=prefix_mask.dtype), prefix_mask[:, :-1]],
            axis=1,
        )

        x = decoder_input
        for block in self.decoder_blocks:
            x = block(x, prefix_mask, decoder_key_mask, causal=True)
        h_hat = self.output_proj(self.output_norm(x))
        return z_rl, h_hat

    def compute_loss(
        self,
        h_vla: at.Float[at.Array, "b n d"],
        prefix_mask: at.Bool[at.Array, "b n"],
    ) -> at.Float[at.Array, " b"]:
        _, h_hat = self(h_vla, prefix_mask)
        target = jax.lax.stop_gradient(h_vla)
        token_loss = jnp.mean(jnp.square(h_hat - target), axis=-1)
        token_loss = jnp.where(prefix_mask, token_loss, 0)
        denom = jnp.maximum(jnp.sum(prefix_mask, axis=-1), 1)
        return jnp.sum(token_loss, axis=-1) / denom
