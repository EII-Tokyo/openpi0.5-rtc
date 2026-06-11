from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


Params = dict[str, Any]


@dataclasses.dataclass(frozen=True)
class RLTActorCriticConfig:
    token_dim: int = 256
    state_dim: int = 32
    action_dim: int = 32
    action_horizon: int = 50
    rlt_chunk_horizon: int = 10
    hidden_dim: int = 512
    gamma: float = 0.99
    tau: float = 0.005
    reference_dropout: float = 0.5
    bc_coef: float = 1.0
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    @property
    def action_size(self) -> int:
        return self.rlt_chunk_horizon * self.action_dim

    @property
    def obs_size(self) -> int:
        return self.token_dim + self.state_dim + self.action_size


def _linear_params(rng: jax.Array, in_dim: int, out_dim: int) -> Params:
    scale = np.sqrt(2.0 / float(in_dim + out_dim))
    return {"w": jax.random.normal(rng, (in_dim, out_dim), dtype=jnp.float32) * scale, "b": jnp.zeros((out_dim,), dtype=jnp.float32)}


def _linear(params: Params, x: jax.Array) -> jax.Array:
    return x @ params["w"] + params["b"]


def _mlp_params(rng: jax.Array, dims: list[int]) -> list[Params]:
    keys = jax.random.split(rng, len(dims) - 1)
    return [_linear_params(key, dims[i], dims[i + 1]) for i, key in enumerate(keys)]


def _mlp(params: list[Params], x: jax.Array) -> jax.Array:
    for i, layer in enumerate(params):
        x = _linear(layer, x)
        if i != len(params) - 1:
            x = jax.nn.gelu(x)
    return x


def init_actor_critic_params(rng: jax.Array, config: RLTActorCriticConfig) -> Params:
    actor_rng, q1_rng, q2_rng = jax.random.split(rng, 3)
    actor_dims = [config.obs_size, config.hidden_dim, config.hidden_dim, config.action_size]
    critic_dims = [config.obs_size + config.action_size, config.hidden_dim, config.hidden_dim, 1]
    return {"actor": _mlp_params(actor_rng, actor_dims), "critic1": _mlp_params(q1_rng, critic_dims), "critic2": _mlp_params(q2_rng, critic_dims)}


def make_actor_input(token: jax.Array, state: jax.Array, reference_action_chunk: jax.Array, config: RLTActorCriticConfig) -> jax.Array:
    ref = reference_action_chunk[:, : config.rlt_chunk_horizon, : config.action_dim].reshape(reference_action_chunk.shape[0], -1)
    state = state[:, : config.state_dim]
    return jnp.concatenate([token, state, ref], axis=-1)


def actor_apply(params: Params, token: jax.Array, state: jax.Array, reference_action_chunk: jax.Array, config: RLTActorCriticConfig) -> jax.Array:
    x = make_actor_input(token, state, reference_action_chunk, config)
    action = _mlp(params["actor"], x)
    return action.reshape((-1, config.rlt_chunk_horizon, config.action_dim))


def critic_apply(critic_params: list[Params], token: jax.Array, state: jax.Array, reference_action_chunk: jax.Array, action_chunk: jax.Array, config: RLTActorCriticConfig) -> jax.Array:
    x = make_actor_input(token, state, reference_action_chunk, config)
    action = action_chunk[:, : config.rlt_chunk_horizon, : config.action_dim].reshape(action_chunk.shape[0], -1)
    q = _mlp(critic_params, jnp.concatenate([x, action], axis=-1))
    return q[:, 0]


def _drop_reference(rng: jax.Array, reference_action_chunk: jax.Array, dropout: float) -> jax.Array:
    keep = jax.random.bernoulli(rng, 1.0 - dropout, (reference_action_chunk.shape[0], 1, 1))
    return jnp.where(keep, reference_action_chunk, 0.0)


def critic_loss(params: Params, target_params: Params, batch, token: jax.Array, next_token: jax.Array, config: RLTActorCriticConfig, rng: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
    ref_rng, next_ref_rng = jax.random.split(rng)
    ref = _drop_reference(ref_rng, batch.reference_action_chunk, config.reference_dropout)
    next_ref = _drop_reference(next_ref_rng, batch.next_reference_action_chunk, config.reference_dropout)
    next_action = actor_apply(target_params, next_token, batch.next_state, next_ref, config)
    target_q1 = critic_apply(target_params["critic1"], next_token, batch.next_state, next_ref, next_action, config)
    target_q2 = critic_apply(target_params["critic2"], next_token, batch.next_state, next_ref, next_action, config)
    target_q = batch.reward + config.gamma * (1.0 - batch.done.astype(jnp.float32)) * jnp.minimum(target_q1, target_q2)
    q1 = critic_apply(params["critic1"], token, batch.state, ref, batch.executed_action_chunk, config)
    q2 = critic_apply(params["critic2"], token, batch.state, ref, batch.executed_action_chunk, config)
    loss = jnp.mean(jnp.square(q1 - target_q)) + jnp.mean(jnp.square(q2 - target_q))
    return loss, {"critic_loss": loss, "q1": jnp.mean(q1), "target_q": jnp.mean(target_q)}


def actor_loss(params: Params, token: jax.Array, batch, config: RLTActorCriticConfig, rng: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
    ref = _drop_reference(rng, batch.reference_action_chunk, config.reference_dropout)
    action = actor_apply(params, token, batch.state, ref, config)
    q = critic_apply(params["critic1"], token, batch.state, ref, action, config)
    bc_target = batch.reference_action_chunk[:, : config.rlt_chunk_horizon, : config.action_dim]
    bc_loss = jnp.mean(jnp.square(action - bc_target))
    loss = -jnp.mean(q) + config.bc_coef * bc_loss
    return loss, {"actor_loss": loss, "actor_q": jnp.mean(q), "bc_loss": bc_loss}


def soft_update(params: Params, target_params: Params, tau: float) -> Params:
    return jax.tree.map(lambda p, t: tau * p + (1.0 - tau) * t, params, target_params)
