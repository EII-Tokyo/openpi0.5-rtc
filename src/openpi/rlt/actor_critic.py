from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


Params = dict[str, Any]


@dataclasses.dataclass(frozen=True)
class RLTActorCriticConfig:
    token_dim: int = 2048
    state_dim: int = 14
    action_dim: int = 14
    action_horizon: int = 30
    rlt_chunk_horizon: int = 30
    hidden_dim: int = 512
    actor_hidden_layers: int = 2
    critic_hidden_layers: int = 2
    gamma: float = 0.99
    tau: float = 0.005
    reference_dropout: float = 0.5
    rl_loss_coef: float = 1.0
    bc_coef: float = 1.0
    action_smooth_coef: float = 0.0
    action_accel_coef: float = 0.0
    critic_loss_coef: float = 1.0
    bc_warmup_steps: int = 200
    actor_update_period: int = 2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    target_policy_noise: float = 0.0
    target_policy_noise_clip: float = 0.15

    def __post_init__(self) -> None:
        if self.rlt_chunk_horizon != self.action_horizon:
            raise ValueError(
                f"RLT actor currently expects full C-step chunks: "
                f"rlt_chunk_horizon={self.rlt_chunk_horizon}, action_horizon={self.action_horizon}"
            )
        if self.bc_warmup_steps < 0:
            raise ValueError(f"bc_warmup_steps must be non-negative, got {self.bc_warmup_steps}")
        if self.rl_loss_coef < 0:
            raise ValueError(f"rl_loss_coef must be non-negative, got {self.rl_loss_coef}")
        if self.critic_loss_coef <= 0:
            raise ValueError(f"critic_loss_coef must be positive, got {self.critic_loss_coef}")
        if self.action_smooth_coef < 0:
            raise ValueError(f"action_smooth_coef must be non-negative, got {self.action_smooth_coef}")
        if self.action_accel_coef < 0:
            raise ValueError(f"action_accel_coef must be non-negative, got {self.action_accel_coef}")
        if self.actor_update_period <= 0:
            raise ValueError(f"actor_update_period must be positive, got {self.actor_update_period}")
        if self.actor_hidden_layers < 0:
            raise ValueError(f"actor_hidden_layers must be non-negative, got {self.actor_hidden_layers}")
        if self.critic_hidden_layers < 0:
            raise ValueError(f"critic_hidden_layers must be non-negative, got {self.critic_hidden_layers}")

    @property
    def action_size(self) -> int:
        return self.rlt_chunk_horizon * self.action_dim

    @property
    def obs_size(self) -> int:
        return self.token_dim + self.state_dim + self.action_size

    @property
    def critic_input_size(self) -> int:
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


def layernorm_rlt_token(rlt_token: jax.Array) -> jax.Array:
    return jax.nn.standardize(rlt_token, axis=-1, epsilon=1e-6)


def init_actor_critic_params(rng: jax.Array, config: RLTActorCriticConfig) -> Params:
    actor_rng, q1_rng, q2_rng = jax.random.split(rng, 3)
    actor_dims = [config.obs_size, *([config.hidden_dim] * config.actor_hidden_layers), config.action_size]
    critic_dims = [config.critic_input_size, *([config.hidden_dim] * config.critic_hidden_layers), 1]
    return {"actor": _mlp_params(actor_rng, actor_dims), "critic1": _mlp_params(q1_rng, critic_dims), "critic2": _mlp_params(q2_rng, critic_dims)}


def make_actor_input(rlt_token: jax.Array, state: jax.Array, reference_action_chunk: jax.Array, config: RLTActorCriticConfig) -> jax.Array:
    rlt_token = layernorm_rlt_token(rlt_token)
    ref = reference_action_chunk[:, : config.rlt_chunk_horizon, : config.action_dim].reshape(reference_action_chunk.shape[0], -1)
    state = state[:, : config.state_dim]
    return jnp.concatenate([rlt_token, state, ref], axis=-1)


def actor_apply(actor_params: list[Params], rlt_token: jax.Array, state: jax.Array, reference_action_chunk: jax.Array, config: RLTActorCriticConfig) -> jax.Array:
    x = make_actor_input(rlt_token, state, reference_action_chunk, config)
    action = _mlp(actor_params, x)
    return action.reshape((-1, config.rlt_chunk_horizon, config.action_dim))


def make_critic_input(rlt_token: jax.Array, state: jax.Array, action_chunk: jax.Array, config: RLTActorCriticConfig) -> jax.Array:
    rlt_token = layernorm_rlt_token(rlt_token)
    state = state[:, : config.state_dim]
    action = action_chunk[:, : config.rlt_chunk_horizon, : config.action_dim].reshape(action_chunk.shape[0], -1)
    return jnp.concatenate([rlt_token, state, action], axis=-1)


def critic_apply(critic_params: list[Params], rlt_token: jax.Array, state: jax.Array, action_chunk: jax.Array, config: RLTActorCriticConfig) -> jax.Array:
    x = make_critic_input(rlt_token, state, action_chunk, config)
    q = _mlp(critic_params, x)
    return q[:, 0]


def _drop_reference(rng: jax.Array, reference_action_chunk: jax.Array, dropout: float) -> jax.Array:
    keep = jax.random.bernoulli(rng, 1.0 - dropout, (reference_action_chunk.shape[0], 1, 1))
    return jnp.where(keep, reference_action_chunk, 0.0)


def critic_loss(params: Params, target_params: Params, batch, config: RLTActorCriticConfig, rng: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
    next_ref = batch.next_reference_action_chunk
    next_action = actor_apply(params["actor"], batch.next_rlt_token, batch.next_state, next_ref, config)
    if config.target_policy_noise > 0.0:
        noise = jax.random.normal(rng, next_action.shape, dtype=next_action.dtype) * config.target_policy_noise
        noise = jnp.clip(noise, -config.target_policy_noise_clip, config.target_policy_noise_clip)
        next_action = next_action + noise
    target_q1 = critic_apply(target_params["critic1"], batch.next_rlt_token, batch.next_state, next_action, config)
    target_q2 = critic_apply(target_params["critic2"], batch.next_rlt_token, batch.next_state, next_action, config)
    bootstrap_discount = config.gamma**config.rlt_chunk_horizon
    target_q = batch.reward + bootstrap_discount * (1.0 - batch.done.astype(jnp.float32)) * jnp.minimum(target_q1, target_q2)
    q1 = critic_apply(params["critic1"], batch.rlt_token, batch.state, batch.executed_action_chunk, config)
    q2 = critic_apply(params["critic2"], batch.rlt_token, batch.state, batch.executed_action_chunk, config)
    unweighted_loss = jnp.mean(jnp.square(q1 - target_q)) + jnp.mean(jnp.square(q2 - target_q))
    loss = config.critic_loss_coef * unweighted_loss
    return loss, {
        "critic_loss": loss,
        "critic_loss_unweighted": unweighted_loss,
        "q1": jnp.mean(q1),
        "target_q": jnp.mean(target_q),
    }


def actor_loss(
    params: Params,
    batch,
    config: RLTActorCriticConfig,
    rng: jax.Array,
    step_idx: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    ref = _drop_reference(rng, batch.reference_action_chunk, config.reference_dropout)
    action = actor_apply(params["actor"], batch.rlt_token, batch.state, ref, config)
    q1 = critic_apply(params["critic1"], batch.rlt_token, batch.state, action, config)
    q2 = critic_apply(params["critic2"], batch.rlt_token, batch.state, action, config)
    q = jnp.minimum(q1, q2)
    bc_target = batch.executed_action_chunk[:, : config.rlt_chunk_horizon, : config.action_dim]
    bc_mask = batch.executed_action_mask[:, : config.rlt_chunk_horizon].astype(jnp.float32)[..., None]
    bc_loss = jnp.sum(jnp.square(action - bc_target) * bc_mask) / jnp.maximum(jnp.sum(bc_mask) * config.action_dim, 1.0)
    action_delta = action[:, 1:] - action[:, :-1]
    target_delta = bc_target[:, 1:] - bc_target[:, :-1]
    smooth_mask = bc_mask[:, 1:] * bc_mask[:, :-1]
    smooth_loss = jnp.sum(jnp.square(action_delta - target_delta) * smooth_mask) / jnp.maximum(
        jnp.sum(smooth_mask) * config.action_dim, 1.0
    )
    action_accel = action[:, 2:] - 2.0 * action[:, 1:-1] + action[:, :-2]
    target_accel = bc_target[:, 2:] - 2.0 * bc_target[:, 1:-1] + bc_target[:, :-2]
    accel_mask = bc_mask[:, 2:] * bc_mask[:, 1:-1] * bc_mask[:, :-2]
    accel_loss = jnp.sum(jnp.square(action_accel - target_accel) * accel_mask) / jnp.maximum(
        jnp.sum(accel_mask) * config.action_dim, 1.0
    )
    rl_loss = -jnp.mean(q)
    rl_active = (step_idx >= config.bc_warmup_steps).astype(jnp.float32)
    rl_loss_weighted = rl_active * config.rl_loss_coef * rl_loss
    bc_loss_weighted = config.bc_coef * bc_loss
    smooth_loss_weighted = config.action_smooth_coef * smooth_loss
    accel_loss_weighted = config.action_accel_coef * accel_loss
    loss = rl_loss_weighted + bc_loss_weighted + smooth_loss_weighted + accel_loss_weighted
    return loss, {
        "actor_loss": loss,
        "actor_rl_loss": rl_loss,
        "actor_rl_active": rl_active,
        "actor_rl_loss_weighted": rl_loss_weighted,
        "bc_loss_weighted": bc_loss_weighted,
        "smooth_loss_weighted": smooth_loss_weighted,
        "accel_loss_weighted": accel_loss_weighted,
        "actor_q": jnp.mean(q),
        "actor_q1": jnp.mean(q1),
        "actor_q2": jnp.mean(q2),
        "bc_loss": bc_loss,
        "smooth_loss": smooth_loss,
        "accel_loss": accel_loss,
        "smooth_valid_steps": jnp.sum(smooth_mask),
        "accel_valid_steps": jnp.sum(accel_mask),
        "bc_valid_steps": jnp.sum(bc_mask),
    }


def soft_update(params: Params, target_params: Params, tau: float) -> Params:
    return jax.tree.map(lambda p, t: tau * p + (1.0 - tau) * t, params, target_params)
