from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


Params = dict[str, Any]

ALOHA_GRIPPER_ACTION_INDICES = (6, 13)


@dataclasses.dataclass(frozen=True)
class RLTActorCriticConfig:
    token_dim: int = 2048
    state_dim: int = 14
    action_dim: int = 14
    action_horizon: int = 30
    rlt_chunk_horizon: int = 30
    action_start_index: int = 10
    hidden_dim: int = 512
    actor_hidden_layers: int = 2
    critic_hidden_layers: int = 2
    gamma: float = 0.99
    tau: float = 0.005
    target_bootstrap_steps: int = 0
    rl_loss_coef: float = 1.0
    critic_loss_coef: float = 1.0
    actor_update_period: int = 2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    actor_clip_gradient_norm: float = 0.0
    critic_clip_gradient_norm: float = 0.0
    target_policy_noise: float = 0.0
    target_policy_noise_clip: float = 0.15
    reference_deviation_threshold: float = 0.047
    reference_deviation_penalty_coef: float = 1000.0
    reference_action_dropout: float = 0.5

    def __post_init__(self) -> None:
        if self.rlt_chunk_horizon != self.action_horizon:
            raise ValueError(
                f"RLT actor currently expects full C-step chunks: "
                f"rlt_chunk_horizon={self.rlt_chunk_horizon}, action_horizon={self.action_horizon}"
            )
        if self.action_start_index < 0:
            raise ValueError(f"action_start_index must be non-negative, got {self.action_start_index}")
        if self.target_bootstrap_steps < 0:
            raise ValueError(f"target_bootstrap_steps must be non-negative, got {self.target_bootstrap_steps}")
        if self.rl_loss_coef < 0:
            raise ValueError(f"rl_loss_coef must be non-negative, got {self.rl_loss_coef}")
        if self.actor_clip_gradient_norm < 0:
            raise ValueError(f"actor_clip_gradient_norm must be non-negative, got {self.actor_clip_gradient_norm}")
        if self.critic_clip_gradient_norm < 0:
            raise ValueError(f"critic_clip_gradient_norm must be non-negative, got {self.critic_clip_gradient_norm}")
        if self.critic_loss_coef <= 0:
            raise ValueError(f"critic_loss_coef must be positive, got {self.critic_loss_coef}")
        if self.actor_update_period <= 0:
            raise ValueError(f"actor_update_period must be positive, got {self.actor_update_period}")
        if self.actor_hidden_layers < 0:
            raise ValueError(f"actor_hidden_layers must be non-negative, got {self.actor_hidden_layers}")
        if self.critic_hidden_layers < 0:
            raise ValueError(f"critic_hidden_layers must be non-negative, got {self.critic_hidden_layers}")
        if self.reference_deviation_threshold < 0:
            raise ValueError(
                f"reference_deviation_threshold must be non-negative, got {self.reference_deviation_threshold}"
            )
        if self.reference_deviation_penalty_coef < 0:
            raise ValueError(
                f"reference_deviation_penalty_coef must be non-negative, got {self.reference_deviation_penalty_coef}"
            )
        if self.reference_deviation_threshold <= 0:
            raise ValueError("reference_deviation_threshold must be positive")
        if not 0.0 <= self.reference_action_dropout < 1.0:
            raise ValueError(
                f"reference_action_dropout must be in [0, 1), got {self.reference_action_dropout}"
            )

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


def gripper_action_mask(config: RLTActorCriticConfig) -> jax.Array:
    mask = jnp.ones((config.action_dim,), dtype=jnp.float32)
    for index in ALOHA_GRIPPER_ACTION_INDICES:
        if index < config.action_dim:
            mask = mask.at[index].set(0.0)
    return mask


def action_window(action_chunk: jax.Array, config: RLTActorCriticConfig) -> jax.Array:
    """Return the C-step action window controlled by RLT.

    In inference-time RTC, a newly sampled chunk starts executing only after
    the handoff delay. With the current robot settings, the actor/critic window
    is therefore 10:35 inside the VLA 50-step chunk. Replay shards may already
    store only this C-step window; in that case return the leading C steps.
    """
    horizon = config.rlt_chunk_horizon
    start = config.action_start_index
    if action_chunk.shape[1] >= start + horizon:
        return action_chunk[:, start : start + horizon, : config.action_dim]
    return action_chunk[:, :horizon, : config.action_dim]


def preserve_gripper_actions(action_chunk: jax.Array, reference_action_chunk: jax.Array, config: RLTActorCriticConfig) -> jax.Array:
    """Keep ALOHA gripper dimensions equal to the reference/VLA action window."""
    reference = action_window(reference_action_chunk, config)
    action = action_chunk[:, : config.rlt_chunk_horizon, : config.action_dim]
    for index in ALOHA_GRIPPER_ACTION_INDICES:
        if index < config.action_dim:
            action = action.at[:, :, index].set(reference[:, :, index])
    return action


def init_actor_critic_params(rng: jax.Array, config: RLTActorCriticConfig) -> Params:
    actor_rng, q1_rng, q2_rng = jax.random.split(rng, 3)
    actor_dims = [config.obs_size, *([config.hidden_dim] * config.actor_hidden_layers), config.action_size]
    critic_dims = [config.critic_input_size, *([config.hidden_dim] * config.critic_hidden_layers), 1]
    return {"actor": _mlp_params(actor_rng, actor_dims), "critic1": _mlp_params(q1_rng, critic_dims), "critic2": _mlp_params(q2_rng, critic_dims)}


def make_actor_input(rlt_token: jax.Array, state: jax.Array, reference_action_chunk: jax.Array, config: RLTActorCriticConfig) -> jax.Array:
    rlt_token = layernorm_rlt_token(rlt_token)
    ref = action_window(reference_action_chunk, config).reshape(reference_action_chunk.shape[0], -1)
    state = state[:, : config.state_dim]
    return jnp.concatenate([rlt_token, state, ref], axis=-1)


def actor_apply(
    actor_params: list[Params],
    rlt_token: jax.Array,
    state: jax.Array,
    reference_action_chunk: jax.Array,
    config: RLTActorCriticConfig,
    *,
    reference_action_input: jax.Array | None = None,
) -> jax.Array:
    if reference_action_input is None:
        reference_action_input = reference_action_chunk
    x = make_actor_input(rlt_token, state, reference_action_input, config)
    action = _mlp(actor_params, x).reshape((-1, config.rlt_chunk_horizon, config.action_dim))
    return preserve_gripper_actions(action, reference_action_chunk, config)


def maybe_drop_reference_action_input(
    reference_action_chunk: jax.Array, config: RLTActorCriticConfig, rng: jax.Array
) -> jax.Array:
    if config.reference_action_dropout <= 0.0:
        return reference_action_chunk
    keep = jax.random.bernoulli(
        rng,
        1.0 - config.reference_action_dropout,
        (reference_action_chunk.shape[0], 1, 1),
    )
    return jnp.where(keep, reference_action_chunk, jnp.zeros_like(reference_action_chunk))


def make_critic_input(rlt_token: jax.Array, state: jax.Array, action_chunk: jax.Array, config: RLTActorCriticConfig) -> jax.Array:
    rlt_token = layernorm_rlt_token(rlt_token)
    state = state[:, : config.state_dim]
    action = action_window(action_chunk, config).reshape(action_chunk.shape[0], -1)
    return jnp.concatenate([rlt_token, state, action], axis=-1)


def critic_apply(critic_params: list[Params], rlt_token: jax.Array, state: jax.Array, action_chunk: jax.Array, config: RLTActorCriticConfig) -> jax.Array:
    x = make_critic_input(rlt_token, state, action_chunk, config)
    q = _mlp(critic_params, x)
    return q[:, 0]


def critic_loss(params: Params, target_params: Params, batch, config: RLTActorCriticConfig, rng: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
    next_ref = batch.normalized_next_reference_action_chunk
    next_ref_input = maybe_drop_reference_action_input(next_ref, config, rng)
    next_action = actor_apply(
        params["actor"],
        batch.next_rlt_token,
        batch.normalized_next_state,
        next_ref,
        config,
        reference_action_input=next_ref_input,
    )
    if config.target_policy_noise > 0.0:
        noise = jax.random.normal(rng, next_action.shape, dtype=next_action.dtype) * config.target_policy_noise
        noise = jnp.clip(noise, -config.target_policy_noise_clip, config.target_policy_noise_clip)
        next_action = preserve_gripper_actions(next_action + noise, next_ref, config)
    target_q1 = critic_apply(target_params["critic1"], batch.next_rlt_token, batch.normalized_next_state, next_action, config)
    target_q2 = critic_apply(target_params["critic2"], batch.next_rlt_token, batch.normalized_next_state, next_action, config)
    bootstrap_steps = config.target_bootstrap_steps or config.rlt_chunk_horizon
    bootstrap_discount = config.gamma**bootstrap_steps
    target_q = batch.td_reward + bootstrap_discount * (1.0 - batch.done.astype(jnp.float32)) * jnp.minimum(target_q1, target_q2)
    q1 = critic_apply(params["critic1"], batch.rlt_token, batch.normalized_state, batch.normalized_executed_action_chunk, config)
    q2 = critic_apply(params["critic2"], batch.rlt_token, batch.normalized_state, batch.normalized_executed_action_chunk, config)
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
) -> tuple[jax.Array, dict[str, jax.Array]]:
    ref = batch.normalized_reference_action_chunk
    ref_input = maybe_drop_reference_action_input(ref, config, rng)
    action = actor_apply(
        params["actor"],
        batch.rlt_token,
        batch.normalized_state,
        ref,
        config,
        reference_action_input=ref_input,
    )
    q1 = critic_apply(params["critic1"], batch.rlt_token, batch.normalized_state, action, config)
    q2 = critic_apply(params["critic2"], batch.rlt_token, batch.normalized_state, action, config)
    q = jnp.minimum(q1, q2)
    non_gripper_mask = gripper_action_mask(config)[None, None, :]
    rl_loss = -jnp.mean(q)
    rl_loss_weighted = config.rl_loss_coef * rl_loss
    loss = rl_loss_weighted
    action_reference = action_window(batch.normalized_reference_action_chunk, config)
    reference_deviation_abs = jnp.abs(action - action_reference) * non_gripper_mask
    reference_deviation_abs_max = jnp.max(reference_deviation_abs)
    reference_deviation_abs_mean = jnp.sum(reference_deviation_abs) / jnp.maximum(
        jnp.sum(jnp.ones_like(reference_deviation_abs) * non_gripper_mask), 1.0
    )
    reference_deviation_excess = jnp.maximum(
        reference_deviation_abs - config.reference_deviation_threshold, 0.0
    ) * non_gripper_mask
    reference_deviation_penalty = jnp.sum(jnp.square(reference_deviation_excess)) / jnp.maximum(
        jnp.sum(jnp.ones_like(reference_deviation_excess) * non_gripper_mask), 1.0
    )
    reference_deviation_penalty_weighted = config.reference_deviation_penalty_coef * reference_deviation_penalty
    loss = loss + reference_deviation_penalty_weighted
    return loss, {
        "actor_loss": loss,
        "actor_rl_loss": rl_loss,
        "actor_rl_loss_weighted": rl_loss_weighted,
        "actor_q": jnp.mean(q),
        "actor_q1": jnp.mean(q1),
        "actor_q2": jnp.mean(q2),
        "reference_deviation_abs_max": reference_deviation_abs_max,
        "reference_deviation_abs_mean": reference_deviation_abs_mean,
        "reference_deviation_penalty": reference_deviation_penalty,
        "reference_deviation_penalty_weighted": reference_deviation_penalty_weighted,
    }


def soft_update(params: Params, target_params: Params, tau: float) -> Params:
    return jax.tree.map(lambda p, t: tau * p + (1.0 - tau) * t, params, target_params)
