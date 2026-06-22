import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class RLTConfig:
    z_dim: int = 2048
    proprio_dim: int = 32
    action_horizon: int = 50
    action_dim: int = 32
    hidden_dim: int = 1024
    num_layers: int = 3
    fixed_std: float = 0.05
    beta: float = 10.0
    gamma: float = 0.99
    tau: float = 0.005
    reference_dropout: float = 0.5
    max_delta: float = 0.1

    @property
    def state_dim(self) -> int:
        return self.z_dim + self.proprio_dim

    @property
    def flat_action_dim(self) -> int:
        return self.action_horizon * self.action_dim


class MLP(nnx.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, num_layers: int, *, rngs: nnx.Rngs):
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.layers = []
        current_dim = in_dim
        for _ in range(num_layers - 1):
            self.layers.append(nnx.Linear(current_dim, hidden_dim, rngs=rngs))
            current_dim = hidden_dim
        self.out = nnx.Linear(current_dim, out_dim, rngs=rngs)

    def __call__(self, x: at.Float[at.Array, "b d"]) -> at.Float[at.Array, "b d"]:
        for layer in self.layers:
            x = nnx.gelu(layer(x))
        return self.out(x)


def make_state(
    z_rl: at.Float[at.Array, "b z"],
    proprio: at.Float[at.Array, "b p"],
) -> at.Float[at.Array, "b d"]:
    return jnp.concatenate([z_rl, proprio], axis=-1)


def flatten_actions(actions: at.Float[at.Array, "b h a"]) -> at.Float[at.Array, "b d"]:
    return actions.reshape(actions.shape[0], -1)


def unflatten_actions(
    actions: at.Float[at.Array, "b d"], *, action_horizon: int, action_dim: int
) -> at.Float[at.Array, "b h a"]:
    return actions.reshape(actions.shape[0], action_horizon, action_dim)


class RLTActor(nnx.Module):
    def __init__(self, config: RLTConfig, *, rngs: nnx.Rngs):
        self.config = config
        in_dim = config.state_dim + config.flat_action_dim
        self.net = MLP(in_dim, config.flat_action_dim, config.hidden_dim, config.num_layers, rngs=rngs)

    def mean_delta(
        self,
        x: at.Float[at.Array, "b d"],
        reference_action: at.Float[at.Array, "b h a"],
    ) -> at.Float[at.Array, "b h a"]:
        actor_input = jnp.concatenate([x, flatten_actions(reference_action)], axis=-1)
        delta = self.net(actor_input)
        delta = unflatten_actions(delta, action_horizon=self.config.action_horizon, action_dim=self.config.action_dim)
        return jnp.clip(delta, -self.config.max_delta, self.config.max_delta)

    def __call__(
        self,
        x: at.Float[at.Array, "b d"],
        reference_action: at.Float[at.Array, "b h a"],
        *,
        rng: at.KeyArrayLike | None = None,
        sample: bool = False,
        intervention_scale: float = 1.0,
    ) -> at.Float[at.Array, "b h a"]:
        delta = self.mean_delta(x, reference_action)
        if sample:
            if rng is None:
                raise ValueError("rng is required when sample=True")
            noise = jax.random.normal(rng, delta.shape, dtype=delta.dtype) * self.config.fixed_std
            delta = jnp.clip(delta + noise, -self.config.max_delta, self.config.max_delta)
        return reference_action + intervention_scale * delta


class RLTCritic(nnx.Module):
    def __init__(self, config: RLTConfig, *, rngs: nnx.Rngs):
        self.config = config
        in_dim = config.state_dim + config.flat_action_dim
        self.net = MLP(in_dim, 1, config.hidden_dim, config.num_layers, rngs=rngs)

    def __call__(
        self,
        x: at.Float[at.Array, "b d"],
        action: at.Float[at.Array, "b h a"],
    ) -> at.Float[at.Array, " b"]:
        critic_input = jnp.concatenate([x, flatten_actions(action)], axis=-1)
        return jnp.squeeze(self.net(critic_input), axis=-1)


class RLTTwinCritic(nnx.Module):
    def __init__(self, config: RLTConfig, *, rngs: nnx.Rngs):
        self.q1 = RLTCritic(config, rngs=rngs)
        self.q2 = RLTCritic(config, rngs=rngs)

    def __call__(
        self,
        x: at.Float[at.Array, "b d"],
        action: at.Float[at.Array, "b h a"],
    ) -> tuple[at.Float[at.Array, " b"], at.Float[at.Array, " b"]]:
        return self.q1(x, action), self.q2(x, action)

    def min_q(
        self,
        x: at.Float[at.Array, "b d"],
        action: at.Float[at.Array, "b h a"],
    ) -> at.Float[at.Array, " b"]:
        q1, q2 = self(x, action)
        return jnp.minimum(q1, q2)


class RLTActorCritic(nnx.Module):
    def __init__(self, config: RLTConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.actor = RLTActor(config, rngs=rngs)
        self.critic = RLTTwinCritic(config, rngs=rngs)
        self.target_actor = RLTActor(config, rngs=rngs)
        self.target_critic = RLTTwinCritic(config, rngs=rngs)
        self.sync_targets()

    def sync_targets(self) -> None:
        """Hard-copy online actor/critic parameters into target networks."""
        nnx.update(self.target_actor, nnx.state(self.actor))
        nnx.update(self.target_critic, nnx.state(self.critic))

    def soft_update_targets(self, tau: float | None = None) -> None:
        """Polyak update target networks: target <- tau * online + (1 - tau) * target."""
        tau = self.config.tau if tau is None else tau
        if not 0.0 <= tau <= 1.0:
            raise ValueError("tau must be in [0, 1]")
        nnx.update(self.target_actor, polyak_update_state(nnx.state(self.actor), nnx.state(self.target_actor), tau))
        nnx.update(self.target_critic, polyak_update_state(nnx.state(self.critic), nnx.state(self.target_critic), tau))

    def target_action(
        self,
        next_x: at.Float[at.Array, "b d"],
        next_reference_action: at.Float[at.Array, "b h a"],
        *,
        rng: at.KeyArrayLike | None = None,
        sample: bool = False,
    ) -> at.Float[at.Array, "b h a"]:
        """Action used only for TD target computation, produced by target actor πθ'."""
        return self.target_actor(next_x, next_reference_action, rng=rng, sample=sample)

    def target_q_min(
        self,
        next_x: at.Float[at.Array, "b d"],
        next_reference_action: at.Float[at.Array, "b h a"],
        *,
        rng: at.KeyArrayLike | None = None,
        sample_target_actor: bool = False,
    ) -> at.Float[at.Array, " b"]:
        next_action = self.target_action(next_x, next_reference_action, rng=rng, sample=sample_target_actor)
        return self.target_critic.min_q(next_x, next_action)


def discount_chunk_rewards(
    reward_seq: at.Float[at.Array, "b h"],
    *,
    gamma: float,
) -> at.Float[at.Array, " b"]:
    powers = gamma ** jnp.arange(reward_seq.shape[-1], dtype=reward_seq.dtype)
    return jnp.sum(reward_seq * powers[None, :], axis=-1)


def td3_target(
    reward_seq: at.Float[at.Array, "b h"],
    done: at.Bool[at.Array, " b"],
    next_q_min: at.Float[at.Array, " b"],
    *,
    gamma: float,
) -> at.Float[at.Array, " b"]:
    horizon = reward_seq.shape[-1]
    reward_return = discount_chunk_rewards(reward_seq, gamma=gamma)
    bootstrap = (gamma**horizon) * next_q_min * (1.0 - done.astype(next_q_min.dtype))
    return jax.lax.stop_gradient(reward_return + bootstrap)


def rlt_td3_target(
    model: RLTActorCritic,
    reward_seq: at.Float[at.Array, "b h"],
    done: at.Bool[at.Array, " b"],
    next_x: at.Float[at.Array, "b d"],
    next_reference_action: at.Float[at.Array, "b h a"],
    *,
    rng: at.KeyArrayLike | None = None,
    sample_target_actor: bool = False,
) -> at.Float[at.Array, " b"]:
    """RLT/TD3 target using target actor πθ' and target twin critics Qψ1', Qψ2'."""
    next_q_min = model.target_q_min(
        next_x,
        next_reference_action,
        rng=rng,
        sample_target_actor=sample_target_actor,
    )
    return td3_target(reward_seq, done, next_q_min, gamma=model.config.gamma)


def polyak_update_state(online_state: nnx.State, target_state: nnx.State, tau: float) -> nnx.State:
    return jax.tree.map(
        lambda online, target: tau * online + (1.0 - tau) * target,
        online_state,
        target_state,
    )


def critic_loss(
    q1: at.Float[at.Array, " b"],
    q2: at.Float[at.Array, " b"],
    target_q: at.Float[at.Array, " b"],
) -> at.Float[at.Array, ""]:
    return jnp.mean(jnp.square(q1 - target_q) + jnp.square(q2 - target_q))


def actor_loss(
    q1_for_actor: at.Float[at.Array, " b"],
    action: at.Float[at.Array, "b h a"],
    reference_action: at.Float[at.Array, "b h a"],
    *,
    beta: float,
) -> at.Float[at.Array, ""]:
    bc_penalty = jnp.mean(jnp.square(action - reference_action), axis=(-2, -1))
    return jnp.mean(-q1_for_actor + beta * bc_penalty)


def awbc_actor_loss(
    actor_action: at.Float[at.Array, "b h a"],
    data_action: at.Float[at.Array, "b h a"],
    advantage: at.Float[at.Array, " b"],
    episode_success: at.Bool[at.Array, " b"],
    *,
    temperature: float,
    max_weight: float,
    min_advantage: float,
    max_action_delta_norm: float,
    data_reference_action: at.Float[at.Array, "b h a"] | None = None,
) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if max_weight < 1.0:
        raise ValueError("max_weight must be >= 1")
    if max_action_delta_norm <= 0.0:
        raise ValueError("max_action_delta_norm must be positive")
    data_delta_norm = jnp.zeros_like(advantage)
    if data_reference_action is not None:
        data_delta = data_action - data_reference_action
        data_delta_norm = jnp.linalg.norm(data_delta.reshape(data_delta.shape[0], -1), axis=-1)
    keep = (
        episode_success.astype(jnp.bool_)
        & (advantage >= min_advantage)
        & (data_delta_norm <= max_action_delta_norm)
    )
    raw_weight = jnp.exp(jnp.maximum(advantage, 0.0) / temperature)
    weight = jnp.clip(raw_weight, 1.0, max_weight) * keep.astype(actor_action.dtype)
    per_sample_bc = jnp.mean(jnp.square(actor_action - data_action), axis=(-2, -1))
    denom = jnp.maximum(jnp.sum(weight), 1.0)
    loss = jnp.sum(weight * per_sample_bc) / denom
    kept_count = jnp.sum(keep.astype(actor_action.dtype))
    return loss, {
        "awbc_keep_fraction": jnp.mean(keep.astype(actor_action.dtype)),
        "awbc_kept_count": kept_count,
        "awbc_weight_mean": jnp.sum(weight) / jnp.maximum(kept_count, 1.0),
        "awbc_advantage_mean": jnp.mean(advantage),
        "awbc_data_delta_norm_mean": jnp.mean(data_delta_norm),
    }
