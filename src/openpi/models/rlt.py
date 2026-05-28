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

