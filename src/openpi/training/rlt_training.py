import dataclasses

from flax import nnx
from flax import struct
import jax
import jax.numpy as jnp
import optax

from openpi.models import rlt
from openpi.shared import array_typing as at


@struct.dataclass
class RLTReplayBatch:
    x: at.Float[at.Array, "b d"]
    action: at.Float[at.Array, "b h a"]
    reference_action: at.Float[at.Array, "b h a"]
    reward_seq: at.Float[at.Array, "b h"]
    next_x: at.Float[at.Array, "b d"]
    next_reference_action: at.Float[at.Array, "b h a"]
    done: at.Bool[at.Array, " b"]


@dataclasses.dataclass(frozen=True)
class RLTTrainingConfig:
    model: rlt.RLTConfig = dataclasses.field(default_factory=rlt.RLTConfig)
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    policy_delay: int = 2
    actor_publish_interval: int = 500
    target_actor_noise: bool = False


@struct.dataclass
class RLTTrainState:
    step: at.Int[at.ArrayLike, ""]
    params: nnx.State
    model_def: nnx.GraphDef[rlt.RLTActorCritic]
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    actor_tx: optax.GradientTransformation = struct.field(pytree_node=False)
    critic_tx: optax.GradientTransformation = struct.field(pytree_node=False)
    policy_delay: int = struct.field(pytree_node=False)
    actor_publish_interval: int = struct.field(pytree_node=False)
    target_actor_noise: bool = struct.field(pytree_node=False)


def make_replay_batch(
    *,
    z_rl: at.Float[at.Array, "b z"],
    proprio: at.Float[at.Array, "b p"],
    action: at.Float[at.Array, "b h a"],
    reference_action: at.Float[at.Array, "b h a"],
    reward_seq: at.Float[at.Array, "b h"],
    next_z_rl: at.Float[at.Array, "b z"],
    next_proprio: at.Float[at.Array, "b p"],
    next_reference_action: at.Float[at.Array, "b h a"],
    done: at.Bool[at.Array, " b"],
) -> RLTReplayBatch:
    return RLTReplayBatch(
        x=rlt.make_state(z_rl, proprio),
        action=action,
        reference_action=reference_action,
        reward_seq=reward_seq,
        next_x=rlt.make_state(next_z_rl, next_proprio),
        next_reference_action=next_reference_action,
        done=done,
    )


def init_train_state(config: RLTTrainingConfig, rng: at.KeyArrayLike) -> RLTTrainState:
    model = rlt.RLTActorCritic(config.model, rngs=nnx.Rngs(rng))
    actor_tx = optax.adam(config.actor_lr)
    critic_tx = optax.adam(config.critic_lr)
    return RLTTrainState(
        step=jnp.asarray(0, dtype=jnp.int32),
        params=nnx.state(model),
        model_def=nnx.graphdef(model),
        actor_opt_state=actor_tx.init(nnx.state(model.actor)),
        critic_opt_state=critic_tx.init(nnx.state(model.critic)),
        actor_tx=actor_tx,
        critic_tx=critic_tx,
        policy_delay=config.policy_delay,
        actor_publish_interval=config.actor_publish_interval,
        target_actor_noise=config.target_actor_noise,
    )


def should_publish_actor(step: int, *, actor_updated: bool, actor_publish_interval: int) -> bool:
    if not actor_updated or actor_publish_interval <= 0:
        return False
    return step % actor_publish_interval == 0


def actor_params_for_inference(state: RLTTrainState) -> nnx.State:
    model = nnx.merge(state.model_def, state.params)
    return nnx.state(model.actor)


def critic_params_for_inference(state: RLTTrainState) -> nnx.State:
    model = nnx.merge(state.model_def, state.params)
    return nnx.state(model.critic)


def train_step(
    state: RLTTrainState,
    batch: RLTReplayBatch,
    rng: at.KeyArrayLike,
) -> tuple[RLTTrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    target_rng, actor_rng = jax.random.split(rng)

    target_q = rlt.rlt_td3_target(
        model,
        batch.reward_seq,
        batch.done,
        batch.next_x,
        batch.next_reference_action,
        rng=target_rng,
        sample_target_actor=state.target_actor_noise,
    )

    def critic_loss_fn(critic: rlt.RLTTwinCritic):
        q1, q2 = critic(batch.x, batch.action)
        loss = rlt.critic_loss(q1, q2, target_q)
        return loss, {
            "q1_mean": jnp.mean(q1),
            "q2_mean": jnp.mean(q2),
            "target_q_mean": jnp.mean(target_q),
            "q1_loss": jnp.mean(jnp.square(q1 - target_q)),
            "q2_loss": jnp.mean(jnp.square(q2 - target_q)),
        }

    (critic_loss_value, critic_info), critic_grads = nnx.value_and_grad(critic_loss_fn, has_aux=True)(model.critic)
    critic_params = nnx.state(model.critic)
    critic_updates, critic_opt_state = state.critic_tx.update(
        critic_grads,
        state.critic_opt_state,
        critic_params,
    )
    nnx.update(model.critic, optax.apply_updates(critic_params, critic_updates))

    next_step = int(state.step) + 1
    actor_updated = next_step % state.policy_delay == 0
    actor_loss_value = jnp.asarray(0.0, dtype=critic_loss_value.dtype)
    actor_q_value = jnp.asarray(0.0, dtype=critic_loss_value.dtype)
    reference_q_value = jnp.asarray(0.0, dtype=critic_loss_value.dtype)
    q_advantage = jnp.asarray(0.0, dtype=critic_loss_value.dtype)
    actor_delta_norm = jnp.asarray(0.0, dtype=critic_loss_value.dtype)
    actor_opt_state = state.actor_opt_state
    if actor_updated:

        def actor_loss_fn(actor: rlt.RLTActor):
            action = actor(batch.x, batch.reference_action, rng=actor_rng, sample=False)
            q1_for_actor = model.critic.q1(batch.x, action)
            q1_for_reference = model.critic.q1(batch.x, batch.reference_action)
            loss = rlt.actor_loss(q1_for_actor, action, batch.reference_action, beta=model.config.beta)
            delta = action - batch.reference_action
            actor_q_mean = jnp.mean(q1_for_actor)
            reference_q_mean = jnp.mean(q1_for_reference)
            return loss, {
                "actor_q_value": actor_q_mean,
                "reference_q_value": reference_q_mean,
                "q_advantage": actor_q_mean - reference_q_mean,
                "actor_delta_norm": jnp.mean(jnp.linalg.norm(delta.reshape(delta.shape[0], -1), axis=-1)),
            }

        (actor_loss_value, actor_info), actor_grads = nnx.value_and_grad(actor_loss_fn, has_aux=True)(model.actor)
        actor_params = nnx.state(model.actor)
        actor_updates, actor_opt_state = state.actor_tx.update(
            actor_grads,
            state.actor_opt_state,
            actor_params,
        )
        nnx.update(model.actor, optax.apply_updates(actor_params, actor_updates))
        model.soft_update_targets()
        actor_q_value = actor_info["actor_q_value"]
        reference_q_value = actor_info["reference_q_value"]
        q_advantage = actor_info["q_advantage"]
        actor_delta_norm = actor_info["actor_delta_norm"]

    publish_actor = should_publish_actor(
        next_step,
        actor_updated=actor_updated,
        actor_publish_interval=state.actor_publish_interval,
    )
    new_state = dataclasses.replace(
        state,
        step=jnp.asarray(next_step, dtype=jnp.int32),
        params=nnx.state(model),
        actor_opt_state=actor_opt_state,
        critic_opt_state=critic_opt_state,
    )
    info = {
        "critic_loss": critic_loss_value,
        "critic_q1_loss": critic_info["q1_loss"],
        "critic_q2_loss": critic_info["q2_loss"],
        "actor_loss": actor_loss_value,
        "actor_updated": jnp.asarray(actor_updated),
        "publish_actor": jnp.asarray(publish_actor),
        "actor_q_value": actor_q_value,
        "reference_q_value": reference_q_value,
        "q_advantage": q_advantage,
        "actor_delta_norm": actor_delta_norm,
        "q1_mean": critic_info["q1_mean"],
        "q2_mean": critic_info["q2_mean"],
        "target_q_mean": critic_info["target_q_mean"],
        "beta": jnp.asarray(model.config.beta, dtype=critic_loss_value.dtype),
    }
    return new_state, info
