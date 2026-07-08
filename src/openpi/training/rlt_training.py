import dataclasses

from flax import nnx
from flax import struct
import jax
import jax.numpy as jnp
import optax

from openpi.models import rlt
from openpi.shared import array_typing as at

ACTOR_LOSS_MODE_TD3 = 0
ACTOR_LOSS_MODE_AWBC = 1
CRITIC_LOSS_MODE_TD3 = 0
CRITIC_LOSS_MODE_CQL = 1
CRITIC_LOSS_MODE_CALQL = 2
CRITIC_TARGET_ACTION_MODE_TARGET_ACTOR = 0
CRITIC_TARGET_ACTION_MODE_REFERENCE_ACTION = 1


@struct.dataclass
class RLTReplayBatch:
    x: at.Float[at.Array, "b d"]
    action: at.Float[at.Array, "b h a"]
    reference_action: at.Float[at.Array, "b h a"]
    reward_seq: at.Float[at.Array, "b h"]
    next_x: at.Float[at.Array, "b d"]
    next_reference_action: at.Float[at.Array, "b h a"]
    done: at.Bool[at.Array, " b"]
    episode_success: at.Bool[at.Array, " b"]
    reference_value: at.Float[at.Array, " b"]


@dataclasses.dataclass(frozen=True)
class RLTTrainingConfig:
    model: rlt.RLTConfig = dataclasses.field(default_factory=rlt.RLTConfig)
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    policy_delay: int = 2
    actor_publish_interval: int = 500
    target_actor_noise: bool = True
    critic_target_action_mode: str = "target_actor"
    actor_loss_mode: str = "td3"
    critic_loss_mode: str = "td3"
    conservative_alpha: float = 0.0
    awbc_temperature: float = 0.2
    awbc_max_weight: float = 20.0
    awbc_min_advantage: float = 0.0
    awbc_max_action_delta_norm: float = 2.0


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
    critic_target_action_mode: int = struct.field(pytree_node=False)
    actor_loss_mode: int = struct.field(pytree_node=False)
    critic_loss_mode: int = struct.field(pytree_node=False)
    conservative_alpha: float = struct.field(pytree_node=False)
    awbc_temperature: float = struct.field(pytree_node=False)
    awbc_max_weight: float = struct.field(pytree_node=False)
    awbc_min_advantage: float = struct.field(pytree_node=False)
    awbc_max_action_delta_norm: float = struct.field(pytree_node=False)


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
    episode_success: at.Bool[at.Array, " b"] | None = None,
    reference_value: at.Float[at.Array, " b"] | None = None,
    reference_gamma: float = 0.99,
) -> RLTReplayBatch:
    if episode_success is None:
        episode_success = jnp.sum(reward_seq, axis=-1) > 0.0
    if reference_value is None:
        reference_value = rlt.discount_chunk_rewards(reward_seq, gamma=reference_gamma)
    return RLTReplayBatch(
        x=rlt.make_state(z_rl, proprio),
        action=action,
        reference_action=reference_action,
        reward_seq=reward_seq,
        next_x=rlt.make_state(next_z_rl, next_proprio),
        next_reference_action=next_reference_action,
        done=done,
        episode_success=episode_success,
        reference_value=reference_value,
    )


def init_train_state(config: RLTTrainingConfig, rng: at.KeyArrayLike) -> RLTTrainState:
    model = rlt.RLTActorCritic(config.model, rngs=nnx.Rngs(rng))
    actor_tx = optax.adam(config.actor_lr)
    critic_tx = optax.adam(config.critic_lr)
    actor_loss_mode = _actor_loss_mode_id(config.actor_loss_mode)
    critic_loss_mode = _critic_loss_mode_id(config.critic_loss_mode)
    critic_target_action_mode = _critic_target_action_mode_id(config.critic_target_action_mode)
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
        critic_target_action_mode=critic_target_action_mode,
        actor_loss_mode=actor_loss_mode,
        critic_loss_mode=critic_loss_mode,
        conservative_alpha=config.conservative_alpha,
        awbc_temperature=config.awbc_temperature,
        awbc_max_weight=config.awbc_max_weight,
        awbc_min_advantage=config.awbc_min_advantage,
        awbc_max_action_delta_norm=config.awbc_max_action_delta_norm,
    )


def _actor_loss_mode_id(mode: str) -> int:
    if mode == "td3":
        return ACTOR_LOSS_MODE_TD3
    if mode == "awbc":
        return ACTOR_LOSS_MODE_AWBC
    raise ValueError(f"Unsupported actor_loss_mode={mode!r}")


def actor_loss_mode_name(mode: int) -> str:
    if mode == ACTOR_LOSS_MODE_TD3:
        return "td3"
    if mode == ACTOR_LOSS_MODE_AWBC:
        return "awbc"
    raise ValueError(f"Unsupported actor_loss_mode id={mode!r}")


def _critic_loss_mode_id(mode: str) -> int:
    if mode == "td3":
        return CRITIC_LOSS_MODE_TD3
    if mode == "cql":
        return CRITIC_LOSS_MODE_CQL
    if mode == "calql":
        return CRITIC_LOSS_MODE_CALQL
    raise ValueError(f"Unsupported critic_loss_mode={mode!r}")


def critic_loss_mode_name(mode: int) -> str:
    if mode == CRITIC_LOSS_MODE_TD3:
        return "td3"
    if mode == CRITIC_LOSS_MODE_CQL:
        return "cql"
    if mode == CRITIC_LOSS_MODE_CALQL:
        return "calql"
    raise ValueError(f"Unsupported critic_loss_mode id={mode!r}")


def _critic_target_action_mode_id(mode: str) -> int:
    if mode == "target_actor":
        return CRITIC_TARGET_ACTION_MODE_TARGET_ACTOR
    if mode == "reference_action":
        return CRITIC_TARGET_ACTION_MODE_REFERENCE_ACTION
    raise ValueError(f"Unsupported critic_target_action_mode={mode!r}")


def critic_target_action_mode_name(mode: int) -> str:
    if mode == CRITIC_TARGET_ACTION_MODE_TARGET_ACTOR:
        return "target_actor"
    if mode == CRITIC_TARGET_ACTION_MODE_REFERENCE_ACTION:
        return "reference_action"
    raise ValueError(f"Unsupported critic_target_action_mode id={mode!r}")


def single_action_conservative_penalty(
    data_q: at.Float[at.Array, " b"],
    conservative_q: at.Float[at.Array, " b"],
    *,
    reference_value: at.Float[at.Array, " b"],
    critic_loss_mode: int,
) -> at.Float[at.Array, ""]:
    conservative_q_for_penalty = jnp.where(
        critic_loss_mode == CRITIC_LOSS_MODE_CALQL,
        jnp.maximum(conservative_q, reference_value),
        conservative_q,
    )
    return jnp.mean(jax.nn.softplus(conservative_q_for_penalty - data_q))


def should_publish_actor(step: int, *, actor_updated: bool, actor_publish_interval: int) -> bool:
    if not actor_updated or actor_publish_interval <= 0:
        return False
    return step % actor_publish_interval == 0


def dropout_reference_action_for_actor(
    reference_action: at.Float[at.Array, "b h a"],
    *,
    dropout: float,
    rng: at.KeyArrayLike,
) -> tuple[at.Float[at.Array, "b h a"], at.Float[at.Array, ""]]:
    """Drop whole reference chunks for actor conditioning, without changing the residual base action."""
    dropout_rate = jnp.asarray(jnp.clip(dropout, 0.0, 1.0), dtype=reference_action.dtype)
    keep = jax.random.bernoulli(
        rng,
        p=1.0 - dropout_rate,
        shape=(reference_action.shape[0], 1, 1),
    )
    conditioning_reference_action = jnp.where(keep, reference_action, jnp.zeros_like(reference_action))
    dropped_fraction = 1.0 - jnp.mean(keep.astype(reference_action.dtype))
    return conditioning_reference_action, dropped_fraction


def actor_params_for_inference(state: RLTTrainState) -> nnx.State:
    model = nnx.merge(state.model_def, state.params)
    return nnx.state(model.actor)


def critic_params_for_inference(state: RLTTrainState) -> nnx.State:
    model = nnx.merge(state.model_def, state.params)
    return nnx.state(model.critic)


def sync_target_params(state: RLTTrainState) -> RLTTrainState:
    model = nnx.merge(state.model_def, state.params)
    model.sync_targets()
    return dataclasses.replace(state, params=nnx.state(model))


def load_critic_params_from_state_dict(
    state: RLTTrainState,
    params: dict,
    *,
    reset_step: bool = True,
) -> RLTTrainState:
    """Load only critic networks from a saved RLT train-state params dict."""
    for key in ("critic", "target_critic"):
        if key not in params:
            raise KeyError(f"Missing {key!r} in source RLT params")
    model = nnx.merge(state.model_def, state.params)
    nnx.update(model.critic, _restore_nnx_state_keys(params["critic"]))
    nnx.update(model.target_critic, _restore_nnx_state_keys(params["target_critic"]))
    step = jnp.asarray(0, dtype=jnp.int32) if reset_step else state.step
    return dataclasses.replace(state, step=step, params=nnx.state(model))


def _restore_nnx_state_keys(value):
    if isinstance(value, dict):
        return {
            int(key) if isinstance(key, str) and key.isdigit() else key: _restore_nnx_state_keys(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_restore_nnx_state_keys(item) for item in value]
    return value


def train_step(
    state: RLTTrainState,
    batch: RLTReplayBatch,
    rng: at.KeyArrayLike,
) -> tuple[RLTTrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    target_rng, actor_rng, conservative_rng, reference_dropout_rng = jax.random.split(rng, 4)

    if state.critic_target_action_mode == CRITIC_TARGET_ACTION_MODE_REFERENCE_ACTION:
        next_q_min = model.target_critic.min_q(batch.next_x, batch.next_reference_action)
        target_q = rlt.td3_target(batch.reward_seq, batch.done, next_q_min, gamma=model.config.gamma)
    else:
        target_q = rlt.rlt_td3_target(
            model,
            batch.reward_seq,
            batch.done,
            batch.next_x,
            batch.next_reference_action,
            rng=target_rng,
            sample_target_actor=state.target_actor_noise,
        )
    conservative_action = jax.lax.stop_gradient(model.actor(batch.x, batch.reference_action, rng=conservative_rng, sample=True))

    def critic_loss_fn(critic: rlt.RLTTwinCritic):
        q1, q2 = critic(batch.x, batch.action)
        td_loss = rlt.critic_loss(q1, q2, target_q)
        conservative_q1, conservative_q2 = critic(batch.x, conservative_action)
        reference_value = jax.lax.stop_gradient(batch.reference_value)
        conservative_penalty = 0.5 * jnp.mean(
            single_action_conservative_penalty(
                q1,
                conservative_q1,
                reference_value=reference_value,
                critic_loss_mode=state.critic_loss_mode,
            )
            + single_action_conservative_penalty(
                q2,
                conservative_q2,
                reference_value=reference_value,
                critic_loss_mode=state.critic_loss_mode,
            )
        )
        conservative_enabled = (state.critic_loss_mode != CRITIC_LOSS_MODE_TD3) & (state.conservative_alpha > 0.0)
        conservative_weight = jnp.where(conservative_enabled, state.conservative_alpha, 0.0)
        loss = td_loss + conservative_weight * conservative_penalty
        reference_q = jnp.minimum(critic.q1(batch.x, batch.reference_action), critic.q2(batch.x, batch.reference_action))
        return loss, {
            "q1_mean": jnp.mean(q1),
            "q2_mean": jnp.mean(q2),
            "target_q_mean": jnp.mean(target_q),
            "q1_loss": jnp.mean(jnp.square(q1 - target_q)),
            "q2_loss": jnp.mean(jnp.square(q2 - target_q)),
            "td_loss": td_loss,
            "conservative_penalty": conservative_penalty,
            "conservative_q_mean": 0.5 * (jnp.mean(conservative_q1) + jnp.mean(conservative_q2)),
            "data_q_mean": 0.5 * (jnp.mean(q1) + jnp.mean(q2)),
            "reference_value_mean": jnp.mean(reference_value),
            "floor_violation_rate": jnp.mean((reference_q < reference_value).astype(jnp.float32)),
        }

    (critic_loss_value, critic_info), critic_grads = nnx.value_and_grad(critic_loss_fn, has_aux=True)(model.critic)
    critic_params = nnx.state(model.critic)
    critic_updates, critic_opt_state = state.critic_tx.update(
        critic_grads,
        state.critic_opt_state,
        critic_params,
    )
    nnx.update(model.critic, optax.apply_updates(critic_params, critic_updates))
    model.soft_update_target_critic()

    next_step = int(state.step) + 1
    actor_updated = next_step % state.policy_delay == 0
    actor_loss_value = jnp.asarray(0.0, dtype=critic_loss_value.dtype)
    actor_q_value = jnp.asarray(0.0, dtype=critic_loss_value.dtype)
    reference_q_value = jnp.asarray(0.0, dtype=critic_loss_value.dtype)
    q_advantage = jnp.asarray(0.0, dtype=critic_loss_value.dtype)
    actor_delta_norm = jnp.asarray(0.0, dtype=critic_loss_value.dtype)
    awbc_keep_fraction = jnp.asarray(0.0, dtype=critic_loss_value.dtype)
    awbc_weight_mean = jnp.asarray(0.0, dtype=critic_loss_value.dtype)
    awbc_advantage_mean = jnp.asarray(0.0, dtype=critic_loss_value.dtype)
    awbc_data_delta_norm_mean = jnp.asarray(0.0, dtype=critic_loss_value.dtype)
    reference_dropout_fraction = jnp.asarray(0.0, dtype=critic_loss_value.dtype)
    actor_opt_state = state.actor_opt_state
    if actor_updated:
        conditioning_reference_action, reference_dropout_fraction = dropout_reference_action_for_actor(
            batch.reference_action,
            dropout=model.config.reference_dropout,
            rng=reference_dropout_rng,
        )

        def actor_loss_fn(actor: rlt.RLTActor):
            action = actor(
                batch.x,
                batch.reference_action,
                conditioning_reference_action=conditioning_reference_action,
                rng=actor_rng,
                sample=True,
            )
            q1_for_actor = model.critic.q1(batch.x, action)
            q1_for_reference = model.critic.q1(batch.x, batch.reference_action)
            if state.actor_loss_mode == ACTOR_LOSS_MODE_TD3:
                loss = rlt.actor_loss(
                    q1_for_actor,
                    action,
                    batch.reference_action,
                    beta=model.config.beta,
                )
                awbc_info = {
                    "awbc_keep_fraction": jnp.asarray(0.0, dtype=loss.dtype),
                    "awbc_weight_mean": jnp.asarray(0.0, dtype=loss.dtype),
                    "awbc_advantage_mean": jnp.asarray(0.0, dtype=loss.dtype),
                    "awbc_data_delta_norm_mean": jnp.asarray(0.0, dtype=loss.dtype),
                }
            else:
                q1_for_data = model.critic.q1(batch.x, batch.action)
                data_advantage = jax.lax.stop_gradient(q1_for_data - q1_for_reference)
                loss, awbc_info = rlt.awbc_actor_loss(
                    action,
                    batch.action,
                    data_advantage,
                    batch.episode_success,
                    temperature=state.awbc_temperature,
                    max_weight=state.awbc_max_weight,
                    min_advantage=state.awbc_min_advantage,
                    max_action_delta_norm=state.awbc_max_action_delta_norm,
                    data_reference_action=batch.reference_action,
                )
            delta = action - batch.reference_action
            actor_q_mean = jnp.mean(q1_for_actor)
            reference_q_mean = jnp.mean(q1_for_reference)
            return loss, {
                "actor_q_value": actor_q_mean,
                "reference_q_value": reference_q_mean,
                "q_advantage": actor_q_mean - reference_q_mean,
                "actor_delta_norm": jnp.mean(jnp.linalg.norm(delta.reshape(delta.shape[0], -1), axis=-1)),
                **awbc_info,
            }

        (actor_loss_value, actor_info), actor_grads = nnx.value_and_grad(actor_loss_fn, has_aux=True)(model.actor)
        actor_params = nnx.state(model.actor)
        actor_updates, actor_opt_state = state.actor_tx.update(
            actor_grads,
            state.actor_opt_state,
            actor_params,
        )
        nnx.update(model.actor, optax.apply_updates(actor_params, actor_updates))
        model.soft_update_target_actor()
        actor_q_value = actor_info["actor_q_value"]
        reference_q_value = actor_info["reference_q_value"]
        q_advantage = actor_info["q_advantage"]
        actor_delta_norm = actor_info["actor_delta_norm"]
        awbc_keep_fraction = actor_info["awbc_keep_fraction"]
        awbc_weight_mean = actor_info["awbc_weight_mean"]
        awbc_advantage_mean = actor_info["awbc_advantage_mean"]
        awbc_data_delta_norm_mean = actor_info["awbc_data_delta_norm_mean"]

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
        "critic_td_loss": critic_info["td_loss"],
        "critic_conservative_penalty": critic_info["conservative_penalty"],
        "critic_conservative_q_mean": critic_info["conservative_q_mean"],
        "critic_data_q_mean": critic_info["data_q_mean"],
        "critic_reference_value_mean": critic_info["reference_value_mean"],
        "critic_floor_violation_rate": critic_info["floor_violation_rate"],
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
        "reference_dropout_rate": jnp.asarray(model.config.reference_dropout, dtype=critic_loss_value.dtype),
        "reference_dropout_fraction": reference_dropout_fraction,
        "critic_target_action_mode": jnp.asarray(state.critic_target_action_mode, dtype=jnp.int32),
        "actor_loss_mode": jnp.asarray(state.actor_loss_mode, dtype=jnp.int32),
        "critic_loss_mode": jnp.asarray(state.critic_loss_mode, dtype=jnp.int32),
        "conservative_alpha": jnp.asarray(state.conservative_alpha, dtype=critic_loss_value.dtype),
        "awbc_keep_fraction": awbc_keep_fraction,
        "awbc_weight_mean": awbc_weight_mean,
        "awbc_advantage_mean": awbc_advantage_mean,
        "awbc_data_delta_norm_mean": awbc_data_delta_norm_mean,
    }
    return new_state, info
