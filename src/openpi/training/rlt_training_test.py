import jax
import jax.numpy as jnp

from openpi.models import rlt
from openpi.training import rlt_training


def _make_config() -> rlt_training.RLTTrainingConfig:
    return rlt_training.RLTTrainingConfig(
        model=rlt.RLTConfig(
            z_dim=8,
            proprio_dim=4,
            action_horizon=5,
            action_dim=3,
            hidden_dim=16,
            num_layers=2,
            beta=2.0,
            tau=0.1,
        ),
        actor_lr=1e-3,
        critic_lr=1e-3,
        policy_delay=2,
        actor_publish_interval=4,
    )


def _flatten_actor_params(state: rlt_training.RLTTrainState) -> jnp.ndarray:
    actor_params = rlt_training.actor_params_for_inference(state).flat_state()
    return jnp.concatenate([jnp.ravel(value.value) for value in actor_params.values()])


def _make_awbc_config() -> rlt_training.RLTTrainingConfig:
    return rlt_training.RLTTrainingConfig(
        model=rlt.RLTConfig(
            z_dim=8,
            proprio_dim=4,
            action_horizon=5,
            action_dim=3,
            hidden_dim=16,
            num_layers=2,
            beta=2.0,
            tau=0.1,
        ),
        actor_lr=1e-3,
        critic_lr=1e-3,
        policy_delay=1,
        actor_publish_interval=1,
        actor_loss_mode="awbc",
        awbc_temperature=0.2,
        awbc_min_advantage=-1.0,
        awbc_max_weight=5.0,
    )


def _make_batch() -> rlt_training.RLTReplayBatch:
    return rlt_training.make_replay_batch(
        z_rl=jnp.ones((4, 8), dtype=jnp.float32),
        proprio=jnp.ones((4, 4), dtype=jnp.float32) * 0.5,
        action=jnp.ones((4, 5, 3), dtype=jnp.float32) * 0.1,
        reference_action=jnp.zeros((4, 5, 3), dtype=jnp.float32),
        reward_seq=jnp.array(
            [
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
            ],
            dtype=jnp.float32,
        ),
        next_z_rl=jnp.ones((4, 8), dtype=jnp.float32) * 0.8,
        next_proprio=jnp.ones((4, 4), dtype=jnp.float32) * 0.25,
        next_reference_action=jnp.zeros((4, 5, 3), dtype=jnp.float32),
        done=jnp.array([True, False, True, False]),
        episode_success=jnp.array([True, False, True, False]),
    )


def test_rlt_train_step_delays_actor_and_publish():
    state = rlt_training.init_train_state(_make_config(), jax.random.key(0))
    batch = _make_batch()

    state, info1 = rlt_training.train_step(state, batch, jax.random.key(1))
    state, info2 = rlt_training.train_step(state, batch, jax.random.key(2))
    state, info3 = rlt_training.train_step(state, batch, jax.random.key(3))
    state, info4 = rlt_training.train_step(state, batch, jax.random.key(4))

    assert int(state.step) == 4
    assert not bool(info1["actor_updated"])
    assert bool(info2["actor_updated"])
    assert not bool(info3["actor_updated"])
    assert bool(info4["actor_updated"])
    assert not bool(info2["publish_actor"])
    assert bool(info4["publish_actor"])
    assert jnp.isfinite(info4["critic_loss"])
    assert jnp.isfinite(info4["actor_loss"])
    assert jnp.isfinite(info4["reference_q_value"])
    assert jnp.isfinite(info4["q_advantage"])
    assert jnp.allclose(info4["q_advantage"], info4["actor_q_value"] - info4["reference_q_value"])


def test_rlt_train_step_actor_update_samples_policy_actions():
    config = rlt_training.RLTTrainingConfig(
        model=rlt.RLTConfig(
            z_dim=8,
            proprio_dim=4,
            action_horizon=5,
            action_dim=3,
            hidden_dim=16,
            num_layers=2,
            beta=2.0,
            fixed_std=0.5,
        ),
        actor_lr=1e-3,
        critic_lr=1e-3,
        policy_delay=1,
        actor_publish_interval=1,
        target_actor_noise=False,
    )
    state_a = rlt_training.init_train_state(config, jax.random.key(0))
    state_b = rlt_training.init_train_state(config, jax.random.key(0))
    batch = _make_batch()

    state_a, info_a = rlt_training.train_step(state_a, batch, jax.random.key(1))
    state_b, info_b = rlt_training.train_step(state_b, batch, jax.random.key(2))

    assert bool(info_a["actor_updated"])
    assert bool(info_b["actor_updated"])
    assert not jnp.allclose(_flatten_actor_params(state_a), _flatten_actor_params(state_b))


def test_rlt_training_config_samples_target_actor_by_default():
    config = rlt_training.RLTTrainingConfig()

    assert config.target_actor_noise is True


def test_rlt_train_step_awbc_reports_filter_metrics():
    state = rlt_training.init_train_state(_make_awbc_config(), jax.random.key(0))
    batch = _make_batch()

    state, info = rlt_training.train_step(state, batch, jax.random.key(1))

    assert int(state.step) == 1
    assert bool(info["actor_updated"])
    assert jnp.isfinite(info["actor_loss"])
    assert info["actor_loss_mode"] == rlt_training.ACTOR_LOSS_MODE_AWBC
    assert 0.0 <= float(info["awbc_keep_fraction"]) <= 1.0
    assert float(info["awbc_weight_mean"]) >= 0.0


def test_actor_params_for_inference_returns_online_actor_only():
    state = rlt_training.init_train_state(_make_config(), jax.random.key(0))
    actor_params = rlt_training.actor_params_for_inference(state)

    flat_keys = actor_params.flat_state().keys()
    assert flat_keys
    assert all("target" not in "/".join(str(part) for part in key) for key in flat_keys)


def test_critic_params_for_inference_returns_online_critic_only():
    state = rlt_training.init_train_state(_make_config(), jax.random.key(0))
    critic_params = rlt_training.critic_params_for_inference(state)

    flat_keys = critic_params.flat_state().keys()
    assert flat_keys
    assert all("target" not in "/".join(str(part) for part in key) for key in flat_keys)


def test_sync_target_params_hard_copies_online_networks():
    state = rlt_training.init_train_state(_make_config(), jax.random.key(0))
    model = rlt_training.nnx.merge(state.model_def, state.params)
    rlt_training.nnx.update(
        model.critic,
        jax.tree.map(lambda value: value + 1.0, rlt_training.nnx.state(model.critic)),
    )
    state = rlt_training.dataclasses.replace(state, params=rlt_training.nnx.state(model))

    state = rlt_training.sync_target_params(state)
    model = rlt_training.nnx.merge(state.model_def, state.params)

    critic_state = rlt_training.nnx.state(model.critic).flat_state()
    target_critic_state = rlt_training.nnx.state(model.target_critic).flat_state()
    for key in critic_state:
        assert jnp.allclose(critic_state[key].value, target_critic_state[key].value)
