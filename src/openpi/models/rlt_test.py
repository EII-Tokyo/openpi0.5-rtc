from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models import rlt


def test_rlt_actor_critic_shapes():
    config = rlt.RLTConfig(
        z_dim=8,
        proprio_dim=4,
        action_horizon=5,
        action_dim=3,
        hidden_dim=16,
        num_layers=2,
        max_delta=0.2,
    )
    model = rlt.RLTActorCritic(config, rngs=nnx.Rngs(0))
    z_rl = jnp.ones((2, 8), dtype=jnp.float32)
    proprio = jnp.ones((2, 4), dtype=jnp.float32)
    x = rlt.make_state(z_rl, proprio)
    reference_action = jnp.zeros((2, 5, 3), dtype=jnp.float32)

    action = model.actor(x, reference_action, rng=jax.random.key(1), sample=True, intervention_scale=0.5)
    q1, q2 = model.critic(x, action)

    assert x.shape == (2, 12)
    assert action.shape == (2, 5, 3)
    assert q1.shape == (2,)
    assert q2.shape == (2,)
    assert jnp.max(jnp.abs(action - reference_action)) <= config.max_delta * 0.5 + 1e-6


def test_target_networks_start_as_online_copies():
    config = rlt.RLTConfig(
        z_dim=8,
        proprio_dim=4,
        action_horizon=5,
        action_dim=3,
        hidden_dim=16,
        num_layers=2,
    )
    model = rlt.RLTActorCritic(config, rngs=nnx.Rngs(0))

    actor_state = nnx.state(model.actor).flat_state()
    target_actor_state = nnx.state(model.target_actor).flat_state()
    critic_state = nnx.state(model.critic).flat_state()
    target_critic_state = nnx.state(model.target_critic).flat_state()

    assert actor_state.keys() == target_actor_state.keys()
    assert critic_state.keys() == target_critic_state.keys()
    for key in actor_state:
        assert jnp.allclose(actor_state[key].value, target_actor_state[key].value)
    for key in critic_state:
        assert jnp.allclose(critic_state[key].value, target_critic_state[key].value)


def test_soft_update_targets_moves_toward_online_params():
    config = rlt.RLTConfig(
        z_dim=8,
        proprio_dim=4,
        action_horizon=5,
        action_dim=3,
        hidden_dim=16,
        num_layers=2,
    )
    model = rlt.RLTActorCritic(config, rngs=nnx.Rngs(0))
    original_target_state = nnx.state(model.target_actor)
    updated_actor_state = jax.tree.map(lambda value: value + 2.0, nnx.state(model.actor))
    nnx.update(model.actor, updated_actor_state)

    model.soft_update_targets(tau=0.25)

    online_state = nnx.state(model.actor).flat_state()
    target_state = nnx.state(model.target_actor).flat_state()
    original_state = original_target_state.flat_state()
    for key in target_state:
        expected = original_state[key].value * 0.75 + online_state[key].value * 0.25
        assert jnp.allclose(target_state[key].value, expected)


def test_td3_losses_are_finite():
    reward_seq = jnp.array([[0, 0, 1], [0, 0, 0]], dtype=jnp.float32)
    done = jnp.array([True, False])
    next_q_min = jnp.array([10.0, 2.0], dtype=jnp.float32)
    target = rlt.td3_target(reward_seq, done, next_q_min, gamma=0.9)

    q1 = jnp.array([0.5, 1.0], dtype=jnp.float32)
    q2 = jnp.array([0.25, 1.5], dtype=jnp.float32)
    action = jnp.ones((2, 3, 2), dtype=jnp.float32) * 0.2
    reference = jnp.zeros((2, 3, 2), dtype=jnp.float32)

    q_loss = rlt.critic_loss(q1, q2, target)
    pi_loss = rlt.actor_loss(q1, action, reference, beta=2.0)

    assert target.shape == (2,)
    assert jnp.allclose(target[0], 0.9**2)
    assert jnp.all(jnp.isfinite(target))
    assert jnp.isfinite(q_loss)
    assert jnp.isfinite(pi_loss)


def test_awbc_actor_loss_weights_high_advantage_success_samples():
    actor_action = jnp.array(
        [
            [[0.9]],
            [[0.2]],
            [[0.4]],
        ],
        dtype=jnp.float32,
    )
    data_action = jnp.array(
        [
            [[1.0]],
            [[1.0]],
            [[1.0]],
        ],
        dtype=jnp.float32,
    )
    advantage = jnp.array([0.4, 0.05, 0.8], dtype=jnp.float32)
    success = jnp.array([True, True, False])

    loss, info = rlt.awbc_actor_loss(
        actor_action,
        data_action,
        advantage,
        success,
        temperature=0.2,
        max_weight=10.0,
        min_advantage=0.1,
        max_action_delta_norm=2.0,
        data_reference_action=jnp.zeros_like(data_action),
    )

    assert jnp.isfinite(loss)
    assert int(info["awbc_kept_count"]) == 1
    assert jnp.allclose(info["awbc_keep_fraction"], 1 / 3)
    assert info["awbc_weight_mean"] > 1.0
    assert loss < rlt.awbc_actor_loss(
        actor_action + 0.5,
        data_action,
        advantage,
        success,
        temperature=0.2,
        max_weight=10.0,
        min_advantage=0.1,
        max_action_delta_norm=2.0,
        data_reference_action=jnp.zeros_like(data_action),
    )[0]


def test_rlt_td3_target_uses_target_network_shapes():
    config = rlt.RLTConfig(
        z_dim=8,
        proprio_dim=4,
        action_horizon=5,
        action_dim=3,
        hidden_dim=16,
        num_layers=2,
        gamma=0.9,
    )
    model = rlt.RLTActorCritic(config, rngs=nnx.Rngs(0))
    x_next = jnp.ones((2, 12), dtype=jnp.float32)
    next_reference = jnp.zeros((2, 5, 3), dtype=jnp.float32)
    reward_seq = jnp.array([[0, 0, 0, 0, 1], [0, 0, 0, 0, 0]], dtype=jnp.float32)
    done = jnp.array([True, False])

    target = rlt.rlt_td3_target(model, reward_seq, done, x_next, next_reference)

    assert target.shape == (2,)
    assert jnp.allclose(target[0], 0.9**4)
    assert jnp.all(jnp.isfinite(target))
