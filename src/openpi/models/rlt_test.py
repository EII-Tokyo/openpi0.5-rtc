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
