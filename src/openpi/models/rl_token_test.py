from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models import pi0_config
from openpi.models import rl_token


def test_rl_token_autoencoder_shapes():
    config = rl_token.RLTokenConfig(
        hidden_dim=8,
        encoder_layers=2,
        decoder_layers=1,
        num_heads=2,
        mlp_dim=16,
        max_prefix_len=5,
    )
    model = rl_token.RLTokenAutoencoder(config, rngs=nnx.Rngs(0))
    h_vla = jnp.ones((2, 4, 8), dtype=jnp.bfloat16)
    prefix_mask = jnp.array([[True, True, True, False], [True, True, False, False]])

    z_rl, h_hat = model(h_vla, prefix_mask)
    loss = model.compute_loss(h_vla, prefix_mask)

    assert z_rl.shape == (2, 8)
    assert h_hat.shape == h_vla.shape
    assert loss.shape == (2,)
    assert jnp.all(jnp.isfinite(loss))


def test_pi0_rl_token_only_loss_shape():
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=4,
        action_horizon=3,
        max_token_len=5,
        rl_token=rl_token.RLTokenConfig(
            hidden_dim=64,
            encoder_layers=1,
            decoder_layers=1,
            num_heads=8,
            mlp_dim=128,
            max_prefix_len=1029,
        ),
        rl_token_only=True,
    )
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=2)
    actions = jnp.ones((2, 3, 4), dtype=jnp.float32)

    loss = model.compute_loss(jax.random.key(1), obs, actions, train=False)

    assert loss.shape == (2, 3)
    assert jnp.all(jnp.isfinite(loss))
