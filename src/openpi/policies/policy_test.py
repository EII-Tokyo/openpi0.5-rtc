import os

os.environ.setdefault("USE_TF", "0")

from openpi_client import action_chunk_broker
import jax
import jax.numpy as jnp
import numpy as np

from openpi.policies import policy as _policy
import pytest

from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi05_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi05_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)


class _FakeRlTokenAutoencoder:
    def encode(self, prefix_out, prefix_mask):
        return jnp.mean(prefix_out, axis=1)


class _FakeRlTokenModel:
    rl_token_autoencoder = _FakeRlTokenAutoencoder()

    def __init__(self):
        self.embed_prefix_hidden_calls = 0
        self.sample_actions_calls = 0
        self.sample_actions_with_prefix_hidden_calls = 0
        self.sample_actions_with_rl_token_calls = 0

    def sample_actions(self, rng, observation, **kwargs):
        self.sample_actions_calls += 1
        prefix_out = jnp.ones((1, 3, 4), dtype=jnp.float32)
        prefix_mask = jnp.ones((1, 3), dtype=bool)
        actions = jnp.ones((1, 2, 3), dtype=jnp.float32)
        if kwargs.get("return_prefix_hidden"):
            return actions, (prefix_out, prefix_mask)
        return actions

    def sample_actions_with_prefix_hidden(self, rng, observation, **kwargs):
        self.sample_actions_with_prefix_hidden_calls += 1
        return self.sample_actions(rng, observation, return_prefix_hidden=True, **kwargs)

    def sample_actions_with_rl_token(self, rng, observation, **kwargs):
        self.sample_actions_with_rl_token_calls += 1
        actions = jnp.ones((1, 2, 3), dtype=jnp.float32)
        z_rl = jnp.full((1, 4), 2.0, dtype=jnp.float32)
        return actions, z_rl

    def guided_inference(self, rng, prev_action, observation, **kwargs):
        return self.sample_actions(rng, observation, **kwargs)

    def guided_inference_with_prefix_hidden(self, rng, prev_action, observation, **kwargs):
        return self.sample_actions(rng, observation, return_prefix_hidden=True, **kwargs)

    def embed_prefix_hidden(self, observation, *, drop_language=False):
        self.embed_prefix_hidden_calls += 1
        return jnp.zeros((1, 3, 4), dtype=jnp.float32), jnp.ones((1, 3), dtype=bool)


def test_infer_reuses_prefix_hidden_for_rl_token():
    model = _FakeRlTokenModel()
    policy = _policy.Policy.__new__(_policy.Policy)
    policy._model = model
    policy._input_transform = lambda x: x
    policy._output_transform = lambda x: x
    policy._sample_kwargs = {}
    policy._metadata = {}
    policy._is_pytorch_model = False
    policy._sample_actions = model.sample_actions
    policy._guided_inference = model.guided_inference
    policy._sample_actions_with_prefix_hidden = model.sample_actions_with_prefix_hidden
    policy._guided_inference_with_prefix_hidden = model.guided_inference_with_prefix_hidden
    policy._sample_actions_with_rl_token = None
    policy._guided_inference_with_rl_token = None
    policy._rng = jax.random.key(0)

    result = policy.infer({
        "state": np.zeros((3,), dtype=np.float32),
        "image": {"cam": np.zeros((2, 2, 3), dtype=np.float32)},
        "image_mask": {"cam": np.array(True)},
    })

    assert model.sample_actions_calls == 1
    assert model.sample_actions_with_prefix_hidden_calls == 1
    assert model.embed_prefix_hidden_calls == 0
    assert result["actions"].shape == (2, 3)
    assert result["z_rl"].shape == (4,)


def test_infer_prefers_direct_sample_actions_with_rl_token():
    model = _FakeRlTokenModel()
    policy = _policy.Policy.__new__(_policy.Policy)
    policy._model = model
    policy._input_transform = lambda x: x
    policy._output_transform = lambda x: x
    policy._sample_kwargs = {}
    policy._metadata = {}
    policy._is_pytorch_model = False
    policy._sample_actions = model.sample_actions
    policy._guided_inference = model.guided_inference
    policy._sample_actions_with_prefix_hidden = model.sample_actions_with_prefix_hidden
    policy._guided_inference_with_prefix_hidden = model.guided_inference_with_prefix_hidden
    policy._sample_actions_with_rl_token = model.sample_actions_with_rl_token
    policy._guided_inference_with_rl_token = None
    policy._rng = jax.random.key(0)

    result = policy.infer({
        "state": np.zeros((3,), dtype=np.float32),
        "image": {"cam": np.zeros((2, 2, 3), dtype=np.float32)},
        "image_mask": {"cam": np.array(True)},
    })

    assert model.sample_actions_with_rl_token_calls == 1
    assert model.sample_actions_with_prefix_hidden_calls == 0
    assert model.embed_prefix_hidden_calls == 0
    assert result["actions"].shape == (2, 3)
    assert np.all(result["z_rl"] == 2.0)
