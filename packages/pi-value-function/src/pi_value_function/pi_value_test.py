from flax import nnx
import jax 
import pytest

from pi_value_function.config import PiValueConfig
from openpi.shared import nnx_utils
from openpi.shared import download

def test_pi_value_model():
    key = jax.random.key(0)
    config = PiValueConfig()
    model = config.create(key)

    batch_size = 2
    obs = config.fake_obs(batch_size)
    # Create fake returns in the range [value_min, value_max]
    returns = jax.random.uniform(key, (batch_size,), minval=config.value_min, maxval=config.value_max)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, returns)
    assert loss.shape == (batch_size,)

    value = nnx_utils.module_jit(model.predict_value)(key, obs)
    assert value.shape == (batch_size,)



