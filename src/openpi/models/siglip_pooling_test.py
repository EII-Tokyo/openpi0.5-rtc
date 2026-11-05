"""Spatial pooling keeps the visual LLM budget fixed without pooling time."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import siglip


@pytest.mark.parametrize("height,width", [(16,16), (32,32), (64,64), (24,40), (17,19)])
def test_spatial_mean_and_gradient(height, width):
    data = np.arange(2*height*width*3, dtype=np.float32).reshape(2,height*width,3)
    actual = siglip.spatial_average_pool_tokens(jnp.asarray(data), (height,width), 256)
    grid = data.reshape(2,height,width,3)
    expected = np.empty((2,16,16,3), dtype=np.float32)
    for row in range(16):
        for col in range(16):
            expected[:,row,col] = grid[:,math.floor(row*height/16):math.ceil((row+1)*height/16),
                math.floor(col*width/16):math.ceil((col+1)*width/16)].mean(axis=(1,2))
    np.testing.assert_allclose(actual, expected.reshape(2,256,3), rtol=1e-5)
    grad = jax.grad(lambda x: siglip.spatial_average_pool_tokens(x,(height,width),256).sum())(jnp.asarray(data))
    assert grad.shape == data.shape and np.isfinite(grad).all()


@pytest.mark.parametrize("grid,tokens", [((8,8),256), ((16,16),255), ((16,16),0)])
def test_invalid_pooling_rejected(grid, tokens):
    with pytest.raises(ValueError):
        siglip.spatial_average_pool_tokens(jnp.ones((1,grid[0]*grid[1],3)), grid, tokens)


@pytest.mark.parametrize("resolution", [224,448])
def test_encoder_video_pool_preserves_projection_parameters(resolution):
    kwargs = dict(num_classes=64, patch_size=(14,14), width=32, depth=1,
                  num_heads=4, mlp_dim=64, pool_type="none")
    images = jnp.zeros((1,2,resolution,resolution,3))
    model = siglip._Module(**kwargs, image_tokens_per_image=256)
    variables = model.init(jax.random.key(0), images)
    actual, info = model.apply(variables, images)
    assert actual.shape == (1,256,64)
    assert info["spatially_pooled"].shape == (1,256,32)
    assert variables["params"]["head"]["kernel"].shape == (32,64)
    baseline = siglip._Module(**kwargs)
    unpooled, _ = baseline.apply(variables, images)
    assert unpooled.shape == (1,(resolution//14)**2,64)
    # The same parameter tree loads in both modes; no trainable pooling parameters.
    expected = siglip.spatial_average_pool_tokens(unpooled, (resolution//14,)*2, 256)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
