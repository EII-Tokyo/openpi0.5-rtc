"""Backend opt-in must preserve native training and parameter compatibility."""

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import cudnn_attention
from openpi.models import pi0_config
from openpi.models import siglip


def test_config_backend_is_opt_in():
    assert pi0_config.Pi0Config().vision_attention_backend == "native"
    with pytest.raises(ValueError, match="Unknown vision attention backend"):
        pi0_config.Pi0Config(vision_attention_backend="invalid")


@pytest.mark.parametrize("scan", [False, True])
def test_training_values_gradients_and_parameters_stay_native(scan):
    kwargs = {
        "num_classes": 16,
        "patch_size": (14, 14),
        "width": 32,
        "depth": 4,
        "num_heads": 4,
        "mlp_dim": 64,
        "pool_type": "none",
        "scan": scan,
    }
    images = jnp.arange(1 * 3 * 28 * 28 * 3, dtype=jnp.float32).reshape(1, 3, 28, 28, 3) / 1000
    native = siglip.Module(**kwargs, attention_backend="native")
    opted = siglip.Module(**kwargs, attention_backend="cudnn")
    params = native.init(jax.random.key(7), images, train=True)
    opted_params = opted.init(jax.random.key(7), images, train=True)
    assert jax.tree.structure(params) == jax.tree.structure(opted_params)
    for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(opted_params), strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(
        native.apply(params, images, train=True)[0], opted.apply(params, images, train=True)[0]
    )

    def grad(module):
        return jax.grad(lambda p: module.apply({"params": p}, images, train=True)[0].sum())(params["params"])

    for a, b in zip(jax.tree.leaves(grad(native)), jax.tree.leaves(grad(opted)), strict=True):
        np.testing.assert_array_equal(a, b)


def test_inference_routes_224_spatial_and_causal_temporal_without_cutoff(monkeypatch):
    calls = []

    def spy(query, key, value, mask=None, **kwargs):
        calls.append((query.shape[-3], None if mask is None else mask.shape))
        return nn.dot_product_attention(query, key, value, mask=mask, deterministic=True)

    monkeypatch.setattr(cudnn_attention, "dot_product_attention", spy)
    module = siglip.Module(
        num_classes=16,
        patch_size=(14, 14),
        width=32,
        depth=4,
        num_heads=4,
        mlp_dim=64,
        pool_type="none",
        attention_backend="cudnn",
    )
    images = jnp.zeros((1, 3, 224, 224, 3))
    params = module.init(jax.random.key(0), images, train=True)
    assert not calls
    module.apply(params, images, train=False, use_fused_attention=False)
    assert not calls
    module.apply(params, images, train=False)
    assert (256, None) in calls
    assert (3, (1, 1, 3, 3)) in calls


@pytest.mark.parametrize(("length", "masked"), [(256, False), (16, True)])
def test_cudnn_call_forwards_mask_and_preserves_query_prescaling(monkeypatch, length, masked):
    seen = {}

    def fake(q, k, v, **kwargs):
        seen.update(query=q, key=k, value=v, **kwargs)
        return v

    monkeypatch.setattr(jax.nn, "dot_product_attention", fake)
    q = jnp.ones((1, length, 2, 8), dtype=jnp.bfloat16)
    mask = jnp.tril(jnp.ones((length, length), dtype=bool))[None, None] if masked else None
    actual = cudnn_attention.dot_product_attention(q, q, q, mask=mask, deterministic=True)
    np.testing.assert_array_equal(actual, q)
    np.testing.assert_array_equal(seen["query"], q / jnp.sqrt(8).astype(q.dtype))
    assert seen["implementation"] == "cudnn"
    assert seen["scale"] == 1.0
    if masked:
        np.testing.assert_array_equal(seen["mask"], mask)
    else:
        assert seen["mask"] is None


def test_unsupported_attention_options_fail_visibly():
    q = jnp.ones((1, 2, 2, 8), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="disable dropout"):
        cudnn_attention.dot_product_attention(q, q, q, dropout_rate=0.1, deterministic=False)
    with pytest.raises(ValueError, match="probability intermediates"):
        cudnn_attention.dot_product_attention(q, q, q, module=object())
    with pytest.raises(ValueError, match="BF16 or FP16"):
        cudnn_attention.dot_product_attention(q.astype(jnp.float32), q, q)


@pytest.mark.parametrize("train", [False, True])
def test_pi0_loss_explicitly_disables_inference_fusion(train):
    from types import SimpleNamespace

    from openpi.models import model as model_lib
    from openpi.models import pi0

    calls = []

    class ReachedPrefixError(Exception):
        pass

    def prefix(obs, **kwargs):
        calls.append(kwargs)
        raise ReachedPrefixError

    fake = SimpleNamespace(training_time_rtc=False, embed_prefix=prefix)
    obs = model_lib.Observation(images={}, image_masks={}, state=jnp.zeros((1, 32)))
    with pytest.raises(ReachedPrefixError):
        pi0.Pi0.compute_loss(fake, jax.random.key(0), obs, jnp.zeros((1, 2, 32)), train=train)
    assert calls == [{"use_fused_attention": False}]
