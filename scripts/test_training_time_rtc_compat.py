#!/usr/bin/env python3
"""Offline compatibility tests for sync / inference-time RTC / training-time RTC."""

from __future__ import annotations

import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch

from openpi.models import pi0_config
from openpi.models.pi0 import make_attn_mask
from openpi.serving.policy import Policy
from openpi.shared import nnx_utils


def _old_compute_loss(model, rng, observation, actions):
    _, noise_rng, time_rng = jax.random.split(rng, 3)
    batch_shape = actions.shape[:-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    time_value = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_expanded = time_value[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(observation, x_t, time_value)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (_, suffix_out), _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
    )
    v_t = model.action_out_proj(suffix_out[:, -model.action_horizon :])
    return jnp.mean(jnp.square(v_t - u_t), axis=-1)


def _dummy_config(*, training_time_rtc: bool = False, rtc_max_delay: int = 10):
    return pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=32,
        action_horizon=50,
        max_token_len=16,
        image_resolution=(224, 224),
        training_time_rtc=training_time_rtc,
        rtc_max_delay=rtc_max_delay,
    )


def _assert_close(name: str, got, expected, atol: float = 0.0):
    diff = float(jnp.max(jnp.abs(jnp.asarray(got) - jnp.asarray(expected))))
    print(f"{name}: max_abs_diff={diff}")
    if diff > atol:
        raise AssertionError(f"{name} diff {diff} > {atol}")


def test_model_api():
    cfg = _dummy_config(training_time_rtc=False)
    model = cfg.create(jax.random.key(0))
    obs = cfg.fake_obs(batch_size=1)
    noise = jax.random.normal(jax.random.key(1), (1, 50, 32))

    normal_new = model.sample_action_chunk(jax.random.key(2), obs, denoising_steps=2, noise=noise)
    normal_again = model.sample_action_chunk(jax.random.key(2), obs, denoising_steps=2, noise=noise)
    _assert_close("sample_action_chunk repeat", normal_again, normal_new)

    action_prefix = jax.random.normal(jax.random.key(3), (1, 50, 32))
    prefixed = model.sample_action_chunk_with_training_time_rtc(
        jax.random.key(2), obs, action_prefix=action_prefix, handoff_delay_steps=10, denoising_steps=2, noise=noise
    )
    _assert_close("training_time_rtc prefix first 10", prefixed[:, :10], action_prefix[:, :10])

    inference_jit = nnx_utils.module_jit(model.sample_action_chunk_with_inference_time_rtc)
    guided = inference_jit(jax.random.key(4), action_prefix, obs, denoising_steps=2, replan_start_step=25, handoff_delay_steps=10, guidance_scale=8.0)
    print(f"sample_action_chunk_with_inference_time_rtc jit shape={guided.shape}")


def test_loss_compatibility():
    actions = jax.random.normal(jax.random.key(5), (1, 50, 32))

    delay0_cfg = _dummy_config(training_time_rtc=True, rtc_max_delay=0)
    delay0_model = delay0_cfg.create(jax.random.key(0))
    obs = delay0_cfg.fake_obs(batch_size=1)
    delay0_loss = delay0_model.compute_loss(jax.random.key(6), obs, actions, train=True)
    ordinary_loss = _old_compute_loss(delay0_model, jax.random.key(6), obs, actions)
    _assert_close("compute_loss training_time_rtc delay=0 vs ordinary", delay0_loss, ordinary_loss)

    delayed_cfg = _dummy_config(training_time_rtc=True, rtc_max_delay=10)
    delayed_model = delayed_cfg.create(jax.random.key(0))
    obs = delayed_cfg.fake_obs(batch_size=1)
    rng = None
    delay = None
    for seed in range(100):
        candidate = jax.random.key(seed)
        sampled_delay = int(jax.random.randint(jax.random.fold_in(candidate, 0), (), 0, 11))
        if sampled_delay > 0:
            rng = candidate
            delay = sampled_delay
            break
    assert rng is not None and delay is not None
    delayed_loss = delayed_model.compute_loss(rng, obs, actions, train=True)
    prefix_max = float(jnp.max(jnp.abs(delayed_loss[:, :delay])))
    suffix_count = delayed_loss[:, delay:].size
    print(f"compute_loss delay>0 sampled_delay={delay} prefix_loss_max={prefix_max} suffix_count={suffix_count}")
    if prefix_max != 0.0:
        raise AssertionError("prefix loss was not masked to zero")


def _policy_obs():
    return {
        "image": {"cam_high": np.zeros((2, 2, 3), dtype=np.float32)},
        "image_mask": {"cam_high": np.array(True)},
        "state": np.zeros(32, dtype=np.float32),
    }


class DispatchModel:
    def __init__(self):
        self.calls = []

    def to(self, device):
        return self

    def eval(self):
        return self

    def sample_action_chunk(self, device, observation, **kwargs):
        self.calls.append("sample_action_chunk")
        return torch.full((1, 50, 32), 1.0)

    def sample_action_chunk_with_inference_time_rtc(self, device, prev_action_chunk, observation, **kwargs):
        self.calls.append("sample_action_chunk_with_inference_time_rtc")
        return torch.full((1, 50, 32), 2.0)

    def sample_action_chunk_with_training_time_rtc(self, device, observation, *, action_prefix, handoff_delay_steps, **kwargs):
        self.calls.append("sample_action_chunk_with_training_time_rtc")
        out = torch.full((1, 50, 32), 7.0)
        out[:, :handoff_delay_steps] = action_prefix[:, :handoff_delay_steps]
        return out



def test_policy_dispatch():
    model = DispatchModel()
    policy = Policy(model, is_pytorch=True, transforms=[], output_transforms=[])
    obs = _policy_obs()
    prev = np.zeros((50, 32), dtype=np.float32)
    prefix = np.arange(50 * 32, dtype=np.float32).reshape(50, 32)

    policy.infer(obs, chunking_mode="sync")
    policy.infer(obs, chunking_mode="inference_time", prev_action=prev)
    training = policy.infer(obs, chunking_mode="training_time", action_prefix=prefix, handoff_delay_steps=10)["actions"]

    expected = ["sample_action_chunk", "sample_action_chunk_with_inference_time_rtc", "sample_action_chunk_with_training_time_rtc"]
    print(f"policy calls={model.calls}")
    if model.calls != expected:
        raise AssertionError(f"unexpected policy calls: {model.calls}")
    if not np.array_equal(training[:10], prefix[:10]):
        raise AssertionError("training-time policy prefix was modified")


def test_offline_modes():
    cfg = _dummy_config(training_time_rtc=True, rtc_max_delay=10)
    model = cfg.create(jax.random.key(0))
    obs = cfg.fake_obs(batch_size=1)
    noise = jnp.ones((1, 50, 32), dtype=jnp.float32)
    prev = jax.random.normal(jax.random.key(8), (1, 50, 32))
    prefix = jax.random.normal(jax.random.key(9), (1, 50, 32))

    timings = {}
    start = time.perf_counter()
    sync = model.sample_action_chunk(jax.random.key(10), obs, denoising_steps=2, noise=noise)
    timings["sync_ms"] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    inference = model.sample_action_chunk_with_inference_time_rtc(jax.random.key(11), prev, obs, denoising_steps=2, replan_start_step=25, handoff_delay_steps=10, guidance_scale=8.0)
    timings["inference_time_ms"] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    training = model.sample_action_chunk_with_training_time_rtc(
        jax.random.key(12), obs, action_prefix=prefix, handoff_delay_steps=10, denoising_steps=2, noise=noise
    )
    timings["training_time_ms"] = (time.perf_counter() - start) * 1000

    print(f"offline shapes sync={sync.shape} inference_time={inference.shape} training_time={training.shape}")
    print("offline timings", {k: round(v, 2) for k, v in timings.items()})
    _assert_close("offline training-time prefix", training[:, :10], prefix[:, :10])


def test_dummy_backward_smoke():
    cfg = _dummy_config(training_time_rtc=True, rtc_max_delay=10)
    model = cfg.create(jax.random.key(0))
    obs = cfg.fake_obs(batch_size=1)
    actions = jax.random.normal(jax.random.key(13), (1, 50, 32))

    def loss_fn(m):
        return jnp.mean(m.compute_loss(jax.random.key(14), obs, actions, train=True))

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    leaves = jax.tree.leaves(grads.to_pure_dict())
    finite = all(bool(jnp.all(jnp.isfinite(x))) for x in leaves if hasattr(x, "shape"))
    print(f"dummy backward loss={float(loss)} grad_leaves={len(leaves)} finite={finite}")
    if not finite:
        raise AssertionError("non-finite gradient in dummy backward smoke test")


def main():
    test_model_api()
    test_loss_compatibility()
    test_policy_dispatch()
    test_offline_modes()
    test_dummy_backward_smoke()
    print("training_time_rtc_compat_tests_ok")


if __name__ == "__main__":
    main()
