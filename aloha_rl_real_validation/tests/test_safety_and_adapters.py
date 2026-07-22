from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from aloha_rl_validation.action_adapter import (
    passthrough_absolute_action,
    rlinf_robotwin_delta_to_canonical_target,
)
from aloha_rl_validation.observation_adapter import adapt_real_observation_for_rlinf
from aloha_rl_validation.policy_loader import (
    inspect_checkpoint_schema,
    validate_strict_openpi_native_compatibility,
)
from aloha_rl_validation.safety_filter import SafetyConfig, SafetyFilter, assert_no_publish_allowed
from aloha_rl_validation.schema import CANONICAL_JOINT_NAMES


def test_nan_rejection():
    filt = SafetyFilter(SafetyConfig())
    action = np.zeros(14)
    action[3] = np.nan
    result = filt.check(action)
    assert not result.accepted
    assert any("NaN" in r for r in result.reasons)


def test_joint_limit_rejection():
    filt = SafetyFilter(SafetyConfig(joint_lower=np.full(14, -1.0), joint_upper=np.full(14, 1.0)))
    action = np.zeros(14)
    action[8] = 2.0
    result = filt.check(action)
    assert not result.accepted
    assert any("upper limit" in r for r in result.reasons)


def test_velocity_limit_rejection():
    os.environ["ALLOW_REAL_ACTUATION"] = "true"
    filt = SafetyFilter(
        SafetyConfig(
            allow_real_actuation=True,
            max_velocity=np.full(14, 0.1),
        )
    )
    assert filt.check(np.zeros(14), now_s=1.0).accepted
    result = filt.check(np.ones(14), now_s=1.01)
    assert not result.accepted
    assert any("velocity" in r for r in result.reasons)
    os.environ.pop("ALLOW_REAL_ACTUATION", None)


def test_first_action_continuity():
    filt = SafetyFilter(SafetyConfig(max_step_delta=np.full(14, 0.05)))
    current = np.zeros(14)
    action = np.zeros(14)
    action[1] = 0.2
    result = filt.check(action, current_qpos=current)
    assert not result.accepted
    assert any("single-step" in r for r in result.reasons)


def test_stale_observation_rejection():
    filt = SafetyFilter(SafetyConfig(max_image_age_s=0.1, max_joint_state_age_s=0.1))
    result = filt.check(np.zeros(14), image_age_s=0.2, joint_state_age_s=0.3)
    assert not result.accepted
    assert any("stale image" in r for r in result.reasons)
    assert any("stale joint state" in r for r in result.reasons)


def test_dry_run_cannot_publish():
    os.environ.pop("ALLOW_REAL_ACTUATION", None)
    with pytest.raises(PermissionError):
        assert_no_publish_allowed()


def test_joint_order_mapping():
    filt = SafetyFilter(SafetyConfig())
    assert filt.validate_joint_names(CANONICAL_JOINT_NAMES) == []
    assert filt.validate_joint_names(tuple(reversed(CANONICAL_JOINT_NAMES)))


def test_absolute_action_conversion():
    raw = np.arange(14, dtype=np.float32)
    out = passthrough_absolute_action(raw)
    np.testing.assert_allclose(out.canonical_action, raw)


def test_delta_action_conversion():
    current = np.ones(14, dtype=np.float32)
    raw = np.full(14, 0.1, dtype=np.float32)
    out = rlinf_robotwin_delta_to_canonical_target(raw, current)
    # Arm joints are delta.
    for idx in [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]:
        assert out.canonical_action[idx] == pytest.approx(1.1)
    # Grippers are absolute.
    assert out.canonical_action[6] == pytest.approx(0.1)
    assert out.canonical_action[13] == pytest.approx(0.1)


def test_gripper_mapping():
    current = np.ones(14, dtype=np.float32)
    raw = np.zeros(14, dtype=np.float32)
    raw[6] = 0.25
    raw[13] = 0.75
    out = rlinf_robotwin_delta_to_canonical_target(raw, current)
    assert out.canonical_action[6] == pytest.approx(0.25)
    assert out.canonical_action[13] == pytest.approx(0.75)


def test_normalize_unnormalize_roundtrip():
    # The validation package must not invent norm transforms. This test only
    # proves that a provided affine normalization can be exactly round-tripped.
    x = np.linspace(-1, 1, 14)
    mean = np.arange(14) * 0.01
    std = np.ones(14) * 0.5
    z = (x - mean) / std
    y = z * std + mean
    np.testing.assert_allclose(x, y)


def test_observation_adapter_requires_three_rlinf_cameras():
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    obs = adapt_real_observation_for_rlinf(
        {"cam_high": img, "cam_left_wrist": img, "cam_right_wrist": img, "cam_low": img},
        np.zeros(14),
        "adjust bottle",
    )
    assert set(obs.images) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}


def test_sft_checkpoint_output_shape_metadata():
    root = Path("aloha_rl_real_validation/checkpoints_meta/sft")
    if not (root / "config.json").exists():
        pytest.skip("checkpoint metadata not downloaded")
    schema = inspect_checkpoint_schema(root)
    assert schema.action_dim == 32
    assert schema.action_horizon == 10
    assert schema.has_trossen_norm_stats


def test_ppo_checkpoint_output_shape_metadata():
    root = Path("aloha_rl_real_validation/checkpoints_meta/ppo")
    if not (root / "config.json").exists():
        pytest.skip("checkpoint metadata not downloaded")
    schema = inspect_checkpoint_schema(root)
    assert schema.action_dim == 32
    assert schema.action_horizon == 10
    assert schema.has_trossen_norm_stats


def test_native_loader_blocked_for_sharded_rlinf_metadata():
    root = Path("aloha_rl_real_validation/checkpoints_meta/sft")
    if not (root / "config.json").exists():
        pytest.skip("checkpoint metadata not downloaded")
    schema = inspect_checkpoint_schema(root)
    blockers = validate_strict_openpi_native_compatibility(schema)
    assert any("single model.safetensors" in b for b in blockers)
