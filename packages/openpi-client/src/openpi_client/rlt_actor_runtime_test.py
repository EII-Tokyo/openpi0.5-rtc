
import json

import jax
import numpy as np

from openpi.models import rlt
from openpi.training import rlt_replay_store
from openpi.training import rlt_training
from openpi_client.rlt_actor_runtime import RLTActorRuntime
from openpi_client.rlt_actor_runtime import _clip_adjusted_action_delta
from scripts import train_rlt_online


class _FixedConfig:
    action_horizon = 4
    action_dim = 2
    z_dim = 8
    proprio_dim = 4


class _FixedActor:
    def __init__(self):
        self.sample_values = []

    def __call__(self, _x, reference_action, *, rng=None, sample=False, intervention_scale=1.0):
        del rng
        self.sample_values.append(sample)
        return reference_action + intervention_scale * np.ones_like(np.asarray(reference_action), dtype=np.float32)


def _write_actor(tmp_path, *, action_horizon=10, action_dim=3):
    config = rlt_training.RLTTrainingConfig(
        model=rlt.RLTConfig(
            z_dim=8,
            proprio_dim=4,
            action_horizon=action_horizon,
            action_dim=action_dim,
            hidden_dim=16,
            num_layers=2,
        )
    )
    state = rlt_training.init_train_state(config, jax.random.key(0))
    return train_rlt_online._save_actor_for_inference(
        state,
        tmp_path,
        5,
        action_horizon=action_horizon,
        replay_shape=rlt_replay_store.ReplayShape(8, 4, 50, action_dim),
        train_shape=rlt_replay_store.ReplayShape(8, 4, action_horizon, action_dim),
        replay_stats=rlt_replay_store.ReplayStats(10, 1, 1, 1, 0),
    )


def test_actor_runtime_loads_latest_and_preserves_suffix(tmp_path):
    actor_dir = _write_actor(tmp_path, action_horizon=10, action_dim=3)
    runtime = RLTActorRuntime(str(tmp_path / "inference_actor" / "LATEST"), poll_interval_seconds=0.0)

    runtime.maybe_reload(force=True)
    result = runtime.apply(
        reference_actions=np.ones((12, 3), dtype=np.float32),
        z_rl=np.zeros((8,), dtype=np.float32),
        proprio=np.zeros((4,), dtype=np.float32),
        context={"actor_requested": True, "intervention_scale": 0.25},
    )

    assert result.applied is True
    assert result.actor_dir == str(actor_dir)
    assert result.actor_step == 5
    assert result.actions.shape == (12, 3)
    np.testing.assert_allclose(result.actions[10:], np.ones((2, 3), dtype=np.float32))
    assert runtime.status()["actor_ready"] is True


def test_actor_runtime_applies_to_requested_action_window(tmp_path):
    _write_actor(tmp_path, action_horizon=10, action_dim=3)
    runtime = RLTActorRuntime(str(tmp_path / "inference_actor" / "LATEST"), poll_interval_seconds=0.0)
    reference = np.ones((25, 3), dtype=np.float32)

    result = runtime.apply(
        reference_actions=reference,
        z_rl=np.zeros((8,), dtype=np.float32),
        proprio=np.zeros((4,), dtype=np.float32),
        context={"actor_requested": True, "intervention_scale": 0.25},
        action_start_index=10,
    )

    assert result.applied is True
    assert result.action_start_index == 10
    assert result.action_horizon == 10
    assert result.action_end_index == 20
    result_delta = result.actions - reference
    np.testing.assert_allclose(result_delta[:10], np.zeros((10, 3), dtype=np.float32))
    assert np.linalg.norm(result_delta[10:20]) > 0.0
    np.testing.assert_allclose(result_delta[20:], np.zeros((5, 3), dtype=np.float32))


def test_actor_runtime_uses_deterministic_actor_actions(tmp_path):
    _write_actor(tmp_path, action_horizon=10, action_dim=3)
    runtime = RLTActorRuntime(str(tmp_path / "inference_actor" / "LATEST"), poll_interval_seconds=0.0)
    reference = np.ones((10, 3), dtype=np.float32)
    context = {"actor_requested": True, "intervention_scale": 0.25}

    first = runtime.apply(
        reference_actions=reference,
        z_rl=np.zeros((8,), dtype=np.float32),
        proprio=np.zeros((4,), dtype=np.float32),
        context=context,
    )
    second = runtime.apply(
        reference_actions=reference,
        z_rl=np.zeros((8,), dtype=np.float32),
        proprio=np.zeros((4,), dtype=np.float32),
        context=context,
    )

    assert first.applied is True
    assert second.applied is True
    np.testing.assert_allclose(first.actions, second.actions)


def test_actor_runtime_ramps_intervention_delta():
    runtime = RLTActorRuntime(None, poll_interval_seconds=0.0)
    fixed_actor = _FixedActor()
    runtime._actor = fixed_actor
    runtime._config = _FixedConfig()
    reference = np.zeros((4, 2), dtype=np.float32)

    result = runtime.apply(
        reference_actions=reference,
        z_rl=np.zeros((8,), dtype=np.float32),
        proprio=np.zeros((4,), dtype=np.float32),
        context={
            "actor_requested": True,
            "intervention_scale": 1.0,
            "intervention_ramp_steps": 4,
        },
    )

    assert result.applied is True
    assert fixed_actor.sample_values == [False]
    expected = np.array(
        [
            [0.0, 0.0],
            [1.0 / 3.0, 1.0 / 3.0],
            [2.0 / 3.0, 2.0 / 3.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(result.actions, expected, rtol=1e-6, atol=1e-6)


def test_actor_runtime_clips_intervention_delta_after_scaling():
    reference = np.zeros((3, 2), dtype=np.float32)
    adjusted = np.array([[0.5, -0.3], [0.02, -0.04], [-0.2, 0.2]], dtype=np.float32)

    clipped = _clip_adjusted_action_delta(
        reference_prefix=reference,
        adjusted_prefix=adjusted,
        max_delta=0.1,
    )

    expected = np.array([[0.1, -0.1], [0.02, -0.04], [-0.1, 0.1]], dtype=np.float32)
    np.testing.assert_allclose(clipped, expected)


def test_actor_runtime_rejects_action_window_that_exceeds_reference(tmp_path):
    _write_actor(tmp_path, action_horizon=10, action_dim=3)
    runtime = RLTActorRuntime(str(tmp_path / "inference_actor" / "LATEST"), poll_interval_seconds=0.0)
    reference = np.ones((15, 3), dtype=np.float32)

    result = runtime.apply(
        reference_actions=reference,
        z_rl=np.zeros((8,), dtype=np.float32),
        proprio=np.zeros((4,), dtype=np.float32),
        context={"actor_requested": True, "intervention_scale": 0.25},
        action_start_index=10,
    )

    assert result.applied is False
    assert "reference horizon" in result.reason
    np.testing.assert_allclose(result.actions, reference)


def test_actor_runtime_bad_metadata_fails_closed(tmp_path):
    actor_dir = _write_actor(tmp_path)
    metadata_path = actor_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("rlt_config")
    metadata_path.write_text(json.dumps(metadata))

    runtime = RLTActorRuntime(str(actor_dir), poll_interval_seconds=0.0)
    result = runtime.apply(
        reference_actions=np.ones((10, 3), dtype=np.float32),
        z_rl=np.zeros((8,), dtype=np.float32),
        proprio=np.zeros((4,), dtype=np.float32),
        context={"actor_requested": True},
    )

    assert result.applied is False
    assert "rlt_config" in result.reason
    np.testing.assert_allclose(result.actions, np.ones((10, 3), dtype=np.float32))
    assert runtime.status()["actor_ready"] is False


def test_actor_runtime_shape_mismatch_fails_closed(tmp_path):
    _write_actor(tmp_path, action_horizon=10, action_dim=3)
    runtime = RLTActorRuntime(str(tmp_path / "inference_actor"), poll_interval_seconds=0.0)
    reference = np.ones((10, 4), dtype=np.float32)

    result = runtime.apply(
        reference_actions=reference,
        z_rl=np.zeros((8,), dtype=np.float32),
        proprio=np.zeros((4,), dtype=np.float32),
        context={"actor_requested": True},
    )

    assert result.applied is False
    assert "action_dim" in result.reason
    np.testing.assert_allclose(result.actions, reference)


def test_actor_runtime_missing_actor_or_disabled_fails_closed(tmp_path):
    runtime = RLTActorRuntime(str(tmp_path / "missing"), poll_interval_seconds=0.0)
    reference = np.ones((10, 3), dtype=np.float32)

    disabled = runtime.apply(
        reference_actions=reference,
        z_rl=np.zeros((8,), dtype=np.float32),
        proprio=np.zeros((4,), dtype=np.float32),
        context={"actor_requested": False},
    )
    missing = runtime.apply(
        reference_actions=reference,
        z_rl=np.zeros((8,), dtype=np.float32),
        proprio=np.zeros((4,), dtype=np.float32),
        context={"actor_requested": True},
    )

    assert disabled.applied is False
    assert disabled.reason == "actor_not_requested"
    assert missing.applied is False
    assert "not found" in missing.reason
    np.testing.assert_allclose(disabled.actions, reference)
    np.testing.assert_allclose(missing.actions, reference)


def test_actor_runtime_critic_gate_reports_metrics_when_active(tmp_path):
    actor_dir = _write_actor(tmp_path, action_horizon=10, action_dim=3)
    runtime = RLTActorRuntime(str(actor_dir), poll_interval_seconds=0.0)
    result = runtime.apply(
        reference_actions=np.ones((10, 3), dtype=np.float32),
        z_rl=np.zeros((8,), dtype=np.float32),
        proprio=np.zeros((4,), dtype=np.float32),
        context={
            "actor_requested": True,
            "critic_gate_enabled": True,
            "critic_gate_margin": -1000.0,
            "critic_gate_temperature": 0.05,
        },
    )

    assert result.applied is True
    assert result.critic_ready is True
    assert result.critic_gate_enabled is True
    assert result.gate_reason == "critic_gate_actor_active"
    assert result.reference_q_value is not None
    assert result.actor_q_value is not None
    assert result.q_advantage == result.actor_q_value - result.reference_q_value
    assert result.key_region_probability is not None


def test_actor_runtime_critic_gate_rejects_low_advantage(tmp_path):
    actor_dir = _write_actor(tmp_path, action_horizon=10, action_dim=3)
    runtime = RLTActorRuntime(str(actor_dir), poll_interval_seconds=0.0)
    reference = np.ones((10, 3), dtype=np.float32)
    result = runtime.apply(
        reference_actions=reference,
        z_rl=np.zeros((8,), dtype=np.float32),
        proprio=np.zeros((4,), dtype=np.float32),
        context={
            "actor_requested": True,
            "critic_gate_enabled": True,
            "critic_gate_margin": 1000.0,
            "critic_gate_temperature": 0.05,
        },
    )

    assert result.applied is False
    assert result.gate_reason == "critic_gate_q_advantage_low"
    assert result.critic_ready is True
    np.testing.assert_allclose(result.actions, reference)
