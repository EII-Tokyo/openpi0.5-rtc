
import json

import jax
import numpy as np

from openpi.models import rlt
from openpi.training import rlt_replay_store
from openpi.training import rlt_training
from openpi_client.rlt_actor_runtime import RLTActorRuntime
from scripts import train_rlt_online


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
