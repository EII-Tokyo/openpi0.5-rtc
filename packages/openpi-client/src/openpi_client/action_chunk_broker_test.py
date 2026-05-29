
import numpy as np

from openpi_client.action_chunk_broker import ActionChunkBroker
from openpi_client.rlt_actor_runtime import RLTActorApplyResult


class _Policy:
    def __init__(self):
        self.obs_seen = []

    def infer(self, obs, *args):
        self.obs_seen.append(dict(obs))
        return {
            "actions": np.arange(6, dtype=np.float32).reshape(3, 2),
            "z_rl": np.ones((8,), dtype=np.float32),
            "state": np.ones((4,), dtype=np.float32),
        }

    def reset(self):
        pass


class _Actor:
    def __init__(self, *, mode="apply"):
        self.mode = mode
        self.calls = []

    def apply(self, *, reference_actions, z_rl, proprio, context):
        self.calls.append((reference_actions.copy(), z_rl.copy(), proprio.copy(), dict(context)))
        if self.mode == "raise":
            raise RuntimeError("actor failed")
        if self.mode == "disabled":
            return RLTActorApplyResult(reference_actions.copy(), False, "actor_not_requested", None, None, None, None)
        return RLTActorApplyResult(reference_actions + 10, True, None, "/tmp/actor", 5, 1.0, 0.5)

    def status(self):
        return {"actor_ready": self.mode == "apply", "actor_step": 5}


def test_actor_disabled_leaves_actions_unchanged_and_sets_reference_action():
    policy = _Policy()
    actor = _Actor(mode="disabled")
    broker = ActionChunkBroker(policy, action_horizon=3, use_rtc=False, rlt_actor_runtime=actor)

    result = broker.infer({"rlt_context": {"actor_requested": False}})

    np.testing.assert_allclose(result["actions"], np.array([0, 1], dtype=np.float32))
    np.testing.assert_allclose(result["reference_action"], np.array([0, 1], dtype=np.float32))
    assert result["rlt_actor_applied"] is False
    assert result["rlt_actor_reason"] == "actor_not_requested"
    assert "rlt_context" not in policy.obs_seen[0]


def test_actor_enabled_replaces_actions_and_preserves_raw_reference():
    actor = _Actor(mode="apply")
    broker = ActionChunkBroker(_Policy(), action_horizon=3, use_rtc=False, rlt_actor_runtime=actor)

    first = broker.infer({"rlt_context": {"actor_requested": True, "intervention_scale": 0.25}})
    second = broker.infer({"rlt_context": {"actor_requested": True, "intervention_scale": 0.25}})

    np.testing.assert_allclose(first["actions"], np.array([10, 11], dtype=np.float32))
    np.testing.assert_allclose(first["reference_action"], np.array([0, 1], dtype=np.float32))
    np.testing.assert_allclose(first["action_full"], np.arange(6, dtype=np.float32).reshape(3, 2) + 10)
    np.testing.assert_allclose(first["reference_action_full"], np.arange(6, dtype=np.float32).reshape(3, 2))
    assert first["rlt_actor_applied"] is True
    assert first["rlt_actor_step"] == 5
    assert second["rlt_actor_applied"] is True
    assert len(actor.calls) == 1


def test_actor_failure_leaves_actions_unchanged_and_records_reason():
    actor = _Actor(mode="raise")
    broker = ActionChunkBroker(_Policy(), action_horizon=3, use_rtc=False, rlt_actor_runtime=actor)

    result = broker.infer({"rlt_context": {"actor_requested": True}})

    np.testing.assert_allclose(result["actions"], np.array([0, 1], dtype=np.float32))
    np.testing.assert_allclose(result["reference_action"], np.array([0, 1], dtype=np.float32))
    assert result["rlt_actor_applied"] is False
    assert "actor failed" in result["rlt_actor_reason"]
