
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
        self.on_apply = None

    def apply(self, *, reference_actions, z_rl, proprio, context, action_start_index=None):
        self.calls.append((reference_actions.copy(), z_rl.copy(), proprio.copy(), dict(context), action_start_index))
        if self.on_apply is not None:
            callback = self.on_apply
            self.on_apply = None
            callback()
        if self.mode == "raise":
            raise RuntimeError("actor failed")
        if self.mode == "disabled" or not context.get("actor_requested", False):
            return RLTActorApplyResult(reference_actions.copy(), False, "actor_not_requested", None, None, None, None, gate_reason="actor_not_requested")
        return RLTActorApplyResult(reference_actions + 10, True, None, "/tmp/actor", 5, 1.0, 0.5, reference_q_value=0.2, actor_q_value=0.7, q_advantage=0.5, key_region_probability=0.9, gate_reason="critic_gate_actor_active", critic_ready=True, critic_gate_enabled=True)

    def maybe_reload(self, *, force=False):
        self.force_reload = force

    def status(self):
        return {"actor_ready": self.mode == "apply", "actor_step": 5}


class _WindowActor:
    def __init__(self, *, horizon=2):
        self.horizon = horizon
        self.calls = []

    def apply(self, *, reference_actions, z_rl, proprio, context, action_start_index=None):
        del z_rl, proprio
        start = int(action_start_index or 0)
        end = start + self.horizon
        self.calls.append((reference_actions.copy(), dict(context), start, end))
        adjusted = reference_actions.copy()
        adjusted[start:end] += 100
        return RLTActorApplyResult(
            adjusted,
            True,
            None,
            "/tmp/actor",
            5,
            1.0,
            0.5,
            action_start_index=start,
            action_horizon=self.horizon,
            action_end_index=end,
        )


class _LongPolicy:
    def __init__(self):
        self.calls = 0

    def infer(self, obs, *args):
        del obs, args
        base = self.calls * 1000
        self.calls += 1
        return {
            "actions": (base + np.arange(10, dtype=np.float32)).reshape(5, 2),
            "z_rl": np.ones((8,), dtype=np.float32),
            "state": np.ones((4,), dtype=np.float32),
        }

    def reset(self):
        pass


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
    assert first["rlt_reference_q"] == 0.2
    assert first["rlt_actor_q"] == 0.7
    assert first["rlt_q_advantage"] == 0.5
    assert first["rlt_key_region_probability"] == 0.9
    assert first["rlt_gate_reason"] == "critic_gate_actor_active"
    assert first["rlt_critic_ready"] is True
    assert first["rlt_critic_gate_enabled"] is True
    assert second["rlt_actor_applied"] is True
    assert len(actor.calls) == 1
    assert actor.calls[0][4] == 0


def test_actor_window_mode_refreshes_instead_of_executing_vla_tail():
    policy = _LongPolicy()
    actor = _WindowActor(horizon=2)
    broker = ActionChunkBroker(policy, action_horizon=5, use_rtc=False, rlt_actor_runtime=actor)
    obs = {
        "rlt_context": {
            "actor_requested": True,
            "actor_execution_mode": "wait_next_chunk",
            "disable_vla_tail_when_actor_active": True,
        }
    }

    first = broker.infer(obs)
    second = broker.infer(obs)
    third = broker.infer(obs)

    np.testing.assert_allclose(first["actions"], np.array([100, 101], dtype=np.float32))
    np.testing.assert_allclose(second["actions"], np.array([102, 103], dtype=np.float32))
    np.testing.assert_allclose(third["actions"], np.array([1100, 1101], dtype=np.float32))
    assert policy.calls == 2
    assert len(actor.calls) == 2


def test_actor_gate_on_invalidates_cached_disabled_chunk():
    actor = _Actor(mode="apply")
    broker = ActionChunkBroker(_Policy(), action_horizon=3, use_rtc=False, rlt_actor_runtime=actor)

    first = broker.infer({"rlt_context": {"actor_requested": False}})
    second = broker.infer({"rlt_context": {"actor_requested": True}})

    np.testing.assert_allclose(first["actions"], np.array([0, 1], dtype=np.float32))
    np.testing.assert_allclose(second["actions"], np.array([10, 11], dtype=np.float32))
    assert first["rlt_actor_applied"] is False
    assert second["rlt_actor_applied"] is True
    assert len(actor.calls) == 2


def test_actor_gate_off_invalidates_cached_actor_chunk():
    actor = _Actor(mode="apply")
    broker = ActionChunkBroker(_Policy(), action_horizon=3, use_rtc=False, rlt_actor_runtime=actor)

    first = broker.infer({"rlt_context": {"actor_requested": True}})
    second = broker.infer({"rlt_context": {"actor_requested": False}})

    np.testing.assert_allclose(first["actions"], np.array([10, 11], dtype=np.float32))
    np.testing.assert_allclose(second["actions"], np.array([0, 1], dtype=np.float32))
    assert first["rlt_actor_applied"] is True
    assert second["rlt_actor_applied"] is False
    assert second["rlt_actor_reason"] == "actor_not_requested"
    assert len(actor.calls) == 2


def test_flush_action_cache_discards_cached_actor_chunk():
    actor = _Actor(mode="apply")
    broker = ActionChunkBroker(_Policy(), action_horizon=3, use_rtc=False, rlt_actor_runtime=actor)

    first = broker.infer({"rlt_context": {"actor_requested": True, "rlt_context_epoch": 1}})
    broker.flush_action_cache("key_region_end")
    second = broker.infer({"rlt_context": {"actor_requested": True, "rlt_context_epoch": 1}})

    np.testing.assert_allclose(first["actions"], np.array([10, 11], dtype=np.float32))
    np.testing.assert_allclose(second["actions"], np.array([10, 11], dtype=np.float32))
    assert len(actor.calls) == 2


def test_explicit_rlt_gate_overrides_observation_context():
    actor = _Actor(mode="apply")
    broker = ActionChunkBroker(_Policy(), action_horizon=3, use_rtc=False, rlt_actor_runtime=actor)

    broker.set_rlt_gate(enabled=True, epoch=7, reason="key_region_start")
    enabled = broker.infer({"rlt_context": {"actor_requested": False, "rlt_context_epoch": 1}})

    broker.set_rlt_gate(enabled=False, epoch=8, reason="key_region_end")
    disabled = broker.infer({"rlt_context": {"actor_requested": True, "rlt_context_epoch": 7}})

    assert enabled["rlt_actor_applied"] is True
    assert enabled["rlt_context_epoch"] == 7
    assert disabled["rlt_actor_applied"] is False
    assert disabled["rlt_context_epoch"] == 8
    assert actor.calls[0][3]["actor_requested"] is True
    assert actor.calls[0][3]["rlt_context_epoch"] == 7
    assert actor.calls[1][3]["actor_requested"] is False
    assert actor.calls[1][3]["rlt_context_epoch"] == 8


def test_foreground_actor_result_is_discarded_when_gate_changes_mid_inference():
    actor = _Actor(mode="apply")
    broker = ActionChunkBroker(_Policy(), action_horizon=3, use_rtc=False, rlt_actor_runtime=actor)
    broker.set_rlt_gate(enabled=True, epoch=1, reason="key_region_start")
    actor.on_apply = lambda: broker.set_rlt_gate(enabled=False, epoch=2, reason="key_region_end")

    result = broker.infer({"rlt_context": {"actor_requested": True, "rlt_context_epoch": 1}})

    assert result["rlt_actor_applied"] is False
    assert result["rlt_context_epoch"] == 2
    np.testing.assert_allclose(result["actions"], np.array([0, 1], dtype=np.float32))
    assert len(actor.calls) == 2
    assert actor.calls[0][3]["actor_requested"] is True
    assert actor.calls[1][3]["actor_requested"] is False


def test_broker_passes_delayed_execution_window_to_actor_runtime():
    actor = _Actor(mode="apply")
    broker = ActionChunkBroker(_Policy(), action_horizon=3, use_rtc=False, rlt_actor_runtime=actor)
    policy_results = {
        "actions": np.arange(100, dtype=np.float32).reshape(50, 2),
        "z_rl": np.ones((8,), dtype=np.float32),
        "state": np.ones((4,), dtype=np.float32),
    }

    broker._apply_rlt_actor_to_policy_results(
        policy_results,
        {"rlt_context": {"actor_requested": True}},
        action_start_index=10,
    )

    assert actor.calls[0][4] == 10


def test_actor_failure_leaves_actions_unchanged_and_records_reason():
    actor = _Actor(mode="raise")
    broker = ActionChunkBroker(_Policy(), action_horizon=3, use_rtc=False, rlt_actor_runtime=actor)

    result = broker.infer({"rlt_context": {"actor_requested": True}})

    np.testing.assert_allclose(result["actions"], np.array([0, 1], dtype=np.float32))
    np.testing.assert_allclose(result["reference_action"], np.array([0, 1], dtype=np.float32))
    assert result["rlt_actor_applied"] is False
    assert "actor failed" in result["rlt_actor_reason"]


def test_broker_exposes_rlt_actor_runtime_status():
    actor = _Actor(mode="apply")
    broker = ActionChunkBroker(_Policy(), action_horizon=3, use_rtc=False, rlt_actor_runtime=actor)

    assert broker.rlt_actor_status() == {"actor_ready": True, "actor_step": 5}
    assert actor.force_reload is True
