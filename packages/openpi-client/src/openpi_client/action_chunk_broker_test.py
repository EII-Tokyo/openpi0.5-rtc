
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
    def __init__(self, *, horizon=10, delta=1000.0):
        self.horizon = horizon
        self.delta = delta
        self.calls = []

    def apply(self, *, reference_actions, z_rl, proprio, context, action_start_index=None):
        start = int(action_start_index or 0)
        end = min(start + self.horizon, reference_actions.shape[0])
        self.calls.append((reference_actions.copy(), z_rl.copy(), proprio.copy(), dict(context), start))
        if not context.get("actor_requested", False):
            return RLTActorApplyResult(
                reference_actions.copy(),
                False,
                "actor_not_requested",
                None,
                None,
                None,
                None,
                gate_reason="actor_not_requested",
                action_start_index=start,
                action_horizon=self.horizon,
                action_end_index=end,
            )
        adjusted_actions = reference_actions.copy()
        adjusted_actions[start:end] += self.delta
        return RLTActorApplyResult(
            adjusted_actions,
            True,
            None,
            "/tmp/actor",
            5,
            1.0,
            self.delta,
            reference_q_value=0.2,
            actor_q_value=0.7,
            q_advantage=0.5,
            key_region_probability=0.9,
            gate_reason="critic_gate_actor_active",
            critic_ready=True,
            critic_gate_enabled=True,
            action_start_index=start,
            action_horizon=self.horizon,
            action_end_index=end,
        )


class _FixedActor:
    def __init__(self, actions):
        self.actions = np.asarray(actions, dtype=np.float32)
        self.calls = []

    def apply(self, *, reference_actions, z_rl, proprio, context, action_start_index=None):
        self.calls.append((reference_actions.copy(), z_rl.copy(), proprio.copy(), dict(context), action_start_index))
        if not context.get("actor_requested", False):
            return RLTActorApplyResult(reference_actions.copy(), False, "actor_not_requested", None, None, None, None)
        return RLTActorApplyResult(
            self.actions.copy(),
            True,
            None,
            "/tmp/actor",
            5,
            float(np.linalg.norm(self.actions - reference_actions)),
            float(np.max(np.abs(self.actions - reference_actions))),
            gate_reason="critic_gate_actor_active",
            critic_ready=True,
            critic_gate_enabled=True,
        )


def _long_policy_results(length=50):
    return {
        "actions": np.arange(length * 2, dtype=np.float32).reshape(length, 2),
        "z_rl": np.ones((8,), dtype=np.float32),
        "state": np.ones((4,), dtype=np.float32),
    }


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


def test_actor_only_modifies_foreground_window_and_preserves_vla_tail():
    actor = _WindowActor(horizon=10, delta=1000.0)
    broker = ActionChunkBroker(_Policy(), action_horizon=50, use_rtc=False, rlt_actor_runtime=actor)
    policy_results = _long_policy_results()
    raw_actions = policy_results["actions"].copy()

    result = broker._apply_rlt_actor_to_policy_results(
        policy_results,
        {"rlt_context": {"actor_requested": True}},
        action_start_index=0,
    )

    np.testing.assert_allclose(result["actions"][:10], raw_actions[:10] + 1000.0)
    np.testing.assert_allclose(result["actions"][10:], raw_actions[10:])
    np.testing.assert_allclose(result["reference_actions"], raw_actions)
    assert result["rlt_actor_action_start_index"] == 0
    assert result["rlt_actor_action_horizon"] == 10
    assert result["rlt_actor_action_end_index"] == 10


def test_actor_only_modifies_background_delayed_window_and_preserves_vla_tail():
    actor = _WindowActor(horizon=10, delta=1000.0)
    broker = ActionChunkBroker(_Policy(), action_horizon=50, use_rtc=False, rlt_actor_runtime=actor)
    policy_results = _long_policy_results()
    raw_actions = policy_results["actions"].copy()

    result = broker._apply_rlt_actor_to_policy_results(
        policy_results,
        {"rlt_context": {"actor_requested": True}},
        action_start_index=10,
    )

    np.testing.assert_allclose(result["actions"][:10], raw_actions[:10])
    np.testing.assert_allclose(result["actions"][10:20], raw_actions[10:20] + 1000.0)
    np.testing.assert_allclose(result["actions"][20:], raw_actions[20:])
    np.testing.assert_allclose(result["reference_actions"], raw_actions)
    assert result["rlt_actor_action_start_index"] == 10
    assert result["rlt_actor_action_horizon"] == 10
    assert result["rlt_actor_action_end_index"] == 20


def test_projected_slow_push_blend_scales_vla_push_and_keeps_actor_align_only():
    reference_actions = np.zeros((3, 14), dtype=np.float32)
    reference_actions[:, 0] = 10.0
    reference_actions[:, 1] = 4.0
    reference_actions[:, 7] = 20.0
    actor_actions = reference_actions.copy()
    actor_actions[:, 0] += 100.0
    actor_actions[:, 1] += 6.0
    actor_actions[:, 7] += 100.0
    actor = _FixedActor(actor_actions)
    broker = ActionChunkBroker(_Policy(), action_horizon=3, use_rtc=False, rlt_actor_runtime=actor)

    result = broker._apply_rlt_actor_to_policy_results(
        {
            "actions": reference_actions,
            "z_rl": np.ones((8,), dtype=np.float32),
            "state": np.zeros((32,), dtype=np.float32),
        },
        {
            "rlt_context": {
                "actor_requested": True,
                "phase": "key_region",
                "rlt_blend_mode": "projected_slow_push",
                "lambda_push": 0.10,
                "lambda_vla_align": 0.50,
                "lambda_actor": 0.20,
                "push_joint_indices": [0, 1],
                "push_axis": [1.0, 0.0],
            }
        },
    )

    np.testing.assert_allclose(result["actions"][:, 0], np.full((3,), 1.0, dtype=np.float32))
    np.testing.assert_allclose(result["actions"][:, 1], np.full((3,), 3.2, dtype=np.float32))
    np.testing.assert_allclose(result["actions"][:, 7], reference_actions[:, 7])
    np.testing.assert_allclose(result["reference_actions"], reference_actions)
    assert result["rlt_blend_mode"] == "projected_slow_push"
    assert result["rlt_blend_preset"] == "custom"
    assert result["rlt_lambda_push"] == 0.10
    assert result["rlt_lambda_vla_align"] == 0.50
    assert result["rlt_lambda_actor"] == 0.20
    assert result["rlt_actor_removed_push_norm"] > 0
    assert result["rlt_actor_align_norm"] > 0


def test_projected_slow_push_invalid_joint_index_falls_back_to_actor_action():
    reference_actions = np.zeros((3, 14), dtype=np.float32)
    actor_actions = reference_actions + 3.0
    actor = _FixedActor(actor_actions)
    broker = ActionChunkBroker(_Policy(), action_horizon=3, use_rtc=False, rlt_actor_runtime=actor)

    result = broker._apply_rlt_actor_to_policy_results(
        {
            "actions": reference_actions,
            "z_rl": np.ones((8,), dtype=np.float32),
            "state": np.zeros((32,), dtype=np.float32),
        },
        {
            "rlt_context": {
                "actor_requested": True,
                "phase": "key_region",
                "rlt_blend_mode": "projected_slow_push",
                "push_joint_indices": [0, 99],
                "push_axis": [1.0, 0.0],
            }
        },
    )

    np.testing.assert_allclose(result["actions"], actor_actions)
    assert result["rlt_blend_preset"] == "invalid"


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
