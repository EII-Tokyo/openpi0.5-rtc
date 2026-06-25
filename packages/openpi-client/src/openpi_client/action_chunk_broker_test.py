
import numpy as np

from openpi_client.action_chunk_broker import (
    ActionChunkBroker,
    _limit_key_region_action_delta,
    _propagate_actor_residual_for_guidance,
)
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


def test_propagates_signed_actor_residual_trend_to_rtc_guidance_tail():
    reference = np.zeros((50, 2), dtype=np.float32)
    adjusted = reference.copy()
    adjusted[:10, 0] = np.array([0.01, 0.03, 0.02, 0.04, 0.03, 0.05, 0.04, 0.05, 0.04, 0.05], dtype=np.float32)
    adjusted[:10, 1] = np.array([-0.02, -0.03, -0.01, -0.04, -0.03, -0.06, -0.05, -0.06, -0.05, -0.06], dtype=np.float32)

    guidance = _propagate_actor_residual_for_guidance(
        reference_actions=reference,
        adjusted_actions=adjusted,
        action_start_index=0,
        action_end_index=10,
        guidance_start_index=25,
        trend_window=5,
        start_weight=0.7,
    )

    np.testing.assert_allclose(guidance[:10], adjusted[:10])
    np.testing.assert_allclose(guidance[10:25], adjusted[10:25])
    np.testing.assert_allclose(guidance[25], np.array([0.0322, -0.0392], dtype=np.float32), rtol=1e-5)
    np.testing.assert_allclose(guidance[49], np.array([0.0, 0.0], dtype=np.float32), atol=1e-6)


def test_actor_apply_keeps_execution_tail_and_adds_separate_rtc_guidance_tail():
    class _SignedActor(_Actor):
        def apply(self, *, reference_actions, z_rl, proprio, context, action_start_index=None):
            adjusted = reference_actions.copy()
            adjusted[:10, 0] += 0.04
            adjusted[:10, 1] -= 0.06
            return RLTActorApplyResult(
                adjusted,
                True,
                None,
                "/tmp/actor",
                5,
                1.0,
                0.06,
                action_start_index=0,
                action_horizon=10,
                action_end_index=10,
            )

    reference = np.zeros((50, 2), dtype=np.float32)
    broker = ActionChunkBroker(_Policy(), action_horizon=50, use_rtc=False, rlt_actor_runtime=_SignedActor())

    results = broker._apply_rlt_actor_to_policy_results(
        {
            "actions": reference,
            "z_rl": np.ones((8,), dtype=np.float32),
            "state": np.ones((4,), dtype=np.float32),
        },
        {"rlt_context": {"actor_requested": True}},
        action_start_index=0,
    )

    np.testing.assert_allclose(results["actions"][25:], reference[25:])
    np.testing.assert_allclose(results["rtc_guidance_actions"][25], np.array([0.028, -0.042], dtype=np.float32))


def test_tail_trend_gate_weakens_same_direction_and_keeps_opposing_direction():
    reference = np.zeros((50, 2), dtype=np.float32)
    reference[25:, 0] = 0.02
    reference[25:, 1] = 0.02
    adjusted = reference.copy()
    adjusted[:10, 0] = 0.04
    adjusted[:10, 1] = -0.04

    guidance = _propagate_actor_residual_for_guidance(
        reference_actions=reference,
        adjusted_actions=adjusted,
        action_start_index=0,
        action_end_index=10,
        guidance_start_index=25,
        start_weight=0.7,
    )

    np.testing.assert_allclose(guidance[25, 0], 0.0298, rtol=1e-5)
    np.testing.assert_allclose(guidance[25, 1], -0.008, rtol=1e-5)


def test_key_region_limiter_constrains_left_arm_delta_conservatively():
    actions = np.zeros((3, 14), dtype=np.float32)
    actions[:, :6] = 1.0
    state = np.zeros((14,), dtype=np.float32)

    limited = _limit_key_region_action_delta(actions, state)

    np.testing.assert_allclose(np.linalg.norm(limited[:, :6] - state[:6], axis=-1), np.full(3, 0.005), rtol=1e-5)
    assert np.max(np.abs(limited[:, :6] - state[:6])) <= 0.0035 + 1e-6
    np.testing.assert_allclose(limited[:, 6:], actions[:, 6:])


def test_key_region_limiter_wraps_continuous_joints_to_nearest_equivalent_angle():
    actions = np.zeros((2, 14), dtype=np.float32)
    state = np.zeros((14,), dtype=np.float32)
    state[5] = 3.10
    actions[:, 5] = -3.10

    limited = _limit_key_region_action_delta(actions, state)

    assert np.all(limited[:, 5] > state[5])
    assert np.max(np.abs(limited[:, 5] - state[5])) <= 0.0035 + 1e-6


def test_key_region_limiter_is_applied_to_actor_actions_but_not_reference():
    class _LargeActor(_Actor):
        def apply(self, *, reference_actions, z_rl, proprio, context, action_start_index=None):
            adjusted = reference_actions.copy()
            adjusted[:, :6] = 1.0
            return RLTActorApplyResult(
                adjusted,
                True,
                None,
                "/tmp/actor",
                5,
                1.0,
                1.0,
                action_start_index=0,
                action_horizon=10,
                action_end_index=10,
            )

    broker = ActionChunkBroker(_Policy(), action_horizon=10, use_rtc=False, rlt_actor_runtime=_LargeActor())
    reference = np.zeros((10, 14), dtype=np.float32)
    results = broker._apply_rlt_actor_to_policy_results(
        {
            "actions": reference,
            "z_rl": np.ones((8,), dtype=np.float32),
            "state": np.zeros((14,), dtype=np.float32),
        },
        {"rlt_context": {"actor_requested": True, "phase": "key_region"}},
    )

    np.testing.assert_allclose(np.linalg.norm(results["actions"][:, :6], axis=-1), np.full(10, 0.005), rtol=1e-5)
    np.testing.assert_allclose(results["reference_actions"], reference)
    np.testing.assert_allclose(results["rtc_guidance_actions"], results["actions"])
    assert results["rlt_action_limited"] is True


def test_key_region_actor_limits_from_robot_state_and_freezes_right_arm():
    class _SignFlipActor(_Actor):
        def apply(self, *, reference_actions, z_rl, proprio, context, action_start_index=None):
            adjusted = reference_actions.copy()
            adjusted[:, 1] = 0.208
            adjusted[:, 2] = -0.592
            adjusted[:, 7:14] = 1.0
            return RLTActorApplyResult(
                adjusted,
                True,
                None,
                "/tmp/actor",
                5,
                1.0,
                1.0,
                action_start_index=0,
                action_horizon=10,
                action_end_index=10,
            )

    broker = ActionChunkBroker(_Policy(), action_horizon=10, use_rtc=False, rlt_actor_runtime=_SignFlipActor())
    reference = np.zeros((10, 14), dtype=np.float32)
    policy_state = np.zeros((14,), dtype=np.float32)
    policy_state[1] = 0.208
    policy_state[2] = -0.592
    robot_state = np.zeros((14,), dtype=np.float32)
    robot_state[1] = -0.210
    robot_state[2] = 0.594
    robot_state[7:14] = np.array([-0.8, 0.02, -0.04, 0.52, 0.51, 0.18, 0.66], dtype=np.float32)

    results = broker._apply_rlt_actor_to_policy_results(
        {
            "actions": reference,
            "z_rl": np.ones((8,), dtype=np.float32),
            "state": policy_state,
        },
        {"state": robot_state, "rlt_context": {"actor_requested": True, "phase": "key_region"}},
    )

    assert abs(results["actions"][0, 1] - robot_state[1]) <= 0.0035 + 1e-6
    assert abs(results["actions"][0, 2] - robot_state[2]) <= 0.0035 + 1e-6
    np.testing.assert_allclose(results["actions"][:, 7:14], np.broadcast_to(robot_state[7:14], (10, 7)))
    assert results["rlt_action_limited"] is True


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
