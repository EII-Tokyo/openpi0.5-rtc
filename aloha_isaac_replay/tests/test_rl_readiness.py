from __future__ import annotations

from aloha_isaac_replay.rl.readiness import build_rl_readiness_report


def test_drive_gate_pass_does_not_make_rl_training_ready() -> None:
    report = build_rl_readiness_report(drive_gate_pass=True, drive_gate_evidence="7/7 steps passed")
    assert report["status"] == "NOT_READY_FOR_RL_TRAINING"
    assert report["overall_rl_training_ready"] is False
    assert report["gates"][0]["status"] == "PASS"
    assert {gate["status"] for gate in report["gates"][1:]} == {"NOT_EVALUATED"}


def test_all_gates_required_for_rl_training_ready() -> None:
    report = build_rl_readiness_report(
        drive_gate_pass=True,
        drive_gate_evidence="tracking pass",
        reset_gate_pass=True,
        causality_gate_pass=True,
        contact_reward_gate_pass=True,
        observation_gate_pass=True,
    )
    assert report["status"] == "READY_FOR_RL_TRAINING"
    assert report["overall_rl_training_ready"] is True

