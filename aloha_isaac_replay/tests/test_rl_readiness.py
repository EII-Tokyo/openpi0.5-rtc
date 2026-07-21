from __future__ import annotations

from aloha_isaac_replay.rl.readiness import build_rl_readiness_report


def test_drive_gate_pass_does_not_make_rl_training_ready() -> None:
    report = build_rl_readiness_report(drive_gate_pass=True, drive_gate_evidence="7/7 steps passed")
    assert report["status"] == "NOT_READY_FOR_RL_TRAINING"
    assert report["overall_rl_training_ready"] is False
    assert report["privileged_state_rl_training_ready"] is False
    assert report["gates"][0]["status"] == "PASS"
    assert {gate["status"] for gate in report["gates"][1:]} == {"NOT_EVALUATED"}
    assert "Fixed-initial-state replay is a calibration gate" in report["replay_scope"]


def test_fixed_pose_and_randomized_true_state_make_privileged_state_rl_ready() -> None:
    report = build_rl_readiness_report(
        drive_gate_pass=True,
        drive_gate_evidence="tracking pass",
        fixed_pose_minimal_task_pass=True,
        fixed_pose_minimal_task_evidence="scripted fixed bottle grasp and lift pass",
        randomized_true_state_task_pass=True,
        randomized_true_state_task_evidence="small random reset train/eval pass with simulator truth",
    )
    assert report["status"] == "READY_FOR_PRIVILEGED_STATE_RL_TRAINING_NOT_CAMERA"
    assert report["privileged_state_rl_training_ready"] is True
    assert report["camera_based_rl_training_ready"] is False
    assert report["overall_rl_training_ready"] is False


def test_camera_gate_required_for_overall_camera_based_rl_ready() -> None:
    report = build_rl_readiness_report(
        drive_gate_pass=True,
        drive_gate_evidence="tracking pass",
        fixed_pose_minimal_task_pass=True,
        randomized_true_state_task_pass=True,
        camera_perception_task_pass=True,
        camera_perception_task_evidence="camera/keypoint observation policy pass",
    )
    assert report["status"] == "READY_FOR_CAMERA_BASED_RL_TRAINING"
    assert report["overall_rl_training_ready"] is True
    assert report["camera_based_rl_training_ready"] is True


def test_legacy_subgates_are_diagnostic_not_canonical_readiness() -> None:
    report = build_rl_readiness_report(
        drive_gate_pass=True,
        drive_gate_evidence="tracking pass",
        reset_gate_pass=True,
        causality_gate_pass=True,
        contact_reward_gate_pass=True,
        observation_gate_pass=True,
    )
    assert report["status"] == "NOT_READY_FOR_RL_TRAINING"
    assert report["legacy_subgates"]["reset_gate_pass"] is True
    assert "diagnostic only" in report["legacy_subgates"]["note"]
