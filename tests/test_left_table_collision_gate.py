import math
from pathlib import Path

from tools.isaac_sim.left_table_collision_gate import (
    TABLE_PATH,
    TrialMetrics,
    aggregate_trials,
    evaluate_trial,
)


TIP = (
    "/World/follower_left/vx300s_left/"
    "follower_left_left_finger_link/collisions/tip"
)


def passing_trial() -> TrialMetrics:
    return TrialMetrics(
        contact_pairs=[(TABLE_PATH, TIP)],
        minimum_target_separation_m=-0.00005,
        minimum_table_local_finger_z_m=-0.00005,
        maximum_visual_collision_error_m=0.0,
        final_target_error_rad=math.radians(8),
        persistent_contact_steps=180,
        finite=True,
        within_joint_limits=True,
        ccd_effective=True,
        disallowed_tip_contacts=[],
        physx_errors=[],
    )


def test_pass_requires_exact_contact_non_crossing_and_blocked_target():
    result = evaluate_trial(passing_trial())

    assert result["status"] == "PASS"
    assert result["target_contact_found"] is True
    assert result["tabletop_penetrated"] is False
    assert result["infeasible_target_blocked"] is True


def test_unrelated_contact_and_tabletop_penetration_fail():
    trial = passing_trial()
    trial.contact_pairs = [("/World/environment/worldBody/__1", TIP)]
    trial.minimum_table_local_finger_z_m = -0.001

    result = evaluate_trial(trial)

    assert result["status"] == "FAIL"
    assert "missing_physical_table_tip_contact" in result["failure_reasons"]
    assert "finger_penetrated_table_top" in result["failure_reasons"]


def test_positive_separation_contact_header_is_only_proximity():
    trial = passing_trial()
    trial.minimum_target_separation_m = 0.0106

    result = evaluate_trial(trial)

    assert result["status"] == "FAIL"
    assert result["target_contact_found"] is False
    assert "missing_physical_table_tip_contact" in result["failure_reasons"]


def test_visual_collision_mismatch_fails_closed():
    trial = passing_trial()
    trial.maximum_visual_collision_error_m = 0.0002

    result = evaluate_trial(trial)

    assert result["status"] == "FAIL"
    assert "visual_collision_mismatch" in result["failure_reasons"]


def test_exactly_three_passing_trials_are_required():
    assert aggregate_trials([evaluate_trial(passing_trial())] * 2)["status"] == "FAIL"
    assert (
        aggregate_trials([evaluate_trial(passing_trial()) for _ in range(3)])["status"]
        == "PASS"
    )


def test_runtime_verifier_uses_signed_contact_and_direct_vertices():
    source = Path("tools/isaac_sim/verify_left_table_collision.py").read_text()

    for required in (
        "TRIAL_COUNT = 3",
        "STRESS_DT = 1.0 / 60.0",
        "SHOULDER_START_DEG = -55.00394821166992",
        "SHOULDER_END_DEG = 20.0",
        "SHOULDER_STEP_DEG = 0.5",
        "HOLD_STEPS = 180",
        "PhysxContactReportAPI.Apply",
        "CreateThresholdAttr().Set(0)",
        "get_contact_report()",
        "datum.separation",
        "capture_viewport_to_file",
        "articulation._articulation_view.get_dof_limits()",
        "articulation._articulation_view.set_joint_position_targets(",
        "collision.HasAPI(UsdPhysics.CollisionAPI)",
        "GetLocalToWorldTransform",
        "GetPointsAttr",
        "table_from_world",
        "minimum_target_separation_m",
        "minimum_table_local_finger_z_m",
        "maximum_visual_collision_error_m",
        "PhysxSceneQuasistaticAPI.Apply",
        '"quasistatic_enabled": True',
        '"min_position_iteration_count": 64',
    ):
        assert required in source
    for forbidden in (
        "save_stage",
        "stage.Save",
        "contactOffset",
        "restOffset",
        "set_gains",
        "ComputeWorldBound",
        "TABLE_BOTTOM_Z_M",
        "BOTTOM_CROSSING_TOLERANCE_M",
    ):
        assert forbidden not in source
