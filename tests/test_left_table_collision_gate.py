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
        minimum_tip_z_m=-0.001,
        final_target_error_rad=math.radians(8),
        persistent_contact_steps=20,
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
    assert result["bottom_crossed"] is False
    assert result["infeasible_target_blocked"] is True


def test_unrelated_contact_and_bottom_crossing_fail():
    trial = passing_trial()
    trial.contact_pairs = [("/World/environment/worldBody/__1", TIP)]
    trial.minimum_tip_z_m = -0.017

    result = evaluate_trial(trial)

    assert result["status"] == "FAIL"
    assert "missing_exact_table_tip_contact" in result["failure_reasons"]
    assert "tested_collider_crossed_table_bottom" in result["failure_reasons"]


def test_exactly_three_passing_trials_are_required():
    assert aggregate_trials([evaluate_trial(passing_trial())] * 2)["status"] == "FAIL"
    assert (
        aggregate_trials([evaluate_trial(passing_trial()) for _ in range(3)])["status"]
        == "PASS"
    )


def test_runtime_verifier_has_fixed_inspector_stress_contract():
    source = Path("tools/isaac_sim/verify_left_table_collision.py").read_text()

    for required in (
        "TRIAL_COUNT = 3",
        "STRESS_DT = 1.0 / 60.0",
        "SHOULDER_START_DEG = -55.00394821166992",
        "SHOULDER_END_DEG = 20.0",
        "SHOULDER_STEP_DEG = 0.5",
        "HOLD_STEPS = 30",
        "PhysxContactReportAPI.Apply",
        "CreateThresholdAttr().Set(0)",
        "get_contact_report()",
        "capture_viewport_to_file",
        "articulation._articulation_view.get_dof_limits()",
        "articulation._articulation_view.set_joint_position_targets(",
        "prim.HasAPI(UsdPhysics.CollisionAPI)",
        '"collider_bounds_m"',
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
    ):
        assert forbidden not in source
