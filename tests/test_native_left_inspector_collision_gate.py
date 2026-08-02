from pathlib import Path

import pytest

from tools.isaac_sim.run_native_left_inspector_collision_gate import (
    aggregate_trial_reports,
    build_trial_launch,
)


TRIAL_SCRIPT = Path("tools/isaac_sim/native_left_inspector_collision_trial.py")
RUNNER_SCRIPT = Path("tools/isaac_sim/run_native_left_inspector_collision_gate.py")


def test_native_trial_uses_full_inspector_authoring_path_and_exact_stage():
    source = TRIAL_SCRIPT.read_text(encoding="utf-8")

    for required in (
        "/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/",
        "aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_diagnostic.usda",
        "165093c3e7bf359b2ef5dbb595feb4ed976b194844830e70f387d6b882c1d6f2",
        'LEFT_ARTICULATION_ROOT = "/World/follower_left/vx300s_left/root_joint"',
        'TABLE_COLLIDER = "/World/environment/worldBody/user_confirmed_table"',
        "INSPECTED_PATHS = (LEFT_ARTICULATION_ROOT, TABLE_COLLIDER)",
        "_expanded_inspected_paths",
        "prim.IsA(UsdPhysics.Joint)",
        "prim.HasAPI(UsdPhysics.RigidBodyAPI)",
        "prim.HasAPI(UsdPhysics.CollisionAPI)",
        'SHOULDER_JOINT = "/World/follower_left/vx300s_left/joints/shoulder"',
        "APPROACH_TARGET_DEG = 20.0",
        "HOLD_TARGET_DEG = 30.0",
        "PhysXInspectorModelControlType.JOINT_DRIVE",
        "get_enable_quasi_static_mode_model().set_value(True)",
        "get_fix_articulation_base_model().set_value(True)",
        "get_enable_gravity_model().set_value(False)",
        "PhysxContactReportAPI.Apply",
        "CreateThresholdAttr().Set(0)",
        "UsdPhysics.DriveAPI.Get",
        "omni.kit.commands.execute(",
        '"ChangeProperty"',
        "_sub_async_sim_run",
        "datum.separation",
        "_live_finger_geometry",
        "TrialMetrics",
        '"trial.json"',
        "post_quit",
        "EXPECTED_JOINT_ROWS = 13",
    ):
        assert required in source
    assert "add_inspector_window" not in source


def test_native_trial_is_disposable_and_never_changes_robot_or_stage():
    source = TRIAL_SCRIPT.read_text(encoding="utf-8")

    for forbidden in (
        "save_stage",
        "stage.Save",
        ".play(",
        "set_joint_position",
        "set_joint_value",
        "set_world_pose",
        "set_local_pose",
        "set_gains",
        "contactOffset",
        "restOffset",
        "dynamixel",
        "serial",
        "192.168.1.103",
    ):
        assert forbidden not in source


def test_runner_builds_full_kit_launch_with_unique_trial_environment(tmp_path):
    launch = build_trial_launch(tmp_path, trial_index=2)

    assert launch.command[0].endswith("/.venv_issac/bin/python")
    assert launch.command[1].endswith("/.venv_issac/bin/isaacsim")
    assert launch.command[2].endswith("/apps/isaacsim.exp.full.kit")
    assert launch.command[3] == "--exec"
    assert launch.command[4].endswith(
        "/tools/isaac_sim/native_left_inspector_collision_trial.py"
    )
    assert launch.environment["CODEX_NATIVE_TRIAL_INDEX"] == "2"
    assert launch.environment["CODEX_NATIVE_TRIAL_OUTPUT_DIR"] == str(
        tmp_path / "trial_02"
    )


def test_runner_requires_exactly_three_independent_passing_trials():
    passing = {
        "status": "PASS",
        "stage_sha256_after": (
            "165093c3e7bf359b2ef5dbb595feb4ed976b194844830e70f387d6b882c1d6f2"
        ),
        "stage_saved": False,
        "real_robot_touched": False,
    }

    aggregate = aggregate_trial_reports(
        [{**passing, "trial_index": index} for index in (1, 2, 3)]
    )

    assert aggregate["status"] == "PASS"
    assert aggregate["trial_count"] == 3

    with pytest.raises(ValueError, match="exactly three"):
        aggregate_trial_reports([{**passing, "trial_index": 1}])

    failed = [{**passing, "trial_index": index} for index in (1, 2, 3)]
    failed[1] = {**failed[1], "status": "FAIL"}
    assert aggregate_trial_reports(failed)["status"] == "FAIL"


def test_runner_terminates_only_completed_disposable_child_after_report():
    source = RUNNER_SCRIPT.read_text(encoding="utf-8")

    for required in (
        "subprocess.Popen(",
        "report_path.is_file()",
        "process.terminate()",
        "process.kill()",
        "controlled_termination",
    ):
        assert required in source


def test_native_trial_exercises_single_joint_inspector_callback():
    source = TRIAL_SCRIPT.read_text(encoding="utf-8")

    for required in (
        "_clear_transient_selection",
        "_inspector_panel._delegate_tree._on_value_changed",
        "target_change_is_isolated",
        '"drive_targets_before"',
        '"drive_targets_after_single_joint_edit"',
        '"single_joint_target_isolated"',
    ):
        assert required in source
