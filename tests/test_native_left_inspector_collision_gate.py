from pathlib import Path


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
        'SHOULDER_JOINT = "/World/follower_left/vx300s_left/joints/shoulder"',
        "PhysXInspectorModelControlType.JOINT_DRIVE",
        "get_enable_quasi_static_mode_model().set_value(True)",
        "get_fix_articulation_base_model().set_value(True)",
        "get_enable_gravity_model().set_value(False)",
        "PhysxContactReportAPI.Apply",
        "CreateThresholdAttr().Set(0)",
        "UsdPhysics.DriveAPI.Get",
        'omni.kit.commands.execute("ChangeProperty"',
        "_sub_async_sim_run",
        "datum.separation",
        "_live_finger_geometry",
        "TrialMetrics",
        '"trial.json"',
        "post_quit",
        "EXPECTED_JOINT_ROWS = 13",
    ):
        assert required in source


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
