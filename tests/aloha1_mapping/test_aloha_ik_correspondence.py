from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/aloha1_mapping/aloha1_ik_correspondence_v3.json"
FROZEN_ACCEPTED_VIDEO_REPORT = (
    ROOT / "reports/aloha1_mapping/aloha1_ik_correspondence_v2.json"
)
KINEMATICS_REPORT = ROOT / "reports/aloha1_mapping/aloha1_task7b2_horizontal_kinematics_v2.json"
CONFIG = ROOT / "configs/aloha1_task7b2_horizontal_grasp.yaml"
TOOL = ROOT / "tools/validate_aloha1_aloha_ik_correspondence.py"
EXPECTED_ARM_ORDER = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_report() -> dict:
    assert REPORT.is_file(), f"missing report: {REPORT}"
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_correspondence_tool_and_versioned_reports_exist() -> None:
    assert TOOL.is_file()
    assert KINEMATICS_REPORT.is_file()
    assert FROZEN_ACCEPTED_VIDEO_REPORT.is_file()


def test_v2_report_remains_frozen_for_user_accepted_video_chain() -> None:
    assert _sha256(FROZEN_ACCEPTED_VIDEO_REPORT) == (
        "6b9af0569b2e1cb829da208b69e36c18fe0dd2ba1d22b12e42b84dc625c279f9"
    )


def test_config_freezes_passing_grasp_editor_and_coupling_inputs() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    frozen = config["frozen_inputs"]
    assert frozen["grasp_editor_v2_semantics"]["required_status"] == "PASS"
    assert frozen["gripper_coupling_ab"]["required_status"] == "PASS"
    assert frozen["gripper_coupling_ab"]["required_classification"] == "PHYSX_MIMIC_PRIMARY"
    assert frozen["gripper_coupling_ab"]["promotion_authorized"] is False
    assert frozen["grasp_editor_v2_native_raw_yaml"]["sha256"] == (
        "fa1270a547e0ac89da2e7afc3965b1d6e7bcc34aab46005d6c65f977e3f69e5e"
    )
    assert frozen["grasp_editor_v2_derived_yaml"]["sha256"] == (
        "a861d2d1f072006f3027bb630c4f7c045a17d0c53020e8bd35d7f64ceeb3c2c5"
    )


def test_correspondence_uses_explicit_six_dof_order_and_all_joint_samples() -> None:
    report = _load_report()
    assert report["status"] == "PASS"
    assert report["aloha_6dof_correspondence"] == "PASS"
    assert report["ik"] == "PASS"
    assert report["joint_order"] == EXPECTED_ARM_ORDER
    assert report["finger_dofs_excluded_from_ik"] == ["gripper", "left_finger", "right_finger"]
    assert report["joint_order_policy"] == "EXPLICIT_SOURCE_ORDER_NOT_ALPHABETICAL"

    cases = report["fk_correspondence"]["cases"]
    assert len(cases) == 13
    assert cases[0]["case"] == "approved_home"
    expected = {f"{joint}_{direction}" for joint in EXPECTED_ARM_ORDER for direction in ("negative", "positive")}
    assert {case["case"] for case in cases[1:]} == expected
    assert all(case["status"] == "PASS" for case in cases)


def test_correspondence_binds_current_inputs_and_preserves_stage() -> None:
    report = _load_report()
    bindings = report["bindings"]
    for name, record in bindings.items():
        if name == "episode18":
            continue
        path = Path(record["path"])
        assert path.is_file(), (name, path)
        assert _sha256(path) == record["sha256"], name
    assert bindings["stage"]["sha256"] == ("2b3f76365ed67532f478d995ae859a88b5639975ac07cb7ac8a53ac679e8205c")
    assert bindings["stage"]["sha256_before"] == bindings["stage"]["sha256_after"]
    assert report["diagnostic_coupling"]["promotion_authorized"] is False
    assert report["diagnostic_coupling"]["classification"] == "PHYSX_MIMIC_PRIMARY"
    assert report["task8"] == "NOT_RUN"


def test_correspondence_residuals_and_transforms_meet_existing_gates() -> None:
    report = _load_report()
    residuals = report["fk_correspondence"]["max_residuals"]
    assert residuals["translation_m"] <= report["gates"]["translation_m"] == 0.001
    assert residuals["rotation_rad"] <= report["gates"]["rotation_rad"] == 0.005

    transforms = report["transform_contract"]
    assert transforms["status"] == "PASS"
    assert transforms["units"] == "m"
    assert transforms["world_origin"] == "TABLETOP_CENTER"
    assert transforms["grasp_editor_closure_error"] <= 1e-12
    assert all(abs(value - 1.0) <= 1e-9 for value in transforms["determinants"].values())


def test_horizontal_waypoints_are_verified_and_not_upright_legacy() -> None:
    report = _load_report()
    geometry = report["horizontal_geometry"]
    assert geometry["status"] == "PASS"
    assert geometry["task_geometry"] == "HORIZONTAL_DYNAMIC_TABLE_SUPPORTED"
    assert abs(geometry["axis_to_table_normal_deg"] - 90.0) <= 1.0
    assert abs(geometry["gripper_line_to_axis_deg"] - 90.0) <= 3.0
    assert geometry["approach_direction_world"] == [0.0, 0.0, -1.0]
    assert geometry["lift_direction_world"] == [0.0, 0.0, 1.0]
    assert report["waypoint_validation"]["status"] == "PASS"
    assert report["waypoint_validation"]["phases"] == [
        "move_to_pregrasp",
        "vertical_descent",
        "vertical_lift",
    ]
    assert report["dynamic_horizontal_bottle_grasp"] == "NOT_RUN"
