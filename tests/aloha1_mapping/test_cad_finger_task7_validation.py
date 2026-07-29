from __future__ import annotations

import json
from pathlib import Path

from tools.aloha1_mapping.cad_finger_task7 import classify_rigid_body_scope
from tools.aloha1_mapping.cad_finger_task7 import classify_task7
from tools.aloha1_mapping.cad_finger_task7 import deterministic_signature

ROOT = Path(__file__).resolve().parents[2]
REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task7_validation.json"
)
SCRIPT = ROOT / "tools/validate_aloha_viper_cad_finger_task7.py"
ROBOT_ASSET = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "supplier_cad_follower_left/1.6/supplier_cad_follower_left.usda"
)
ROBOT_SCHEMA_ASSET = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "supplier_cad_follower_left_robot_schema/1.2/"
    "supplier_cad_follower_left_robot_schema.usda"
)


def test_task7_classification_preserves_fail_and_partial() -> None:
    assert classify_task7([{"status": "PASS"}], []) == "PASS"
    assert classify_task7([{"status": "PASS"}], ["measurement"]) == "PARTIAL"
    assert (
        classify_task7([{"status": "FAIL"}], ["measurement"]) == "FAIL"
    )


def test_task7_signature_ignores_only_repeat_wrapper() -> None:
    report = {
        "status": "PARTIAL",
        "checks": [{"name": "dof_order", "status": "PASS"}],
        "repeat_determinism": {"pass": False},
    }
    first = deterministic_signature(report)
    report["repeat_determinism"] = {"pass": True}
    assert deterministic_signature(report) == first
    report["checks"][0]["status"] = "FAIL"
    assert deterministic_signature(report) != first


def test_rigid_body_scope_requires_joint_participation_evidence() -> None:
    assert classify_rigid_body_scope(
        path="/robot/camera_focus",
        rigid_body_enabled=True,
        joint_body_target=True,
        only_fixed_joint_targets=True,
        collider_count=0,
    ) == "FIXED_REFERENCE_HELPER_EXCLUDE_FROM_ROBOT_DIAGNOSTIC"
    assert classify_rigid_body_scope(
        path="/robot/base",
        rigid_body_enabled=True,
        joint_body_target=True,
        only_fixed_joint_targets=False,
        collider_count=0,
    ) == "PARTICIPATING_BODY_MISSING_COLLIDER_HARD_BLOCKER"
    assert classify_rigid_body_scope(
        path="/robot/link",
        rigid_body_enabled=True,
        joint_body_target=True,
        only_fixed_joint_targets=False,
        collider_count=1,
    ) == "ROBOT_RIGID_BODY"


def test_task7_script_is_scoped_to_approved_supplier_stage() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "aloha_viperx_supplier_cad_bottle_task5.usda" in source
    assert "supplier_cad_follower_left.usda" in source
    assert "aloha_viper_cad_finger_task5_bottle.json" in source
    assert "IsaacSim.PhysicsRules" in source
    assert "IsaacSim.RobotRules" in source
    assert "IsaacSim.SimReadyAssetRules" in source
    assert 'extension_id = "isaacsim.asset.validation"' in source
    assert "set_extension_enabled_immediate" in source
    assert "world.step(" not in source
    assert "SurfaceGripper" not in source
    assert "except Exception:" in source
    assert "traceback.print_exc()" in source
    assert "app.close()" in source


def test_task7_report_is_machine_readable_and_deterministic() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "PARTIAL"
    assert report["scope"] == (
        "SUPPLIER_CAD_FOLLOWER_LEFT_DIAGNOSTIC_ONLY"
    )
    assert report["task5_static_hold"]["status"] == "PASS"
    assert report["task5_static_hold"]["pass_count"] == 20
    assert report["task5_static_hold"]["maximum_drop_m"] == (
        0.0004539191722869873
    )
    assert report["robot_articulation_roots"] == [
        "/supplier_cad_follower_left/vx300s_left/vx300s_left"
    ]
    assert report["runtime_dof_order"] == [
        "vx300s_left_waist",
        "vx300s_left_shoulder",
        "vx300s_left_elbow",
        "vx300s_left_forearm_roll",
        "vx300s_left_wrist_angle",
        "vx300s_left_wrist_rotate",
        "vx300s_left_left_finger",
        "vx300s_left_right_finger",
    ]
    assert report["repeat_determinism"]["pass"] is True
    assert report["task8"] == "NOT_RUN"
    assert report["final_default_collider_modified"] is False
    assert report["angular_tessellation"]["status"] == "PASS"
    assert report["base_evidence"]["status"] == "PASS"
    assert "HARD_BLOCKER_PRODUCTION_ANGULAR_TESSELLATION" not in (
        report["hard_blockers"]
    )
    assert (
        "HARD_BLOCKER_ROBOT_BASE_BODY_MISSING_COLLIDER_AND_DYNAMICS_EVIDENCE"
        not in report["hard_blockers"]
    )
    assert report["validation_targets"] == {
        "IsaacSim.PhysicsRules": str(ROBOT_ASSET.resolve()),
        "IsaacSim.RobotRules": str(ROBOT_SCHEMA_ASSET.resolve()),
        "IsaacSim.SimReadyAssetRules": str(ROBOT_ASSET.resolve()),
    }
    assert {
        item["category"] for item in report["official_rules"]["categories"]
    } == {
        "IsaacSim.PhysicsRules",
        "IsaacSim.RobotRules",
        "IsaacSim.SimReadyAssetRules",
    }
    physics = next(
        item
        for item in report["official_rules"]["categories"]
        if item["category"] == "IsaacSim.PhysicsRules"
    )
    assert physics["blocking_issue_count"] == 0
    robot = next(
        item
        for item in report["official_rules"]["categories"]
        if item["category"] == "IsaacSim.RobotRules"
    )
    assert robot["blocking_issue_count"] == 0
