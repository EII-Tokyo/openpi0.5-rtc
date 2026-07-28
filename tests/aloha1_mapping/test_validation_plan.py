import json
from pathlib import Path

from tools.aloha1_mapping.validation import build_validation_plan
from tools.aloha1_mapping.validation import classify_validation
from tools.aloha1_mapping.validation import load_required_machine_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_validation_plan_has_explicit_articulations_and_dof_order() -> None:
    plan = build_validation_plan(PROJECT_ROOT)

    assert plan["expected_articulation_count"] == 2
    assert plan["robots"][0]["articulation_prim"] == ("/aloha1_workcell/Robots/follower_left/root_joint")
    assert plan["robots"][0]["dof_order"][-3:] == [
        "gripper",
        "left_finger",
        "right_finger",
    ]
    assert plan["official_rule_categories"] == [
        "IsaacSim.PhysicsRules",
        "IsaacSim.RobotRules",
        "IsaacSim.SimReadyAssetRules",
    ]
    assert Path(plan["gripper_validation_report"]) == (PROJECT_ROOT / "reports/aloha1_mapping/gripper_validation.json")
    assert plan["required_task5_gripper_statuses"] == ["PASS", "PARTIAL"]


def test_validation_classification_is_never_vague() -> None:
    assert classify_validation([{"status": "PASS"}], []) == "PASS"
    assert classify_validation([{"status": "PASS"}], ["calibration"]) == "PARTIAL"
    assert classify_validation([{"status": "FAIL", "name": "dof_order"}], ["calibration"]) == "FAIL"


def test_required_machine_report_preserves_exact_status(tmp_path: Path) -> None:
    report_path = tmp_path / "gripper.json"
    report_path.write_text(
        json.dumps({"schema_version": 1, "status": "PARTIAL"}),
        encoding="utf-8",
    )

    check = load_required_machine_report(
        report_path,
        name="Task5.GripperValidation",
        accepted_statuses=["PASS", "PARTIAL"],
    )

    assert check["status"] == "PASS"
    assert check["evidence"]["reported_status"] == "PARTIAL"
    assert check["evidence"]["report"]["status"] == "PARTIAL"


def test_missing_or_failed_required_machine_report_fails(
    tmp_path: Path,
) -> None:
    missing = load_required_machine_report(
        tmp_path / "missing.json",
        name="Task5.GripperValidation",
        accepted_statuses=["PASS", "PARTIAL"],
    )
    assert missing["status"] == "FAIL"

    report_path = tmp_path / "gripper.json"
    report_path.write_text(
        json.dumps({"schema_version": 1, "status": "FAIL"}),
        encoding="utf-8",
    )
    failed = load_required_machine_report(
        report_path,
        name="Task5.GripperValidation",
        accepted_statuses=["PASS", "PARTIAL"],
    )
    assert failed["status"] == "FAIL"
