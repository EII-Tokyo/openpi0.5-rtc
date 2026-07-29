from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/validate_aloha_viper_follower_right_task7.py"
REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_follower_right_task7_validation.json"
)
AGGREGATE = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_task7_aggregate_validation.json"
)


def _load() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_right_task7_validator_is_robot_local_and_gateway_verified() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "supplier_cad_follower_right.usda" in source
    assert "ROBOT_LOCAL_DIAGNOSTIC_NOT_WORKCELL_PLACEMENT" in source
    assert "MCPJUNGLE_NVIDIA_OFFICIAL_API_VERIFIED" in source
    assert "192.168.1.103" not in source
    assert "SurfaceGripper" not in source


def test_right_task7_keeps_runtime_failure_and_workcell_boundary() -> None:
    report = _load()

    assert report["status"] in {"FAIL", "PARTIAL"}
    assert report["scope"] == (
        "ROBOT_LOCAL_DIAGNOSTIC_NOT_WORKCELL_PLACEMENT"
    )
    assert report["robot_local"]["arm_one_joint"] == "PASS"
    assert report["robot_local"]["gripper_motion_direction"] == "PASS"
    assert report["robot_local"]["aperture_monotonicity"] == "PASS"
    assert report["robot_local"]["mimic_accuracy"] == "FAIL"
    assert report["robot_local"]["overall"] == "PARTIAL"
    assert report["dual_arm_workcell_placement"]["status"] == "PARTIAL"
    assert report["dual_arm_workcell_placement"]["verified"] is False
    assert report["hard_blockers"] == [
        "HARD_BLOCKER_FOLLOWER_RIGHT_WORKCELL_INSTALL_TRANSFORM"
    ]
    assert report["task8"] == "NOT_RUN"


def test_right_task7_runs_all_official_rule_categories_twice() -> None:
    report = _load()
    categories = {
        item["category"]: item
        for item in report["official_rules"]["categories"]
    }

    assert set(categories) == {
        "IsaacSim.PhysicsRules",
        "IsaacSim.RobotRules",
        "IsaacSim.SimReadyAssetRules",
    }
    assert report["repeat_determinism"]["run_count"] == 2
    assert report["repeat_determinism"]["fresh_stage_open_each_run"] is True
    assert report["repeat_determinism"]["pass"] is True
    assert all(
        item["status"] in {"PASS", "FAIL", "PARTIAL"}
        for item in categories.values()
    )


def test_right_task7_screenshot_evidence_is_auxiliary() -> None:
    report = _load()
    screenshot = report["robot_local"]["screenshot_evidence"]

    assert screenshot["status"] == "PASS"
    assert screenshot["raw_count"] == 7
    assert screenshot["annotated_count"] == 7
    assert screenshot["auxiliary_only"] is True
    assert screenshot["numeric_runtime_authoritative"] is True


def test_task7_aggregate_separates_robot_local_and_workcell_results() -> None:
    report = json.loads(AGGREGATE.read_text(encoding="utf-8"))

    assert report["status"] == "FAIL"
    assert report["follower_left"]["status"] == "PARTIAL"
    assert report["follower_right_robot_local"]["status"] == "FAIL"
    assert report["follower_right_robot_local"]["cad_available"] is True
    assert (
        report["dual_arm_workcell_placement"]["status"] == "PARTIAL"
    )
    assert report["dual_arm_workcell_placement"]["verified"] is False
    assert (
        "HARD_BLOCKER_APPROVED_STAGE_CONTAINS_FOLLOWER_LEFT_ONLY"
        not in report["hard_blockers"]
    )
    assert (
        "HARD_BLOCKER_FOLLOWER_RIGHT_WORKCELL_INSTALL_TRANSFORM"
        in report["hard_blockers"]
    )
    assert report["task8"] == "NOT_RUN"
