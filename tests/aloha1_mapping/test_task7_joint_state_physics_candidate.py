import json
from pathlib import Path

from tools.build_aloha1_task7_joint_state_physics_candidates import _should_normalize_text_layer

ROOT = Path(__file__).resolve().parents[2]
REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha1_task7_joint_state_physics_candidate.json"
)
BUILDER = ROOT / "tools/build_aloha1_task7_joint_state_physics_candidates.py"


def test_only_explicit_usda_layers_are_text_normalized() -> None:
    assert _should_normalize_text_layer(Path("wrapper.usda")) is True
    assert _should_normalize_text_layer(Path("robot_physics.usd")) is False


def test_joint_state_candidate_closes_only_the_packaging_omission() -> None:
    assert BUILDER.is_file()
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "PASS"
    assert report["task7"] == "PARTIAL"
    assert report["task8"] == "NOT_RUN"
    assert report["final_or_default_asset_modified"] is False
    for name in ("follower_left", "follower_right"):
        candidate = report["candidates"][name]
        assert candidate["joint_path"].endswith("/joints/gripper")
        assert candidate["joint_type"] == "RevoluteJoint"
        assert candidate["joint_state_axis"] == "angular"
        assert candidate["joint_state_api_readback"] is True
        assert candidate["authored_state_values"] is False
        assert candidate["authored_drive_values"] is False
        assert candidate["source_stage_modified"] is False
        assert candidate["baseline"]["blocking_issue_count"] == 5
        assert candidate["official_rules"]["blocking_issue_count"] == 4
        assert candidate["official_rules"]["warning_count"] == 0
        assert candidate["official_rules"]["deterministic_repeat"] is True
        assert candidate["removed_issue_rules"] == ["JointHasJointStateAPI"]
        assert candidate["remaining_issue_rule_counts"] == {
            "MimicAPICheck": 1,
            "RigidBodyHasCollider": 3,
        }
