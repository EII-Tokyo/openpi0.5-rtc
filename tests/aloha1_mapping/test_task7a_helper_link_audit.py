from pathlib import Path

import pytest

from tools.aloha1_mapping.task7a_helper_link_audit import HELPER_SUFFIXES
from tools.aloha1_mapping.task7a_helper_link_audit import audit_urdf_helper_links
from tools.aloha1_mapping.task7a_helper_link_audit import classify_helper_semantics

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/audit_aloha1_task7a_helper_links.py"


@pytest.mark.parametrize("robot", ["follower_left", "follower_right"])
def test_generated_follower_helper_links_are_geometry_free_source_frames(
    robot: str,
) -> None:
    urdf = ROOT / "generated/urdf" / f"{robot}.urdf"

    records = audit_urdf_helper_links(urdf, robot)

    assert set(records) == {f"{robot}_{suffix}" for suffix in HELPER_SUFFIXES}
    for record in records.values():
        assert record["visual_count"] == 0
        assert record["collision_count"] == 0
        assert record["inertial_count"] == 1
        assert record["mass_kg"] == pytest.approx(0.001)
        assert record["invent_collider_allowed"] is False
        assert record["remove_rigid_body_api_allowed"] is False


@pytest.mark.parametrize("robot", ["follower_left", "follower_right"])
def test_helper_semantics_preserve_joint_frames_without_guessing_geometry(
    robot: str,
) -> None:
    records = audit_urdf_helper_links(
        ROOT / "generated/urdf" / f"{robot}.urdf",
        robot,
    )

    assert records[f"{robot}_ee_arm_link"]["semantic_class"] == ("VIRTUAL_KINEMATIC_HELPER")
    assert records[f"{robot}_fingers_link"]["semantic_class"] == ("VIRTUAL_KINEMATIC_HELPER")
    assert records[f"{robot}_ee_gripper_link"]["semantic_class"] == ("FIXED_FRAME_ALIAS")


def test_ambiguous_mass_bearing_link_is_not_reclassified_as_virtual() -> None:
    result = classify_helper_semantics(
        visual_count=0,
        collision_count=0,
        inertial_count=1,
        parent_joint_types=["revolute"],
        child_joint_types=[],
    )

    assert result == "MASS_BEARING_SOURCE_LINK_WITHOUT_COLLIDER"


def test_geometry_on_nominal_helper_is_inconclusive() -> None:
    result = classify_helper_semantics(
        visual_count=1,
        collision_count=0,
        inertial_count=1,
        parent_joint_types=["fixed"],
        child_joint_types=["prismatic"],
    )

    assert result == "INCONCLUSIVE"


def test_runtime_probe_is_read_only_and_frozen_to_approved_stage() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf" in source
    assert "GetPrimStack" in source
    assert "UsdPhysics.CollisionAPI" in source
    assert "stage_modified" in source
    assert ".Set(" not in source
    assert ".Apply(" not in source
    assert "CreateNew" not in source


def test_current_helper_report_pins_installed_isaac_5_1_rule_source() -> None:
    report = __import__("json").loads(
        (
            ROOT
            / "reports/aloha1_mapping/"
            "aloha1_task7a_helper_link_semantics.json"
        ).read_text(encoding="utf-8")
    )

    source = report["official_rule_source"]["installed_python_source"]
    assert source["absolute_path"].endswith(
        "isaacsim/asset/validation/physics_rules.py"
    )
    assert len(source["sha256"]) == 64
    assert source["class_name"] == "RigidBodyHasCollider"
    assert source["class_first_line"] > 0
