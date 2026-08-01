from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "tools/aloha1_mapping/cad_link_collision_semantics.py"
DRIVER = ROOT / "tools/audit_aloha1_cad_link_collision_semantics.py"
REPORT = (
    ROOT
    / "reports/aloha1_mapping/aloha1_cad_link_collision_semantics.json"
)
ALLOWED = {
    "PHYSICAL_CAD_DERIVABLE",
    "VIRTUAL_FRAME_NO_COLLIDER",
    "PHYSICAL_EXISTING_VALIDATED_COLLIDER",
    "HARD_BLOCKER_CAD_TO_LINK_IDENTITY",
}


def _load_module():
    assert MODULE.is_file()
    spec = importlib.util.spec_from_file_location(
        "cad_link_collision_semantics",
        MODULE,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_link_classifier_uses_only_the_approved_vocabulary() -> None:
    module = _load_module()

    assert module.ALLOWED_CLASSIFICATIONS == ALLOWED
    assert (
        module.classify_link(
            helper_semantic="VIRTUAL_KINEMATIC_HELPER",
            accepted_cad_finger=False,
            cad_object_name=None,
        )
        == "VIRTUAL_FRAME_NO_COLLIDER"
    )
    assert (
        module.classify_link(
            helper_semantic=None,
            accepted_cad_finger=True,
            cad_object_name="Part__Feature007",
        )
        == "PHYSICAL_EXISTING_VALIDATED_COLLIDER"
    )
    assert (
        module.classify_link(
            helper_semantic=None,
            accepted_cad_finger=False,
            cad_object_name="Part__Feature",
        )
        == "PHYSICAL_CAD_DERIVABLE"
    )
    assert (
        module.classify_link(
            helper_semantic=None,
            accepted_cad_finger=False,
            cad_object_name=None,
        )
        == "HARD_BLOCKER_CAD_TO_LINK_IDENTITY"
    )


def test_current_report_classifies_every_follower_link_without_inventing_geometry() -> None:
    assert DRIVER.is_file()
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "PARTIAL"
    assert report["unclassified_link_count"] == 0
    assert report["link_count"] == 28
    assert report["classification_counts"] == {
        "HARD_BLOCKER_CAD_TO_LINK_IDENTITY": 4,
        "PHYSICAL_CAD_DERIVABLE": 14,
        "PHYSICAL_EXISTING_VALIDATED_COLLIDER": 4,
        "VIRTUAL_FRAME_NO_COLLIDER": 6,
    }

    for record in report["links"]:
        assert record["classification"] in ALLOWED
        assert record["robot"] in {"follower_left", "follower_right"}
        assert record["urdf_link_name"].startswith(record["robot"])
        assert record["usd_prim_path"].startswith(f"/World/{record['robot']}/")
        assert record["unit_conversion_mm_to_m"] == 0.001
        assert record["evidence_paths"]
        assert "source_placement_matrix" in record
        assert "cad_to_link_matrix" in record
        assert "existing_collision_meshes" in record

    virtual = [
        record
        for record in report["links"]
        if record["classification"] == "VIRTUAL_FRAME_NO_COLLIDER"
    ]
    assert len(virtual) == 6
    assert {record["link_suffix"] for record in virtual} == {
        "ee_arm_link",
        "fingers_link",
        "ee_gripper_link",
    }
    for record in virtual:
        assert record["invent_collider_allowed"] is False
        assert record["visual_count"] == 0
        assert record["collision_count"] == 0
        assert record["cad_object"] is None

    accepted_fingers = [
        record
        for record in report["links"]
        if record["classification"]
        == "PHYSICAL_EXISTING_VALIDATED_COLLIDER"
    ]
    assert len(accepted_fingers) == 4
    for record in accepted_fingers:
        assert record["cad_to_link_matrix"] is not None
        assert record["transform_determinant"] == 1.0
        assert record["mirror_used"] is False

    derivable = [
        record
        for record in report["links"]
        if record["classification"] == "PHYSICAL_CAD_DERIVABLE"
    ]
    assert len(derivable) == 14
    for record in derivable:
        assert record["cad_object"] is not None
        assert record["source_placement_matrix"] is not None
        assert record["cad_to_link_matrix"] is None
        assert record["registration_status"] == (
            "PENDING_PHASE3_NUMERICAL_REGISTRATION"
        )

    hard_blockers = [
        record
        for record in report["links"]
        if record["classification"]
        == "HARD_BLOCKER_CAD_TO_LINK_IDENTITY"
    ]
    assert {record["link_suffix"] for record in hard_blockers} == {
        "gripper_prop_link",
        "gripper_bar_link",
    }


def test_helper_rigid_body_rule_is_not_closed_with_fake_colliders() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["helper_frame_decision"] == (
        "DO_NOT_INVENT_COLLIDER_AND_DO_NOT_REMOVE_RIGIDBODY_WITHOUT_SEPARATE_REGRESSION"
    )
    assert report["final_or_default_asset_modified"] is False
    assert report["task8"] == "NOT_RUN"
