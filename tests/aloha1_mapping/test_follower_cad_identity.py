from __future__ import annotations

import json
from pathlib import Path

from tools.aloha1_mapping.follower_cad_identity import classify_follower_cad_identity
from tools.aloha1_mapping.follower_cad_identity_report import canonicalize_follower_urdf

ROOT = Path(__file__).resolve().parents[2]
REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_follower_left_right_cad_identity.json"
)
SCRIPT = ROOT / "tools/audit_aloha_viper_follower_cad_identity.py"


def _common_evidence() -> dict[str, object]:
    return {
        "follower_models": {
            "follower_left": "aloha_vx300s",
            "follower_right": "aloha_vx300s",
        },
        "follower_xacro_paths": {
            "follower_left": "descriptions/aloha_vx300s.urdf.xacro",
            "follower_right": "descriptions/aloha_vx300s.urdf.xacro",
        },
        "normalized_urdf_equal": True,
        "supplier_sales_identity": "pair of ViperX 300 6DOF arms",
    }


def test_single_complete_viper_product_is_verified_reusable() -> None:
    cad = {
        "root_products": [
            {
                "name": "Dummy_Aloha_VX_v3",
                "label": "Dummy Aloha VX v3",
                "complete_viper_product": True,
                "shape_valid": True,
                "placement_determinant": 1.0,
                "mirror": False,
            }
        ],
        "product_instances": [],
        "handed_finger_pair_verified": True,
        "gripper_assembly_semantics_verified": True,
    }
    result = classify_follower_cad_identity(cad, _common_evidence())
    assert result["classification"] == (
        "VERIFIED_SINGLE_REUSABLE_ROBOT_PRODUCT"
    )
    assert result["robot_local_identity_verified"] is True
    assert result["workcell_placement_verified"] is False


def test_two_instances_require_same_product_and_nonmirrored_placements() -> None:
    cad = {
        "root_products": [],
        "product_instances": [
            {
                "name": "left",
                "source_product": "viper",
                "geometry_signature": "same",
                "placement_determinant": 1.0,
                "mirror": False,
            },
            {
                "name": "right",
                "source_product": "viper",
                "geometry_signature": "same",
                "placement_determinant": 1.0,
                "mirror": False,
            },
        ],
        "handed_finger_pair_verified": True,
        "gripper_assembly_semantics_verified": True,
    }
    result = classify_follower_cad_identity(cad, _common_evidence())
    assert result["classification"] == (
        "VERIFIED_IDENTICAL_ROBOT_PRODUCT_INSTANCES"
    )


def test_different_products_are_not_collapsed_by_visual_similarity() -> None:
    cad = {
        "root_products": [],
        "product_instances": [
            {
                "name": "left",
                "source_product": "viper",
                "geometry_signature": "viper-shape",
                "placement_determinant": 1.0,
                "mirror": False,
            },
            {
                "name": "right",
                "source_product": "widow",
                "geometry_signature": "widow-shape",
                "placement_determinant": 1.0,
                "mirror": False,
            },
        ],
        "handed_finger_pair_verified": True,
        "gripper_assembly_semantics_verified": True,
    }
    result = classify_follower_cad_identity(cad, _common_evidence())
    assert result["classification"] == "DIFFERENT_LEFT_RIGHT_PRODUCTS"


def test_model_or_urdf_mismatch_is_inconclusive_for_single_product() -> None:
    evidence = _common_evidence()
    evidence["normalized_urdf_equal"] = False
    cad = {
        "root_products": [
            {
                "name": "viper",
                "complete_viper_product": True,
                "shape_valid": True,
                "placement_determinant": 1.0,
                "mirror": False,
            }
        ],
        "product_instances": [],
        "handed_finger_pair_verified": True,
        "gripper_assembly_semantics_verified": True,
    }
    result = classify_follower_cad_identity(cad, evidence)
    assert result["classification"] == "INCONCLUSIVE"


def test_side_prefixed_urdfs_canonicalize_to_the_same_robot_product() -> None:
    left = """
    <robot name="follower_left">
      <link name="follower_left/base_link"/>
      <joint name="follower_left/waist" type="revolute"/>
    </robot>
    """
    right = """
    <robot name="follower_right">
      <link name="follower_right/base_link"/>
      <joint name="follower_right/waist" type="revolute"/>
    </robot>
    """
    assert canonicalize_follower_urdf(left, "follower_left") == (
        canonicalize_follower_urdf(right, "follower_right")
    )


def test_generated_identity_report_preserves_scope_and_blocker_boundary() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "PARTIAL"
    assert report["classification"] == (
        "VERIFIED_SINGLE_REUSABLE_ROBOT_PRODUCT"
    )
    assert report["robot_local_identity_verified"] is True
    assert report["workcell_placement_verified"] is False
    assert report["cad_product_inventory"]["root_product_count"] == 1
    assert report["cad_product_inventory"]["instance_count"] == 0
    assert report["cad_product_inventory"]["step_product_record_count"] == 9
    assert report["urdf_identity"]["normalized_equal"] is True
    assert (
        report["urdf_identity"]["normalized_sha256"]["follower_left"]
        == report["urdf_identity"]["normalized_sha256"]["follower_right"]
    )
    assert report["toolchain"]["freecad_version"] == "1.1.1"
    assert report["toolchain"]["opencascade_version"] == "7.8.1"
    assert report["toolchain"]["linear_deflection_mm"] == 0.2
    assert report["toolchain"]["angular_deflection_deg"] == 20.0
    assert report["supplier_fingers"]["left_finger"]["obj_sha256"].startswith(
        "c6710d0fe5b2030a"
    )
    assert report["supplier_fingers"]["right_finger"]["obj_sha256"].startswith(
        "b0979c5d55fee448"
    )
    assert report["brep_validity"]["status"] == "PARTIAL"
    assert set(report["brep_validity"]["invalid_object_names"]) == {
        "Dummy_Aloha_VX_v3",
        "Part__Feature005",
    }
    assert report["hard_blockers"] == [
        "HARD_BLOCKER_FOLLOWER_RIGHT_WORKCELL_INSTALL_TRANSFORM"
    ]
    assert report["license"]["status"] == "UNKNOWN_HARD_BLOCKER"
    assert report["task8"] == "NOT_RUN"


def test_identity_report_script_uses_only_frozen_local_evidence() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "freecad_identity_audit.json" in source
    assert "aloha1_xacro_args.yaml" in source
    assert "follower_left.urdf" in source
    assert "follower_right.urdf" in source
    assert "aloha_purchased_model_identification.json" in source
    assert "final_fresh_tessellation/manifest.json" in source
