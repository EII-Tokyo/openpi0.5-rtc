from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT / "tools/build_aloha_viper_supplier_cad_follower_right_asset.py"
)
REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_supplier_cad_follower_right_asset.json"
)


def test_builder_uses_explicit_right_product_reference_not_left_rename() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "follower_right/follower_right.usd" in source
    assert 'Sdf.Path("/follower_right")' in source
    assert "AddReference" in source
    assert "supplier_cad_finger_mesh.usda" in source
    assert "BatchNamespaceEdit" not in source
    assert '.replace("follower_left", "follower_right")' not in source
    assert "ROBOT_LOCAL_DIAGNOSTIC_NOT_WORKCELL_PLACEMENT" in source


def test_generated_right_asset_has_one_resolved_articulation() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["scope"] == (
        "ROBOT_LOCAL_DIAGNOSTIC_NOT_WORKCELL_PLACEMENT"
    )
    assert report["root_prim"] == "/follower_right"
    assert report["robot_product_prim"] == "/follower_right/vx300s_right"
    assert report["articulation_roots"] == [
        "/follower_right/vx300s_right/root_joint"
    ]
    assert report["articulation_count"] == 1
    assert report["construction"]["method"] == (
        "EXPLICIT_REFERENCE_TO_VERSION_PINNED_FOLLOWER_RIGHT_PRODUCT_"
        "PLUS_SUPPLIER_CAD_FINGER_LAYERS"
    )
    assert report["construction"]["robot_geometry_mirrored"] is False
    assert report["construction"]["workcell_placement_authored"] is False


def test_generated_right_asset_preserves_relationships_and_dof_order() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["relationship_validation"]["invalid_joint_body_targets"] == []
    assert report["relationship_validation"]["invalid_robot_link_targets"] == []
    assert report["relationship_validation"]["invalid_robot_joint_targets"] == []
    assert report["relationship_validation"]["all_targets_under_robot_product"]
    assert report["dof_order"] == [
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
        "gripper",
        "left_finger",
        "right_finger",
    ]
    assert report["dof_count"] == 9
    assert all(
        path.startswith("/follower_right/vx300s_right/joints/")
        for path in report["dof_paths"]
    )


def test_generated_right_asset_uses_only_supplier_v2_handed_fingers() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["supplier_fingers"]["left_finger"]["source_obj_sha256"] == (
        "c6710d0fe5b2030a32722d9df5c0b553c771c9d61d92b8ddaec36c94c5963488"
    )
    assert report["supplier_fingers"]["right_finger"]["source_obj_sha256"] == (
        "b0979c5d55fee448dab512dc75b1251bab17d94892decd01de9a6e76c01482d1"
    )
    assert report["supplier_fingers"]["left_finger"]["cad_side"] == "+X"
    assert report["supplier_fingers"]["right_finger"]["cad_side"] == "-X"
    assert report["supplier_fingers"]["mirrored"] is False
    assert report["supplier_fingers"]["generic_856_face_active"] is False
    assert report["supplier_fingers"]["new_mesh_readback"] == {
        "left_finger": {"point_count": 831, "face_count": 1662},
        "right_finger": {"point_count": 831, "face_count": 1662},
    }


def test_generated_right_asset_preserves_all_protected_inputs() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["protected_inputs_unchanged"] is True
    assert report["source_right_asset"]["sha256"] == (
        "86d850cea5b35fb2969d3a78834317b51e2ac0d301f09aaaa9dad191f9bb3d5d"
    )
    assert report["source_right_asset"]["modified"] is False
    assert report["approved_left_review_stage"]["sha256"] == (
        "b24afe3678155654892c69517fc58ecd970108d68cec56b02dc6fdcb8bf4493e"
    )
    assert report["approved_left_review_stage"]["modified"] is False
    assert report["final_default_collider_modified"] is False
    assert report["hard_blockers"] == [
        "HARD_BLOCKER_FOLLOWER_RIGHT_WORKCELL_INSTALL_TRANSFORM"
    ]
    assert report["task8"] == "NOT_RUN"
