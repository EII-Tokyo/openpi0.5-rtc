from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "tools/build_aloha1_cad_derived_collider_diagnostic_stage.py"
ASSET_ROOT = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "cad_derived_full_body_colliders/1.0"
)
ROOT_LAYER = ASSET_ROOT / "aloha1_cad_derived_full_body_collider_diagnostic.usda"
GEOMETRY_LAYER = ASSET_ROOT / "geometry/cad_derived_colliders.usda"
PHYSICS_LAYER = ASSET_ROOT / "physics/cad_derived_colliders_physics.usda"
REPORT = ROOT / "reports/aloha1_mapping/aloha1_cad_derived_collider_stage.json"
NATIVE_PROBE = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha1_cad_derived_collider_stage_native_probe.json"
)


def test_diagnostic_stage_is_isolated_and_uses_only_frozen_inputs() -> None:
    assert BUILDER.is_file()
    assert ROOT_LAYER.is_file()
    assert GEOMETRY_LAYER.is_file()
    assert PHYSICS_LAYER.is_file()

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "PARTIAL"
    assert report["source_stage"]["sha256_before"] == (
        "d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf"
    )
    assert report["source_stage"]["sha256_after"] == report["source_stage"]["sha256_before"]
    assert report["source_hashes_unchanged"] is True
    assert report["source_or_imported_asset_modified"] is False
    assert report["final_or_default_asset_modified"] is False
    assert report["drive_modified"] is False
    assert report["material_modified"] is False
    assert report["timestep_modified"] is False
    assert report["task8"] == "NOT_RUN"

    assert report["root_layer"]["absolute_path"] == str(ROOT_LAYER.resolve())
    assert report["root_layer"]["default_prim"] == "/World"
    assert report["root_layer"]["sublayers"] == [
        "physics/cad_derived_colliders_physics.usda",
        "geometry/cad_derived_colliders.usda",
        (
            "../../signal_correspondence/1.0/"
            "aloha1_signal_correspondence_workcell.usda"
        ),
    ]
    assert report["stage_open_probe"] == "PASS"
    native = json.loads(NATIVE_PROBE.read_text(encoding="utf-8"))
    assert native["status"] == "PASS"
    assert native["runtime"] == {
        "isaac_sim": "5.1.0.0",
        "kit": "107.3.3",
        "physx": "107.3.26",
    }
    assert native["stage"]["sha256_before"] == report["root_layer"]["sha256"]
    assert native["stage"]["sha256_after"] == native["stage"]["sha256_before"]
    assert native["new_collider_count"] == 12
    assert native["all_new_colliders_convex_hull"] is True


def test_composed_stage_has_exactly_the_supported_new_colliders() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    colliders = report["new_collider_readback"]
    assert len(colliders) == 12
    assert len({record["prim_path"] for record in colliders}) == 12
    assert all(record["collision_enabled"] is True for record in colliders)
    assert all(record["approximation"] == "convexHull" for record in colliders)
    assert all(record["type_name"] == "Mesh" for record in colliders)
    assert all(record["purpose"] == "guide" for record in colliders)
    assert all(record["point_count"] > 0 for record in colliders)
    assert all(record["face_count"] > 0 for record in colliders)

    assert report["existing_finger_collider_readback"]["count"] == 4
    assert report["existing_finger_collider_readback"]["all_convex_hull"] is True
    assert report["virtual_frame_collider_count"] == 0
    assert report["blocked_link_colliders_authored"] == []
    assert report["source_instance_proxy_collider_count_before"] == 18
    assert report["deactivated_source_collision_instances"]["count"] == 14
    assert report["deactivated_source_collision_instances"]["all_inactive"] is True
    assert report["baseline_fallback_collider_readback"]["count"] == 4
    assert report["baseline_fallback_collider_readback"]["all_convex_hull"] is True
    assert report["baseline_fallback_collider_readback"]["all_enabled"] is True
    assert report["duplicate_source_candidate_collision_gate"] == "PASS"
    assert set(report["blocked_physical_links"]) == {
        "follower_left_wrist_link",
        "follower_right_wrist_link",
        "follower_left_gripper_prop_link",
        "follower_right_gripper_prop_link",
    }
    assert report["gripper_bar_fixed_group_coverage"]["count"] == 2


def test_articulation_and_control_composition_are_unchanged() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["articulation_count"] == 2
    assert report["articulation_root_paths"] == [
        "/World/follower_left/vx300s_left/root_joint",
        "/World/follower_right/vx300s_right/root_joint",
    ]
    assert report["joint_signature_unchanged"] is True
    assert report["rigid_body_signature_unchanged"] is True
    assert report["required_prims_valid"] is True
    assert report["collision_authoring_scope"] == (
        "NEW_DIAGNOSTIC_MESH_PRIMS_PLUS_SOURCE_INSTANCE_DEACTIVATION"
    )
