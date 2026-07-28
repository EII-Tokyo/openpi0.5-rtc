from __future__ import annotations

import math
import json
from pathlib import Path

from tools.aloha1_mapping.cad_finger_diagnostic import (
    DIAGNOSTIC_COLLISION_POLICY,
)
from tools.aloha1_mapping.cad_finger_diagnostic import build_mesh_payload

ROOT = Path(__file__).resolve().parents[2]
MESH_ROOT = (
    ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
    "viper_gripper/tessellation_angular_controlled/run_a"
)


def test_supplier_cad_meshes_map_to_separate_handed_link_sides() -> None:
    left = build_mesh_payload("left", MESH_ROOT / "left_finger.obj")
    right = build_mesh_payload("right", MESH_ROOT / "right_finger.obj")

    assert left["point_count"] == 831
    assert right["point_count"] == 831
    assert left["triangle_count"] == 1662
    assert right["triangle_count"] == 1662
    # Link-local geometry straddles zero.  At the legal closed qpos the
    # supplier left/right pair must reconstruct on Stage +Y/-Y respectively.
    assert left["closed_gripper_aabb_m"]["y_min"] > 0.0
    assert right["closed_gripper_aabb_m"]["y_max"] < 0.0
    for axis in ("x", "y", "z"):
        assert math.isclose(
            left["aabb_m"][f"{axis}_size"],
            right["aabb_m"][f"{axis}_size"],
            rel_tol=0.0,
            abs_tol=2.0e-6,
        )


def test_diagnostic_asset_never_promotes_cad_visual_to_collider() -> None:
    assert DIAGNOSTIC_COLLISION_POLICY == {
        "source_collision_branches": "UNCHANGED",
        "cad_mesh_role": "VISUAL_ONLY",
        "new_collision_api_applied": False,
        "final_default_collider_modified": False,
    }


def test_saved_diagnostic_asset_passes_visual_only_composition_gates() -> None:
    report = json.loads(
        (
            ROOT
            / "reports/aloha1_mapping/"
            "aloha_viper_cad_finger_diagnostic_asset_v2.json"
        ).read_text(encoding="utf-8")
    )

    assert report["status"] == "PASS"
    assert all(report["gates"].values())
    assert report["collision_inventory"]["difference"] == []
    assert report["forbidden_configuration_specs"] == []
    assert report["source_stage"]["sha256_before"] == report[
        "source_stage"
    ]["sha256_after"]
    assert report["mapping"]["linear_determinant"] == 1.0
    assert report["mapping"]["mirror_used"] is False
    for record in report["visual_records"].values():
        assert record["visuals_is_instance"] is False
        assert record["old_visual_active"] is False
        assert record["mesh_is_instance_proxy"] is False
        assert record["mesh_has_collision_api"] is False
        assert record["mesh_has_rigid_body_api"] is False
