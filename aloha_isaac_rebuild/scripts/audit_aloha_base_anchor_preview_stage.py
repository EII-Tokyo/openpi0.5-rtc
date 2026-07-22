#!/usr/bin/env python3
"""Audit the A10 visual-only robot/base-anchor measurement checkpoint stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pxr import Usd, UsdGeom

from audit_aloha_physical_layout_preview_stage import EXPECTED_A9_CUBES
from audit_aloha_physical_layout_preview_stage import EXPECTED_DERIVED_FROM_MEASURED_PATHS
from audit_aloha_physical_layout_preview_stage import EXPECTED_MEASURED_SOURCE_PATHS
from audit_aloha_physical_layout_preview_stage import audit as audit_a9
from audit_aloha_support_frame_components_stage import EXPECTED_DEFAULT_PRIM


BASE_ANCHOR_ROOT = "/aloha/base_anchor"
EXPECTED_A10_CUBES = {
    "/aloha/base_anchor/visuals/measured_base_y_band/shared_y_band_not_full_footprint",
    "/aloha/base_anchor/visuals/measured_base_y_band/near_cam_low_measured_y_edge",
    "/aloha/base_anchor/visuals/measured_base_y_band/near_cam_high_measured_y_edge",
    "/aloha/base_anchor/visuals/x_extent_measurement_zones/left_base_x_extent_missing_zone",
    "/aloha/base_anchor/visuals/x_extent_measurement_zones/right_base_x_extent_missing_zone",
    "/aloha/base_anchor/visuals/direction_hints/base_y_band_center_marker",
    "/aloha/base_anchor/visuals/direction_hints/support_frame_x_direction_hint",
    "/aloha/base_anchor/visuals/direction_hints/measured_base_y_depth_direction_hint",
}
EXPECTED_A10_MEASURED_PATHS = {
    "/aloha/base_anchor/visuals/measured_base_y_band/shared_y_band_not_full_footprint",
    "/aloha/base_anchor/visuals/measured_base_y_band/near_cam_low_measured_y_edge",
    "/aloha/base_anchor/visuals/measured_base_y_band/near_cam_high_measured_y_edge",
}
EXPECTED_A10_PENDING_PATHS = {
    "/aloha/base_anchor/visuals/x_extent_measurement_zones/left_base_x_extent_missing_zone",
    "/aloha/base_anchor/visuals/x_extent_measurement_zones/right_base_x_extent_missing_zone",
}
EXPECTED_A10_DERIVED_PATHS = {
    "/aloha/base_anchor/visuals/direction_hints/base_y_band_center_marker",
}
EXPECTED_MISSING_SCOPES = {
    "/aloha/base_anchor/missing_measurements/left_base_x_min_m",
    "/aloha/base_anchor/missing_measurements/left_base_x_max_m",
    "/aloha/base_anchor/missing_measurements/right_base_x_min_m",
    "/aloha/base_anchor/missing_measurements/right_base_x_max_m",
    "/aloha/base_anchor/missing_measurements/exact_left_base_anchor_frame",
    "/aloha/base_anchor/missing_measurements/exact_right_base_anchor_frame",
    "/aloha/base_anchor/missing_measurements/base_yaw_or_skew",
    "/aloha/base_anchor/missing_measurements/base_height_relative_to_table",
}
FORBIDDEN_TYPE_COUNTS = [
    "PhysicsScene",
    "PhysicsFixedJoint",
    "PhysicsRevoluteJoint",
    "PhysicsPrismaticJoint",
    "RenderProduct",
]
FORBIDDEN_API_MARKERS = [
    "PhysicsRigidBodyAPI",
    "PhysicsCollisionAPI",
    "PhysicsMassAPI",
    "PhysicsArticulationRootAPI",
    "PhysxSchemaPhysxSceneAPI",
]
FORBIDDEN_TEXT = [
    "stationary_ai.usd",
    "aloha2_menagerie_scene_deep_black_real_start_pose",
    "/scene/worldBody",
    "/scene/StartupViewCamera",
    "PhysicsScene",
    "CollisionAPI",
    "RigidBodyAPI",
    "MassAPI",
    "ArticulationRootAPI",
    "RenderProduct",
    "hdf5",
    "controller",
    "ros_bridge_publish = true",
    "openpi_runtime_capture = true",
]


def _custom_attrs(prim: Usd.Prim) -> dict:
    return {
        attr.GetName(): attr.Get()
        for attr in prim.GetAuthoredAttributes()
        if attr.GetName().startswith("aloha:")
    }


def audit(stage_path: Path) -> dict:
    text = stage_path.read_text(encoding="utf-8")
    stage = Usd.Stage.Open(str(stage_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not open stage: {stage_path}")

    a9_result = audit_a9(stage_path)
    type_counts: dict[str, int] = {}
    api_counts: dict[str, int] = {}
    references = []
    payloads = []
    a10_cubes = {}
    a10_measured_paths = []
    a10_pending_paths = []
    a10_derived_paths = []
    missing_scopes = set()
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        prim_type = prim.GetTypeName() or "Typeless"
        type_counts[prim_type] = type_counts.get(prim_type, 0) + 1
        for api in prim.GetAppliedSchemas():
            api_counts[api] = api_counts.get(api, 0) + 1
        if prim.HasAuthoredReferences():
            references.append(path)
        if prim.HasAuthoredPayloads():
            payloads.append(path)
        attrs = _custom_attrs(prim)
        if path in EXPECTED_A10_CUBES:
            a10_cubes[path] = {"type": prim_type, "custom_attrs": attrs}
        if path in EXPECTED_A10_MEASURED_PATHS and attrs.get("aloha:sourceType") == "MEASURED":
            a10_measured_paths.append(path)
        if path in EXPECTED_A10_PENDING_PATHS and attrs.get("aloha:measurementStatus") == "PENDING_MEASUREMENT":
            a10_pending_paths.append(path)
        if (
            path in EXPECTED_A10_DERIVED_PATHS
            and attrs.get("aloha:sourceType") == "DERIVED"
            and attrs.get("aloha:measurementStatus") == "DERIVED_FROM_MEASURED"
        ):
            a10_derived_paths.append(path)
        if path in EXPECTED_MISSING_SCOPES:
            missing_scopes.add(path)

    default_prim = stage.GetDefaultPrim()
    default_prim_path = str(default_prim.GetPath()) if default_prim else None
    root = stage.GetPrimAtPath(EXPECTED_DEFAULT_PRIM)
    root_custom_data = root.GetCustomData() if root else {}
    base_root = stage.GetPrimAtPath(BASE_ANCHOR_ROOT)
    base_attrs = _custom_attrs(base_root) if base_root else {}
    forbidden_type_hits = {
        name: type_counts.get(name, 0)
        for name in FORBIDDEN_TYPE_COUNTS
        if type_counts.get(name, 0) != 0
    }
    forbidden_api_hits = {
        name: api_counts.get(name, 0)
        for name in FORBIDDEN_API_MARKERS
        if api_counts.get(name, 0) != 0
    }
    forbidden_text_hits = [needle for needle in FORBIDDEN_TEXT if needle in text]

    a10_semantics_ok = True
    for path, details in a10_cubes.items():
        attrs = details["custom_attrs"]
        a10_semantics_ok = a10_semantics_ok and details["type"] == "Cube"
        a10_semantics_ok = a10_semantics_ok and attrs.get("aloha:visualOnly") is True
        a10_semantics_ok = a10_semantics_ok and attrs.get("aloha:physicsEligible") is False
        a10_semantics_ok = a10_semantics_ok and attrs.get("aloha:collisionEligible") is False
        a10_semantics_ok = a10_semantics_ok and attrs.get("aloha:cameraExtrinsicsEligible") is False
        a10_semantics_ok = a10_semantics_ok and attrs.get("aloha:renderEligible") is False
        a10_semantics_ok = a10_semantics_ok and attrs.get("aloha:trainingEligible") is False
        if path in EXPECTED_A10_MEASURED_PATHS:
            a10_semantics_ok = a10_semantics_ok and attrs.get("aloha:fullBaseFootprint") is False
            a10_semantics_ok = a10_semantics_ok and attrs.get("aloha:xExtentMeasured") is False
        if path in EXPECTED_A10_PENDING_PATHS:
            a10_semantics_ok = a10_semantics_ok and attrs.get("aloha:isActualBaseFootprint") is False
            a10_semantics_ok = a10_semantics_ok and attrs.get("aloha:baseGeometryComplete") is False

    # A10 extends A9, so A9's own exact measured-source set should still be
    # present. We check it directly to avoid A9 audit rejecting A10 additions.
    all_a9_sources_present = (
        set(a9_result.get("measured_source_paths", [])) == EXPECTED_MEASURED_SOURCE_PATHS
        and set(a9_result.get("derived_from_measured_paths", [])) == EXPECTED_DERIVED_FROM_MEASURED_PATHS
        and not a9_result.get("unexpected_measured_source_hits")
    )

    ok = (
        default_prim_path == EXPECTED_DEFAULT_PRIM
        and UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
        and UsdGeom.GetStageMetersPerUnit(stage) == 1.0
        and set(a10_cubes) == EXPECTED_A10_CUBES
        and set(a10_measured_paths) == EXPECTED_A10_MEASURED_PATHS
        and set(a10_pending_paths) == EXPECTED_A10_PENDING_PATHS
        and set(a10_derived_paths) == EXPECTED_A10_DERIVED_PATHS
        and missing_scopes == EXPECTED_MISSING_SCOPES
        and type_counts.get("Camera", 0) == 4
        and type_counts.get("Cube", 0) == 48
        and a10_semantics_ok
        and all_a9_sources_present
        and base_attrs.get("aloha:stageRole") == "A10_robot_base_anchor_measurement_checkpoint"
        and base_attrs.get("aloha:measurementStatus") == "MEASURED_Y_BAND_ONLY"
        and base_attrs.get("aloha:baseYEdgesMeasured") is True
        and base_attrs.get("aloha:baseGeometryComplete") is False
        and base_attrs.get("aloha:measuredCadReady") is False
        and base_attrs.get("aloha:physicsEligible") is False
        and base_attrs.get("aloha:collisionEligible") is False
        and base_attrs.get("aloha:cameraExtrinsicsEligible") is False
        and base_attrs.get("aloha:renderEligible") is False
        and base_attrs.get("aloha:trainingEligible") is False
        and root_custom_data.get("aloha1_rebuild_stage") == "A10_robot_base_anchor_preview"
        and root_custom_data.get("base_y_edges_measured") is True
        and root_custom_data.get("base_geometry_complete") is False
        and root_custom_data.get("base_x_extent_ready") is False
        and root_custom_data.get("robot_base_placement_ready") is False
        and root_custom_data.get("camera_work_deferred") is True
        and root_custom_data.get("training_eligible") is False
        and root_custom_data.get("rl_ready") is False
        and not forbidden_type_hits
        and not forbidden_api_hits
        and not forbidden_text_hits
        and not references
        and not payloads
    )

    return {
        "ok": ok,
        "stage_path": str(stage_path),
        "default_prim": default_prim_path,
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "type_counts": type_counts,
        "api_counts": api_counts,
        "a10_cube_paths": sorted(a10_cubes),
        "a10_measured_paths": sorted(a10_measured_paths),
        "a10_pending_paths": sorted(a10_pending_paths),
        "a10_derived_paths": sorted(a10_derived_paths),
        "missing_measurement_scopes": sorted(missing_scopes),
        "a10_semantics_ok": a10_semantics_ok,
        "a9_sources_present": all_a9_sources_present,
        "base_anchor_attrs": base_attrs,
        "root_custom_data": root_custom_data,
        "forbidden_type_hits": forbidden_type_hits,
        "forbidden_api_hits": forbidden_api_hits,
        "forbidden_text_hits": forbidden_text_hits,
        "references": references,
        "payloads": payloads,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(args.stage)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
