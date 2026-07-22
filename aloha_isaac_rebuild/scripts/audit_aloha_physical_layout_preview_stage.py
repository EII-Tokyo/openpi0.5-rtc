#!/usr/bin/env python3
"""Audit the A9 visual-only non-camera physical-layout preview stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pxr import Usd, UsdGeom

from audit_aloha_camera_default_pose_preview_stage import EXPECTED_CAMERAS
from audit_aloha_camera_default_pose_preview_stage import EXPECTED_PREVIEW_DIRECTIONS
from audit_aloha_camera_default_pose_preview_stage import EXPECTED_PREVIEW_MARKERS
from audit_aloha_camera_marker_visual_stage import EXPECTED_MARKERS
from audit_aloha_support_frame_components_stage import EXPECTED_ALIGNMENT_GUIDES
from audit_aloha_support_frame_components_stage import EXPECTED_COMPONENTS
from audit_aloha_support_frame_components_stage import EXPECTED_DEFAULT_PRIM


PHYSICAL_LAYOUT_ROOT = "/aloha/physical_layout"
EXPECTED_A9_CUBES = {
    "/aloha/physical_layout/visuals/table_reference_outline/x_min_edge",
    "/aloha/physical_layout/visuals/table_reference_outline/x_max_edge",
    "/aloha/physical_layout/visuals/table_reference_outline/y_min_edge",
    "/aloha/physical_layout/visuals/table_reference_outline/y_max_edge",
    "/aloha/physical_layout/visuals/world_frame_reference/origin",
    "/aloha/physical_layout/visuals/world_frame_reference/positive_x_axis",
    "/aloha/physical_layout/visuals/world_frame_reference/positive_y_axis",
    "/aloha/physical_layout/visuals/world_frame_reference/positive_z_axis",
    "/aloha/physical_layout/visuals/base_y_reference/near_cam_low_y_edge",
    "/aloha/physical_layout/visuals/base_y_reference/near_cam_high_y_edge",
}
EXPECTED_TABLE_REFERENCE_CUBES = {
    "/aloha/physical_layout/visuals/table_reference_outline/x_min_edge",
    "/aloha/physical_layout/visuals/table_reference_outline/x_max_edge",
    "/aloha/physical_layout/visuals/table_reference_outline/y_min_edge",
    "/aloha/physical_layout/visuals/table_reference_outline/y_max_edge",
}
EXPECTED_DERIVED_FROM_MEASURED_PATHS = {
    "/aloha/support_frame/visuals/y_extension_layout/y_extension_cross_member_260mm_1",
    "/aloha/support_frame/visuals/y_extension_layout/y_extension_cross_member_260mm_4",
}
EXPECTED_MEASURED_SOURCE_PATHS = (
    set(EXPECTED_COMPONENTS.values()) - EXPECTED_DERIVED_FROM_MEASURED_PATHS
) | set(EXPECTED_ALIGNMENT_GUIDES.values()) | {
    "/aloha/physical_layout/visuals/base_y_reference/near_cam_low_y_edge",
    "/aloha/physical_layout/visuals/base_y_reference/near_cam_high_y_edge",
}
EXPECTED_FUTURE_SCOPES = {
    "/aloha/physical_layout/future_measurements/pipe_pose",
    "/aloha/physical_layout/future_measurements/bottle_initial_pose",
}
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

    type_counts: dict[str, int] = {}
    api_counts: dict[str, int] = {}
    references = []
    payloads = []
    cameras = set()
    markers = set()
    components = set()
    guides = set()
    preview_markers = set()
    preview_directions = set()
    a9_cubes = {}
    future_scopes = set()
    camera_xform_ops = {}
    measured_source_paths = []
    derived_from_measured_paths = []
    unexpected_measured_source_hits = []
    table_reference_checks = {}
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
        if prim_type == "Camera":
            cameras.add(path)
            camera_xform_ops[path] = [op.GetOpName() for op in UsdGeom.Xformable(prim).GetOrderedXformOps()]
        if path in EXPECTED_MARKERS.values():
            markers.add(path)
        if path in EXPECTED_COMPONENTS.values():
            components.add(path)
        if path in EXPECTED_ALIGNMENT_GUIDES.values():
            guides.add(path)
        if path in EXPECTED_PREVIEW_MARKERS.values():
            preview_markers.add(path)
        if path in EXPECTED_PREVIEW_DIRECTIONS.values():
            preview_directions.add(path)
        if path in EXPECTED_A9_CUBES:
            a9_cubes[path] = {"type": prim_type, "custom_attrs": _custom_attrs(prim)}
        attrs = _custom_attrs(prim)
        if (
            (path.startswith("/aloha/support_frame") or path.startswith(PHYSICAL_LAYOUT_ROOT))
            and attrs.get("aloha:sourceType") == "MEASURED"
        ):
            measured_source_paths.append(path)
            if path not in EXPECTED_MEASURED_SOURCE_PATHS:
                unexpected_measured_source_hits.append(path)
        if (
            (path.startswith("/aloha/support_frame") or path.startswith(PHYSICAL_LAYOUT_ROOT))
            and attrs.get("aloha:sourceType") == "DERIVED"
            and attrs.get("aloha:measurementStatus") == "DERIVED_FROM_MEASURED"
        ):
            derived_from_measured_paths.append(path)
        if path in EXPECTED_TABLE_REFERENCE_CUBES:
            visibility_attr = prim.GetAttribute("visibility")
            visibility = visibility_attr.Get() if visibility_attr else None
            table_reference_checks[path] = {
                "visibility": str(visibility),
                "reference_only": attrs.get("aloha:referenceOnly"),
                "support_frame_measurement": attrs.get("aloha:supportFrameMeasurement"),
                "main_frame_measurement": attrs.get("aloha:mainFrameMeasurement"),
            }
        if path in EXPECTED_FUTURE_SCOPES:
            future_scopes.add(path)

    default_prim = stage.GetDefaultPrim()
    default_prim_path = str(default_prim.GetPath()) if default_prim else None
    root = stage.GetPrimAtPath(EXPECTED_DEFAULT_PRIM)
    root_custom_data = root.GetCustomData() if root else {}
    layout_root = stage.GetPrimAtPath(PHYSICAL_LAYOUT_ROOT)
    layout_attrs = _custom_attrs(layout_root) if layout_root else {}
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

    a9_semantics_ok = True
    for details in a9_cubes.values():
        attrs = details["custom_attrs"]
        a9_semantics_ok = a9_semantics_ok and details["type"] == "Cube"
        a9_semantics_ok = a9_semantics_ok and attrs.get("aloha:visualOnly") is True
        a9_semantics_ok = a9_semantics_ok and attrs.get("aloha:physicsEligible") is False
        a9_semantics_ok = a9_semantics_ok and attrs.get("aloha:collisionEligible") is False
        a9_semantics_ok = a9_semantics_ok and attrs.get("aloha:renderEligible") is False
        a9_semantics_ok = a9_semantics_ok and attrs.get("aloha:trainingEligible") is False

    table_reference_hidden_ok = (
        set(table_reference_checks) == EXPECTED_TABLE_REFERENCE_CUBES
        and all(item["visibility"] == "invisible" for item in table_reference_checks.values())
        and all(item["reference_only"] is True for item in table_reference_checks.values())
        and all(item["support_frame_measurement"] is False for item in table_reference_checks.values())
        and all(item["main_frame_measurement"] is False for item in table_reference_checks.values())
    )
    camera_xforms_unchanged = all(not ops for ops in camera_xform_ops.values())
    ok = (
        default_prim_path == EXPECTED_DEFAULT_PRIM
        and UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
        and UsdGeom.GetStageMetersPerUnit(stage) == 1.0
        and cameras == set(EXPECTED_CAMERAS.values())
        and markers == set(EXPECTED_MARKERS.values())
        and components == set(EXPECTED_COMPONENTS.values())
        and guides == set(EXPECTED_ALIGNMENT_GUIDES.values())
        and preview_markers == set(EXPECTED_PREVIEW_MARKERS.values())
        and preview_directions == set(EXPECTED_PREVIEW_DIRECTIONS.values())
        and set(a9_cubes) == EXPECTED_A9_CUBES
        and set(measured_source_paths) == EXPECTED_MEASURED_SOURCE_PATHS
        and set(derived_from_measured_paths) == EXPECTED_DERIVED_FROM_MEASURED_PATHS
        and not unexpected_measured_source_hits
        and table_reference_hidden_ok
        and future_scopes == EXPECTED_FUTURE_SCOPES
        and type_counts.get("Camera", 0) == 4
        and type_counts.get("Cube", 0) == 40
        and a9_semantics_ok
        and camera_xforms_unchanged
        and layout_attrs.get("aloha:cameraWorkDeferred") is True
        and layout_attrs.get("aloha:physicsEligible") is False
        and layout_attrs.get("aloha:collisionEligible") is False
        and layout_attrs.get("aloha:renderEligible") is False
        and layout_attrs.get("aloha:trainingEligible") is False
        and root_custom_data.get("a9_scope") == "non_camera_physical_layout_visual_checkpoint"
        and root_custom_data.get("camera_work_deferred") is True
        and root_custom_data.get("camera_calibration_ready") is False
        and root_custom_data.get("camera_render_ready") is False
        and root_custom_data.get("physical_layout_physics_ready") is False
        and root_custom_data.get("physical_layout_collision_ready") is False
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
        "camera_paths": sorted(cameras),
        "camera_xform_ops": camera_xform_ops,
        "camera_xforms_unchanged": camera_xforms_unchanged,
        "a9_cube_paths": sorted(a9_cubes),
        "measured_source_paths": sorted(measured_source_paths),
        "derived_from_measured_paths": sorted(derived_from_measured_paths),
        "unexpected_measured_source_hits": unexpected_measured_source_hits,
        "table_reference_checks": table_reference_checks,
        "table_reference_hidden_ok": table_reference_hidden_ok,
        "future_measurement_scopes": sorted(future_scopes),
        "a9_semantics_ok": a9_semantics_ok,
        "layout_attrs": layout_attrs,
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
