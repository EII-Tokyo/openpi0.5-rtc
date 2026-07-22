#!/usr/bin/env python3
"""Audit the A5 clean ALOHA support-frame component visual stage."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pxr import Usd, UsdGeom

from audit_aloha_camera_marker_visual_stage import EXPECTED_CAMERAS, EXPECTED_MARKERS


EXPECTED_DEFAULT_PRIM = "/aloha"
EXPECTED_COMPONENTS = {
    "original_cam_low_side_rail": "/aloha/support_frame/visuals/y_extension_layout/original_cam_low_side_rail",
    "extension_outer_camera_rail": "/aloha/support_frame/visuals/y_extension_layout/extension_outer_camera_rail",
    "y_extension_cross_member_260mm_1": "/aloha/support_frame/visuals/y_extension_layout/y_extension_cross_member_260mm_1",
    "y_extension_cross_member_260mm_2": "/aloha/support_frame/visuals/y_extension_layout/y_extension_cross_member_260mm_2",
    "y_extension_cross_member_260mm_3": "/aloha/support_frame/visuals/y_extension_layout/y_extension_cross_member_260mm_3",
    "y_extension_cross_member_260mm_4": "/aloha/support_frame/visuals/y_extension_layout/y_extension_cross_member_260mm_4",
    "cam_low_vertical_mount_post_100mm": "/aloha/support_frame/visuals/y_extension_layout/cam_low_vertical_mount_post_100mm",
}
EXPECTED_ALIGNMENT_GUIDES = {
    "base_edge_near_cam_low_y_guide": "/aloha/support_frame/visuals/base_alignment_guides/base_edge_near_cam_low_y_guide",
    "base_edge_near_cam_high_y_guide": "/aloha/support_frame/visuals/base_alignment_guides/base_edge_near_cam_high_y_guide",
}
EXPECTED_COMPONENT_CENTERS = {
    "/aloha/support_frame/visuals/y_extension_layout/original_cam_low_side_rail": (0.0, 0.3025, 0.010),
    "/aloha/support_frame/visuals/y_extension_layout/extension_outer_camera_rail": (0.0, 0.5825, 0.010),
    "/aloha/support_frame/visuals/y_extension_layout/y_extension_cross_member_260mm_1": (-0.600, 0.4425, 0.010),
    "/aloha/support_frame/visuals/y_extension_layout/y_extension_cross_member_260mm_2": (-0.433554, 0.4425, 0.010),
    "/aloha/support_frame/visuals/y_extension_layout/y_extension_cross_member_260mm_3": (0.433554, 0.4425, 0.010),
    "/aloha/support_frame/visuals/y_extension_layout/y_extension_cross_member_260mm_4": (0.600, 0.4425, 0.010),
    "/aloha/support_frame/visuals/y_extension_layout/cam_low_vertical_mount_post_100mm": (0.030, 0.5825, 0.070),
    "/aloha/support_frame/visuals/base_alignment_guides/base_edge_near_cam_low_y_guide": (0.0, 0.1325, 0.010),
    "/aloha/support_frame/visuals/base_alignment_guides/base_edge_near_cam_high_y_guide": (0.0, -0.0775, 0.010),
}
FORBIDDEN_TYPE_COUNTS = [
    "PhysicsScene",
    "PhysicsFixedJoint",
    "PhysicsRevoluteJoint",
    "PhysicsPrismaticJoint",
]
FORBIDDEN_API_MARKERS = [
    "PhysicsRigidBodyAPI",
    "PhysicsCollisionAPI",
    "PhysicsMassAPI",
    "PhysicsArticulationRootAPI",
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
]


def _custom_attrs(prim: Usd.Prim) -> dict:
    return {
        attr.GetName(): attr.Get()
        for attr in prim.GetAuthoredAttributes()
        if attr.GetName().startswith("aloha:")
    }


def _translate(prim: Usd.Prim) -> tuple[float, float, float] | None:
    attr = prim.GetAttribute("xformOp:translate")
    value = attr.Get() if attr else None
    if value is None:
        return None
    return (float(value[0]), float(value[1]), float(value[2]))


def _near_tuple(actual: tuple[float, float, float] | None, expected: tuple[float, float, float]) -> bool:
    return actual is not None and all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(actual, expected))


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
    components = {}
    guides = {}
    center_checks = {}
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
        if path in EXPECTED_MARKERS.values():
            markers.add(path)
        if path in EXPECTED_COMPONENTS.values():
            components[path] = {
                "type": prim_type,
                "custom_attrs": _custom_attrs(prim),
            }
        if path in EXPECTED_ALIGNMENT_GUIDES.values():
            guides[path] = {
                "type": prim_type,
                "custom_attrs": _custom_attrs(prim),
            }
        if path in EXPECTED_COMPONENT_CENTERS:
            actual = _translate(prim)
            expected = EXPECTED_COMPONENT_CENTERS[path]
            center_checks[path] = {
                "actual": actual,
                "expected": expected,
                "ok": _near_tuple(actual, expected),
            }

    default_prim = stage.GetDefaultPrim()
    default_prim_path = str(default_prim.GetPath()) if default_prim else None
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

    component_semantics_ok = True
    for name, path in EXPECTED_COMPONENTS.items():
        attrs = components.get(path, {}).get("custom_attrs", {})
        component_semantics_ok = component_semantics_ok and components.get(path, {}).get("type") == "Cube"
        component_semantics_ok = component_semantics_ok and attrs.get("aloha:componentName") == name
        component_semantics_ok = component_semantics_ok and attrs.get("aloha:visualOnly") is True
        source_kind = attrs.get("aloha:sourceKind")
        component_semantics_ok = component_semantics_ok and (
            source_kind == "aloha1_visual_layout_v0"
            or source_kind.startswith("user_measured_")
            or source_kind.startswith("derived_")
        )
        if not source_kind.startswith("derived_"):
            component_semantics_ok = component_semantics_ok and attrs.get("aloha:sourceType") == "MEASURED"
            component_semantics_ok = component_semantics_ok and attrs.get("aloha:measurementStatus") == "MEASURED"
        else:
            component_semantics_ok = component_semantics_ok and attrs.get("aloha:sourceType") == "DERIVED"
            component_semantics_ok = component_semantics_ok and attrs.get("aloha:measurementStatus") == "DERIVED_FROM_MEASURED"
        component_semantics_ok = component_semantics_ok and attrs.get("aloha:measuredCadReady") is False
        component_semantics_ok = component_semantics_ok and attrs.get("aloha:physicsEligible") is False
        component_semantics_ok = component_semantics_ok and attrs.get("aloha:collisionEligible") is False
        component_semantics_ok = component_semantics_ok and attrs.get("aloha:cameraExtrinsicsEligible") is False
        component_semantics_ok = component_semantics_ok and attrs.get("aloha:trainingEligible") is False

    guide_semantics_ok = True
    for name, path in EXPECTED_ALIGNMENT_GUIDES.items():
        attrs = guides.get(path, {}).get("custom_attrs", {})
        guide_semantics_ok = guide_semantics_ok and guides.get(path, {}).get("type") == "Cube"
        guide_semantics_ok = guide_semantics_ok and attrs.get("aloha:guideName") == name
        guide_semantics_ok = guide_semantics_ok and attrs.get("aloha:sourceType") == "MEASURED"
        guide_semantics_ok = guide_semantics_ok and attrs.get("aloha:sourceKind") == "user_measured_base_y_edges"
        guide_semantics_ok = guide_semantics_ok and attrs.get("aloha:measurementStatus") == "MEASURED"
        guide_semantics_ok = guide_semantics_ok and attrs.get("aloha:visualOnly") is True
        guide_semantics_ok = guide_semantics_ok and attrs.get("aloha:baseGeometryComplete") is False
        guide_semantics_ok = guide_semantics_ok and attrs.get("aloha:physicsEligible") is False
        guide_semantics_ok = guide_semantics_ok and attrs.get("aloha:collisionEligible") is False
        guide_semantics_ok = guide_semantics_ok and attrs.get("aloha:trainingEligible") is False

    ok = (
        default_prim_path == EXPECTED_DEFAULT_PRIM
        and UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
        and UsdGeom.GetStageMetersPerUnit(stage) == 1.0
        and cameras == set(EXPECTED_CAMERAS.values())
        and markers == set(EXPECTED_MARKERS.values())
        and set(components) == set(EXPECTED_COMPONENTS.values())
        and set(guides) == set(EXPECTED_ALIGNMENT_GUIDES.values())
        and type_counts.get("Camera", 0) == 4
        and type_counts.get("Cube", 0) == 22
        and component_semantics_ok
        and guide_semantics_ok
        and set(center_checks) == set(EXPECTED_COMPONENT_CENTERS)
        and all(item["ok"] for item in center_checks.values())
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
        "marker_paths": sorted(markers),
        "component_paths": sorted(components),
        "component_details": components,
        "component_semantics_ok": component_semantics_ok,
        "alignment_guide_paths": sorted(guides),
        "alignment_guide_details": guides,
        "alignment_guide_semantics_ok": guide_semantics_ok,
        "center_checks": center_checks,
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
