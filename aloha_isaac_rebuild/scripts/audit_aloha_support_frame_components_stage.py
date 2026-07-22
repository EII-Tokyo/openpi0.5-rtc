#!/usr/bin/env python3
"""Audit the A5 clean ALOHA support-frame component visual stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pxr import Usd, UsdGeom

from audit_aloha_camera_marker_visual_stage import EXPECTED_CAMERAS, EXPECTED_MARKERS


EXPECTED_DEFAULT_PRIM = "/aloha"
EXPECTED_COMPONENTS = {
    "top_steel_rail_proxy": "/aloha/support_frame/visuals/components/top_steel_rail_proxy",
    "extension_outer_camera_rail_proxy": "/aloha/support_frame/visuals/components/extension_outer_camera_rail_proxy",
    "support_pipe_260mm_1": "/aloha/support_frame/visuals/components/support_pipe_260mm_1",
    "support_pipe_260mm_2": "/aloha/support_frame/visuals/components/support_pipe_260mm_2",
    "support_pipe_260mm_3": "/aloha/support_frame/visuals/components/support_pipe_260mm_3",
    "support_pipe_260mm_4": "/aloha/support_frame/visuals/components/support_pipe_260mm_4",
}
EXPECTED_ALIGNMENT_GUIDES = {
    "base_edge_near_cam_low_y_guide": "/aloha/support_frame/visuals/alignment_guides/base_edge_near_cam_low_y_guide",
    "base_edge_near_cam_high_y_guide": "/aloha/support_frame/visuals/alignment_guides/base_edge_near_cam_high_y_guide",
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
        component_semantics_ok = component_semantics_ok and attrs.get("aloha:sourceKind") == "aloha1_visual_layout_v0"
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
        and type_counts.get("Cube", 0) == 21
        and component_semantics_ok
        and guide_semantics_ok
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
