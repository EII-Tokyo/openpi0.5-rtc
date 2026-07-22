#!/usr/bin/env python3
"""Audit the A7 clean ALOHA camera default-pose preview stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pxr import Usd, UsdGeom

from audit_aloha_support_frame_components_stage import EXPECTED_COMPONENTS
from audit_aloha_support_frame_components_stage import EXPECTED_DEFAULT_PRIM
from audit_aloha_support_frame_components_stage import EXPECTED_ALIGNMENT_GUIDES
from audit_aloha_camera_marker_visual_stage import EXPECTED_CAMERAS, EXPECTED_MARKERS


EXPECTED_PREVIEW_MARKERS = {
    name: f"/aloha/visuals/camera_default_pose_preview/{name}_default_optical_center"
    for name in EXPECTED_CAMERAS
}
EXPECTED_PREVIEW_DIRECTIONS = {
    name: f"/aloha/visuals/camera_default_pose_preview/{name}_default_optical_center/default_direction_hint"
    for name in EXPECTED_CAMERAS
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
]
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
    preview_markers = {}
    preview_directions = {}
    camera_details = {}
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
            camera_details[path] = _custom_attrs(prim)
        if path in EXPECTED_MARKERS.values():
            markers.add(path)
        if path in EXPECTED_COMPONENTS.values():
            components.add(path)
        if path in EXPECTED_ALIGNMENT_GUIDES.values():
            guides.add(path)
        if path in EXPECTED_PREVIEW_MARKERS.values():
            preview_markers[path] = {"type": prim_type, "custom_attrs": _custom_attrs(prim)}
        if path in EXPECTED_PREVIEW_DIRECTIONS.values():
            preview_directions[path] = {"type": prim_type, "custom_attrs": _custom_attrs(prim)}

    default_prim = stage.GetDefaultPrim()
    default_prim_path = str(default_prim.GetPath()) if default_prim else None
    root = stage.GetPrimAtPath(EXPECTED_DEFAULT_PRIM)
    root_custom_data = root.GetCustomData() if root else {}
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

    preview_semantics_ok = True
    for path, details in {**preview_markers, **preview_directions}.items():
        attrs = details["custom_attrs"]
        preview_semantics_ok = preview_semantics_ok and details["type"] == "Cube"
        preview_semantics_ok = preview_semantics_ok and attrs.get("aloha:sourceType") == "TUNED"
        preview_semantics_ok = preview_semantics_ok and attrs.get("aloha:sourceKind") == "default_placeholder_until_measured"
        preview_semantics_ok = preview_semantics_ok and attrs.get("aloha:visualPreviewOnly") is True
        preview_semantics_ok = preview_semantics_ok and attrs.get("aloha:measuredPose") is False
        preview_semantics_ok = preview_semantics_ok and attrs.get("aloha:renderEligible") is False
        preview_semantics_ok = preview_semantics_ok and attrs.get("aloha:trainingEligible") is False
        preview_semantics_ok = preview_semantics_ok and attrs.get("aloha:physicsEligible") is False
        preview_semantics_ok = preview_semantics_ok and attrs.get("aloha:collisionEligible") is False

    camera_semantics_ok = True
    for path, attrs in camera_details.items():
        camera_semantics_ok = camera_semantics_ok and attrs.get("aloha:defaultPosePreviewOnly") is True
        camera_semantics_ok = camera_semantics_ok and attrs.get("aloha:measuredPose") is False
        camera_semantics_ok = camera_semantics_ok and attrs.get("aloha:renderEligible") is False
        camera_semantics_ok = camera_semantics_ok and attrs.get("aloha:trainingEligible") is False
        camera_semantics_ok = camera_semantics_ok and attrs.get("aloha:extrinsicsStatus") == "DEFAULT_PLACEHOLDER_NOT_MEASURED"

    ok = (
        default_prim_path == EXPECTED_DEFAULT_PRIM
        and UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
        and UsdGeom.GetStageMetersPerUnit(stage) == 1.0
        and cameras == set(EXPECTED_CAMERAS.values())
        and markers == set(EXPECTED_MARKERS.values())
        and components == set(EXPECTED_COMPONENTS.values())
        and guides == set(EXPECTED_ALIGNMENT_GUIDES.values())
        and set(preview_markers) == set(EXPECTED_PREVIEW_MARKERS.values())
        and set(preview_directions) == set(EXPECTED_PREVIEW_DIRECTIONS.values())
        and type_counts.get("Camera", 0) == 4
        and type_counts.get("Cube", 0) == 30
        and preview_semantics_ok
        and camera_semantics_ok
        and root_custom_data.get("camera_default_pose_preview_only") is True
        and root_custom_data.get("camera_calibration_ready") is False
        and root_custom_data.get("camera_render_ready") is False
        and root_custom_data.get("training_eligible") is False
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
        "preview_marker_paths": sorted(preview_markers),
        "preview_direction_paths": sorted(preview_directions),
        "preview_semantics_ok": preview_semantics_ok,
        "camera_semantics_ok": camera_semantics_ok,
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
