#!/usr/bin/env python3
"""Audit the A4 clean ALOHA camera marker visual stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pxr import Usd, UsdGeom

from audit_aloha_camera_semantic_stage import EXPECTED_CAMERAS


EXPECTED_DEFAULT_PRIM = "/aloha"
EXPECTED_MARKERS = {
    "cam_high": "/aloha/visuals/camera_markers/cam_high_marker",
    "cam_low": "/aloha/visuals/camera_markers/cam_low_marker",
    "cam_left_wrist": "/aloha/visuals/camera_markers/cam_left_wrist_marker",
    "cam_right_wrist": "/aloha/visuals/camera_markers/cam_right_wrist_marker",
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
FORBIDDEN_AUTHORED_CAMERA_ATTRS = [
    "focalLength",
    "horizontalAperture",
    "verticalAperture",
    "horizontalApertureOffset",
    "verticalApertureOffset",
    "clippingRange",
    "focusDistance",
    "fStop",
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
    camera_details = {}
    marker_details = {}
    direction_details = {}

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
            camera_details[path] = {
                "authored_attrs": sorted(attr.GetName() for attr in prim.GetAuthoredAttributes()),
                "custom_attrs": _custom_attrs(prim),
            }
        if path in EXPECTED_MARKERS.values():
            marker_details[path] = {
                "type": prim_type,
                "custom_attrs": _custom_attrs(prim),
                "custom_data": prim.GetCustomData(),
            }
        if path.endswith("/view_direction_hint"):
            direction_details[path] = {
                "type": prim_type,
                "custom_attrs": _custom_attrs(prim),
            }

    default_prim = stage.GetDefaultPrim()
    default_prim_path = str(default_prim.GetPath()) if default_prim else None
    actual_camera_paths = set(camera_details)
    expected_camera_paths = set(EXPECTED_CAMERAS.values())
    actual_marker_paths = set(marker_details)
    expected_marker_paths = set(EXPECTED_MARKERS.values())

    forbidden_type_hits = {
        type_name: type_counts.get(type_name, 0)
        for type_name in FORBIDDEN_TYPE_COUNTS
        if type_counts.get(type_name, 0) != 0
    }
    forbidden_api_hits = {
        api_name: api_counts.get(api_name, 0)
        for api_name in FORBIDDEN_API_MARKERS
        if api_counts.get(api_name, 0) != 0
    }
    forbidden_text_hits = [needle for needle in FORBIDDEN_TEXT if needle in text]
    authored_camera_attr_hits = {
        path: [
            name
            for name in details["authored_attrs"]
            if name in FORBIDDEN_AUTHORED_CAMERA_ATTRS
        ]
        for path, details in camera_details.items()
    }
    authored_camera_attr_hits = {
        path: hits for path, hits in authored_camera_attr_hits.items() if hits
    }

    camera_semantics_ok = True
    for name, path in EXPECTED_CAMERAS.items():
        attrs = camera_details.get(path, {}).get("custom_attrs", {})
        camera_semantics_ok = camera_semantics_ok and attrs.get("aloha:semanticName") == name
        camera_semantics_ok = camera_semantics_ok and attrs.get("aloha:extrinsicsStatus") == "MISSING"
        camera_semantics_ok = camera_semantics_ok and attrs.get("aloha:trainingEligible") is False
        camera_semantics_ok = camera_semantics_ok and attrs.get("aloha:renderEligible") is False

    marker_semantics_ok = True
    marker_linkage_ok = True
    for name, marker_path in EXPECTED_MARKERS.items():
        attrs = marker_details.get(marker_path, {}).get("custom_attrs", {})
        marker_semantics_ok = marker_semantics_ok and attrs.get("aloha:visualMarkerOnly") is True
        marker_semantics_ok = marker_semantics_ok and attrs.get("aloha:trainingEligible") is False
        marker_semantics_ok = marker_semantics_ok and attrs.get("aloha:replayEligible") is False
        marker_semantics_ok = marker_semantics_ok and attrs.get("aloha:rlReady") is False
        marker_semantics_ok = marker_semantics_ok and attrs.get("aloha:physicsEligible") is False
        marker_semantics_ok = marker_semantics_ok and attrs.get("aloha:collisionEligible") is False
        marker_semantics_ok = marker_semantics_ok and attrs.get("aloha:sourceKind") == "manual_visual_layout_only"
        marker_semantics_ok = marker_semantics_ok and attrs.get("aloha:poseSource") == "MANUAL_SCHEMATIC"
        marker_semantics_ok = marker_semantics_ok and attrs.get("aloha:measuredPose") is False
        marker_semantics_ok = marker_semantics_ok and attrs.get("aloha:forbiddenAsExtrinsics") is True
        marker_semantics_ok = marker_semantics_ok and attrs.get("aloha:extrinsicsSource") == "MISSING"
        marker_semantics_ok = marker_semantics_ok and attrs.get("aloha:extrinsicsStatus") == "MISSING_REAL_EXTRINSICS"
        marker_linkage_ok = marker_linkage_ok and attrs.get("aloha:linkedSensorPrim") == EXPECTED_CAMERAS[name]

    direction_semantics_ok = len(direction_details) == 4 and all(
        details["type"] == "Cube"
        and details["custom_attrs"].get("aloha:visualMarkerOnly") is True
        and details["custom_attrs"].get("aloha:sourceKind") == "manual_visual_layout_only"
        and details["custom_attrs"].get("aloha:forbiddenAsExtrinsics") is True
        and details["custom_attrs"].get("aloha:extrinsicsSource") == "MISSING"
        for details in direction_details.values()
    )

    ok = (
        default_prim_path == EXPECTED_DEFAULT_PRIM
        and UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
        and UsdGeom.GetStageMetersPerUnit(stage) == 1.0
        and actual_camera_paths == expected_camera_paths
        and actual_marker_paths == expected_marker_paths
        and type_counts.get("Camera", 0) == 4
        and type_counts.get("Cube", 0) == 13
        and camera_semantics_ok
        and marker_semantics_ok
        and marker_linkage_ok
        and direction_semantics_ok
        and not authored_camera_attr_hits
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
        "expected_camera_paths": EXPECTED_CAMERAS,
        "actual_camera_paths": sorted(actual_camera_paths),
        "expected_marker_paths": EXPECTED_MARKERS,
        "actual_marker_paths": sorted(actual_marker_paths),
        "direction_paths": sorted(direction_details),
        "camera_semantics_ok": camera_semantics_ok,
        "marker_semantics_ok": marker_semantics_ok,
        "marker_linkage_ok": marker_linkage_ok,
        "direction_semantics_ok": direction_semantics_ok,
        "authored_camera_attr_hits": authored_camera_attr_hits,
        "forbidden_type_hits": forbidden_type_hits,
        "forbidden_api_hits": forbidden_api_hits,
        "forbidden_text_hits": forbidden_text_hits,
        "references": references,
        "payloads": payloads,
        "marker_details": marker_details,
        "direction_details": direction_details,
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
        args.json_output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
