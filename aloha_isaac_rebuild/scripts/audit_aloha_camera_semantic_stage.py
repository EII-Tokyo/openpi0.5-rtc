#!/usr/bin/env python3
"""Audit the A3 clean ALOHA camera semantic skeleton stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pxr import Usd, UsdGeom


EXPECTED_DEFAULT_PRIM = "/aloha"
EXPECTED_CAMERAS = {
    "cam_high": "/aloha/cam_high_link/cam_high_color_frame/cam_high",
    "cam_low": "/aloha/cam_low_link/cam_low_color_frame/cam_low",
    "cam_left_wrist": "/aloha/left_camera_link/cam_left_wrist_color_frame/cam_left_wrist",
    "cam_right_wrist": "/aloha/right_camera_link/cam_right_wrist_color_frame/cam_right_wrist",
}
EXPECTED_COLOR_FRAMES = {
    "/aloha/cam_high_link/cam_high_color_frame",
    "/aloha/cam_low_link/cam_low_color_frame",
    "/aloha/left_camera_link/cam_left_wrist_color_frame",
    "/aloha/right_camera_link/cam_right_wrist_color_frame",
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
    "replay",
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


def _authored_attr_names(prim: Usd.Prim) -> set[str]:
    return {attr.GetName() for attr in prim.GetAuthoredAttributes()}


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
    color_frames = []
    for prim in stage.Traverse():
        prim_type = prim.GetTypeName() or "Typeless"
        type_counts[prim_type] = type_counts.get(prim_type, 0) + 1
        for api in prim.GetAppliedSchemas():
            api_counts[api] = api_counts.get(api, 0) + 1
        if prim.HasAuthoredReferences():
            references.append(str(prim.GetPath()))
        if prim.HasAuthoredPayloads():
            payloads.append(str(prim.GetPath()))
        path = str(prim.GetPath())
        if path in EXPECTED_COLOR_FRAMES:
            color_frames.append(path)
        if prim_type == "Camera":
            authored_attrs = sorted(_authored_attr_names(prim))
            custom_attrs = {
                attr.GetName(): attr.Get()
                for attr in prim.GetAuthoredAttributes()
                if attr.GetName().startswith("aloha:")
            }
            camera_details[path] = {
                "authored_attrs": authored_attrs,
                "custom_attrs": custom_attrs,
                "custom_data": prim.GetCustomData(),
            }

    default_prim = stage.GetDefaultPrim()
    default_prim_path = str(default_prim.GetPath()) if default_prim else None

    expected_camera_paths = set(EXPECTED_CAMERAS.values())
    actual_camera_paths = set(camera_details)
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
    eligibility_ok = True
    status_ok = True
    semantic_names_ok = True
    for name, path in EXPECTED_CAMERAS.items():
        details = camera_details.get(path)
        if not details:
            eligibility_ok = False
            status_ok = False
            semantic_names_ok = False
            continue
        attrs = details["custom_attrs"]
        semantic_names_ok = semantic_names_ok and attrs.get("aloha:semanticName") == name
        status_ok = status_ok and attrs.get("aloha:calibrationStatus") == "MISSING"
        status_ok = status_ok and attrs.get("aloha:intrinsicsStatus") == "MISSING"
        status_ok = status_ok and attrs.get("aloha:extrinsicsStatus") == "MISSING"
        eligibility_ok = eligibility_ok and attrs.get("aloha:renderEligible") is False
        eligibility_ok = eligibility_ok and attrs.get("aloha:trainingEligible") is False
        eligibility_ok = eligibility_ok and attrs.get("aloha:rosBridgeEligible") is False
        eligibility_ok = eligibility_ok and attrs.get("aloha:openpiCaptureEligible") is False

    ok = (
        default_prim_path == EXPECTED_DEFAULT_PRIM
        and UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
        and UsdGeom.GetStageMetersPerUnit(stage) == 1.0
        and actual_camera_paths == expected_camera_paths
        and set(color_frames) == EXPECTED_COLOR_FRAMES
        and type_counts.get("Camera", 0) == 4
        and type_counts.get("Cube", 0) == 5
        and semantic_names_ok
        and status_ok
        and eligibility_ok
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
        "root_children": [str(child.GetPath()) for child in stage.GetPseudoRoot().GetChildren()],
        "type_counts": type_counts,
        "api_counts": api_counts,
        "expected_camera_paths": EXPECTED_CAMERAS,
        "actual_camera_paths": sorted(actual_camera_paths),
        "color_frames": sorted(color_frames),
        "camera_details": camera_details,
        "semantic_names_ok": semantic_names_ok,
        "status_ok": status_ok,
        "eligibility_ok": eligibility_ok,
        "authored_camera_attr_hits": authored_camera_attr_hits,
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
        args.json_output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
