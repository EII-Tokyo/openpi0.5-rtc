#!/usr/bin/env python3
"""Audit A13 original ALOHA1 visual baseline stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pxr import Usd, UsdGeom


EXPECTED_DEFAULT_PRIM = "/aloha"
EXPECTED_ROOTS = {
    "/aloha/support_frame",
    "/aloha/frame_link",
    "/aloha/frame_link/visuals",
    "/aloha/floor_reference_link",
    "/aloha/tabletop_link",
    "/aloha/Looks",
    "/aloha/source_manifest",
    "/aloha/follower_left_base_link",
    "/aloha/follower_left_link_1",
    "/aloha/follower_left_link_6",
    "/aloha/follower_left_gripper_left",
    "/aloha/follower_right_base_link",
    "/aloha/follower_right_link_1",
    "/aloha/follower_right_link_6",
    "/aloha/follower_right_gripper_right",
}
FORBIDDEN_TYPE_COUNTS = {
    "PhysicsScene",
    "PhysicsFixedJoint",
    "PhysicsRevoluteJoint",
    "PhysicsPrismaticJoint",
    "RenderProduct",
    "RenderSettings",
    "RenderVar",
}
FORBIDDEN_API_MARKERS = {
    "PhysicsRigidBodyAPI",
    "PhysicsCollisionAPI",
    "PhysicsMassAPI",
    "PhysicsArticulationRootAPI",
}
FORBIDDEN_PATH_FRAGMENTS = {
    "/joints",
    "/Render",
}


def _custom_attrs(prim: Usd.Prim) -> dict:
    return {
        attr.GetName(): attr.Get()
        for attr in prim.GetAuthoredAttributes()
        if attr.GetName().startswith("aloha:")
    }


def audit(stage_path: Path) -> dict:
    stage = Usd.Stage.Open(str(stage_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not open stage: {stage_path}")

    type_counts: dict[str, int] = {}
    api_counts: dict[str, int] = {}
    references: list[str] = []
    payloads: list[str] = []
    paths: set[str] = set()
    forbidden_paths: list[str] = []
    visual_reference_paths: list[str] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        paths.add(path)
        prim_type = prim.GetTypeName() or "Typeless"
        type_counts[prim_type] = type_counts.get(prim_type, 0) + 1
        for api in prim.GetAppliedSchemas():
            api_counts[api] = api_counts.get(api, 0) + 1
        if prim.HasAuthoredReferences():
            references.append(path)
            if (
                path.startswith("/aloha/frame_link/visuals/")
                or path.startswith("/aloha/follower_left_")
                or path.startswith("/aloha/follower_right_")
                or path == "/aloha/tabletop_link"
                or path == "/aloha/floor_reference_link"
                or path == "/aloha/Looks"
            ):
                visual_reference_paths.append(path)
        if prim.HasAuthoredPayloads():
            payloads.append(path)
        if any(fragment in path for fragment in FORBIDDEN_PATH_FRAGMENTS):
            forbidden_paths.append(path)

    default_prim = stage.GetDefaultPrim()
    default_prim_path = str(default_prim.GetPath()) if default_prim else None
    root = stage.GetPrimAtPath(EXPECTED_DEFAULT_PRIM)
    root_attrs = _custom_attrs(root) if root else {}
    manifest = stage.GetPrimAtPath("/aloha/source_manifest")
    manifest_data = manifest.GetCustomData() if manifest else {}
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
    support_count = int(manifest_data.get("support_frame_visual_component_count", -1))
    left_count = int(manifest_data.get("left_robot_visual_link_count", -1))
    right_count = int(manifest_data.get("right_robot_visual_link_count", -1))
    ok = (
        default_prim_path == EXPECTED_DEFAULT_PRIM
        and UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
        and UsdGeom.GetStageMetersPerUnit(stage) == 1.0
        and EXPECTED_ROOTS.issubset(paths)
        and support_count == 35
        and left_count == 10
        and right_count == 10
        and len(visual_reference_paths) >= 56
        and not payloads
        and not forbidden_type_hits
        and not forbidden_api_hits
        and not forbidden_paths
        and root_attrs.get("aloha:visualOnly") is True
        and root_attrs.get("aloha:physicsEligible") is False
        and root_attrs.get("aloha:collisionEligible") is False
        and root_attrs.get("aloha:articulationCompatible") is False
        and root_attrs.get("aloha:stationaryAiRuntimeCompatible") is False
        and root_attrs.get("aloha:trainingEligible") is False
    )
    return {
        "ok": ok,
        "stage_path": str(stage_path),
        "default_prim": default_prim_path,
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "type_counts": type_counts,
        "api_counts": api_counts,
        "reference_count": len(references),
        "visual_reference_count": len(visual_reference_paths),
        "payloads": payloads,
        "forbidden_type_hits": forbidden_type_hits,
        "forbidden_api_hits": forbidden_api_hits,
        "forbidden_paths": forbidden_paths[:40],
        "collision_named_paths_note": "Source ALOHA1 conversion uses /collisions as visible geometry containers; A13 permits the path name only when PhysicsCollisionAPI is absent.",
        "expected_roots_missing": sorted(EXPECTED_ROOTS - paths),
        "support_frame_visual_component_count": support_count,
        "left_robot_visual_link_count": left_count,
        "right_robot_visual_link_count": right_count,
        "source_aloha1_usd": manifest_data.get("source_aloha1_usd"),
        "known_missing_measurement": root.GetCustomDataByKey("known_missing_measurement") if root else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(args.stage)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
