#!/usr/bin/env python3
"""Audit the ALOHA1 clean skeleton USD stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pxr import Usd, UsdGeom


EXPECTED_DEFAULT_PRIM = "/aloha"

EXPECTED_PRIMS = {
    "/Render": "Scope",
    "/meshes": "Scope",
    "/visuals": "Scope",
    "/colliders": "Scope",
    "/aloha": "Xform",
    "/aloha/Looks": "Scope",
    "/aloha/joints": "Scope",
    "/aloha/table_link": "Xform",
    "/aloha/support_frame": "Xform",
    "/aloha/cam_high_link": "Xform",
    "/aloha/cam_low_link": "Xform",
    "/aloha/left_base_link": "Xform",
    "/aloha/right_base_link": "Xform",
    "/aloha/left_camera_link": "Xform",
    "/aloha/right_camera_link": "Xform",
}

FORBIDDEN_SUBSTRINGS = [
    "stationary_ai.usd",
    "aloha2_menagerie_scene_deep_black_real_start_pose",
    "aloha_isaac_menagerie_deep_black_real_start_pose",
    "/scene/worldBody",
    "/scene/left_base_link",
    "/scene/right_base_link",
    "follower_left_joint_0",
    "follower_right_joint_0",
    "carriage_joint",
    "0.044",
    "replay",
    "hdf5",
]

FORBIDDEN_TYPE_COUNTS = [
    "Mesh",
    "Camera",
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


def audit(stage_path: Path) -> dict:
    text = stage_path.read_text(encoding="utf-8")
    stage = Usd.Stage.Open(str(stage_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not open stage: {stage_path}")

    default_prim = stage.GetDefaultPrim()
    default_prim_path = str(default_prim.GetPath()) if default_prim else None

    type_counts: dict[str, int] = {}
    api_counts: dict[str, int] = {}
    prim_types: dict[str, str] = {}
    relationship_targets: dict[str, list[str]] = {}
    references = []
    payloads = []

    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        prim_type = prim.GetTypeName() or "Typeless"
        type_counts[prim_type] = type_counts.get(prim_type, 0) + 1
        prim_types[prim_path] = prim_type

        for api in prim.GetAppliedSchemas():
            api_counts[api] = api_counts.get(api, 0) + 1

        if prim.HasAuthoredReferences():
            references.append(prim_path)
        if prim.HasAuthoredPayloads():
            payloads.append(prim_path)

        for rel in prim.GetRelationships():
            targets = [str(target) for target in rel.GetTargets()]
            if targets:
                relationship_targets[f"{prim_path}.{rel.GetName()}"] = targets

    expected_errors = []
    for path, expected_type in EXPECTED_PRIMS.items():
        observed_type = prim_types.get(path)
        if observed_type != expected_type:
            expected_errors.append(
                {"path": path, "expected": expected_type, "observed": observed_type}
            )

    forbidden_text_hits = [needle for needle in FORBIDDEN_SUBSTRINGS if needle in text]
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

    missing_relationship_targets = {}
    for rel_name, targets in relationship_targets.items():
        missing = [target for target in targets if not stage.GetPrimAtPath(target).IsValid()]
        if missing:
            missing_relationship_targets[rel_name] = missing

    ok = (
        default_prim_path == EXPECTED_DEFAULT_PRIM
        and UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
        and UsdGeom.GetStageMetersPerUnit(stage) == 1.0
        and not expected_errors
        and not forbidden_text_hits
        and not forbidden_type_hits
        and not forbidden_api_hits
        and not references
        and not payloads
        and not missing_relationship_targets
    )

    return {
        "ok": ok,
        "stage_path": str(stage_path),
        "default_prim": default_prim_path,
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "root_children": [str(child.GetPath()) for child in stage.GetPseudoRoot().GetChildren()],
        "expected_errors": expected_errors,
        "type_counts": type_counts,
        "api_counts": api_counts,
        "forbidden_text_hits": forbidden_text_hits,
        "forbidden_type_hits": forbidden_type_hits,
        "forbidden_api_hits": forbidden_api_hits,
        "references": references,
        "payloads": payloads,
        "missing_relationship_targets": missing_relationship_targets,
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
