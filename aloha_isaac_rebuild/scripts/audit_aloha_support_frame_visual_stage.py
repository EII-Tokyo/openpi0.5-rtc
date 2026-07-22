#!/usr/bin/env python3
"""Audit the A2 visual-only ALOHA support frame footprint stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pxr import Usd, UsdGeom


EXPECTED_DEFAULT_PRIM = "/aloha"
TABLE_PATH = "/aloha/table_link/visuals/tabletop_reference_proxy"
SUPPORT_FRAME_ROOT = "/aloha/support_frame/visuals/outer_footprint"
EXPECTED_TABLE_SCALE = (1.2192, 0.7490, 0.0200)
EXPECTED_TABLE_TRANSLATE = (0.0, 0.0, -0.0100)
EXPECTED_TABLE_BBOX_MIN = (-0.6096, -0.3745, -0.0200)
EXPECTED_TABLE_BBOX_MAX = (0.6096, 0.3745, 0.0)
EXPECTED_SUPPORT_FRAME_OUTER_LENGTH_M = 1.220
EXPECTED_SUPPORT_FRAME_OUTER_WIDTH_M = 0.625
EXPECTED_MARKER_WIDTH_M = 0.010
EXPECTED_MARKER_HEIGHT_M = 0.010
EXPECTED_BBOX_MIN = (-0.610, -0.3125, 0.0)
EXPECTED_BBOX_MAX = (0.610, 0.3125, 0.010)
EXPECTED_MARKERS = {
    f"{SUPPORT_FRAME_ROOT}/front_outer_edge",
    f"{SUPPORT_FRAME_ROOT}/back_outer_edge",
    f"{SUPPORT_FRAME_ROOT}/left_outer_edge",
    f"{SUPPORT_FRAME_ROOT}/right_outer_edge",
}

FORBIDDEN_TYPE_COUNTS = [
    "Camera",
    "PhysicsScene",
    "PhysicsFixedJoint",
    "PhysicsRevoluteJoint",
    "PhysicsPrismaticJoint",
    "Mesh",
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
    "table_length",
    "table_width",
    "replay",
    "hdf5",
    "PhysicsScene",
    "CollisionAPI",
    "RigidBodyAPI",
    "MassAPI",
    "ArticulationRootAPI",
]


def _tuple_close(a: tuple[float, ...], b: tuple[float, ...], eps: float = 1e-6) -> bool:
    return len(a) == len(b) and all(abs(float(x) - float(y)) <= eps for x, y in zip(a, b))


def _read_xform_ops(prim: Usd.Prim) -> dict[str, tuple[float, ...]]:
    result = {}
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        value = op.Get()
        result[op.GetOpName()] = tuple(float(v) for v in value)
    return result


def _cube_bbox(
    size: float | None,
    translate: tuple[float, ...] | None,
    scale: tuple[float, ...] | None,
) -> dict[str, tuple[float, float, float]] | None:
    if size is None or translate is None or scale is None:
        return None
    half = float(size) / 2.0
    mins = tuple(float(t) - half * float(s) for t, s in zip(translate, scale))
    maxs = tuple(float(t) + half * float(s) for t, s in zip(translate, scale))
    return {"min": mins, "max": maxs}


def _merge_bbox(
    boxes: list[dict[str, tuple[float, float, float]]],
) -> dict[str, tuple[float, float, float]] | None:
    if not boxes:
        return None
    mins = tuple(min(box["min"][i] for box in boxes) for i in range(3))
    maxs = tuple(max(box["max"][i] for box in boxes) for i in range(3))
    return {"min": mins, "max": maxs}


def audit(stage_path: Path) -> dict:
    text = stage_path.read_text(encoding="utf-8")
    stage = Usd.Stage.Open(str(stage_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not open stage: {stage_path}")

    type_counts: dict[str, int] = {}
    api_counts: dict[str, int] = {}
    references = []
    payloads = []
    marker_paths = []
    marker_details = {}
    marker_boxes = []
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
        if path in EXPECTED_MARKERS:
            marker_paths.append(path)
            size = UsdGeom.Cube(prim).GetSizeAttr().Get()
            ops = _read_xform_ops(prim)
            bbox = _cube_bbox(size, ops.get("xformOp:translate"), ops.get("xformOp:scale"))
            marker_details[path] = {"size": size, "ops": ops, "bbox": bbox}
            if bbox:
                marker_boxes.append(bbox)

    default_prim = stage.GetDefaultPrim()
    default_prim_path = str(default_prim.GetPath()) if default_prim else None
    table = stage.GetPrimAtPath(TABLE_PATH)
    table_size = UsdGeom.Cube(table).GetSizeAttr().Get() if table.IsValid() else None
    table_ops = _read_xform_ops(table) if table.IsValid() else {}
    table_bbox = _cube_bbox(
        table_size,
        table_ops.get("xformOp:translate"),
        table_ops.get("xformOp:scale"),
    )
    combined_bbox = _merge_bbox(marker_boxes)

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

    marker_path_set = set(marker_paths)
    all_markers_size_one = all(
        float(detail["size"]) == 1.0 for detail in marker_details.values()
    )
    all_marker_widths_visual = all(
        EXPECTED_MARKER_WIDTH_M in (
            round(float(v), 6) for v in detail["ops"].get("xformOp:scale", ())
        )
        for detail in marker_details.values()
    )
    ok = (
        default_prim_path == EXPECTED_DEFAULT_PRIM
        and UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
        and UsdGeom.GetStageMetersPerUnit(stage) == 1.0
        and table.IsValid()
        and table.GetTypeName() == "Cube"
        and float(table_size) == 1.0
        and table_bbox is not None
        and _tuple_close(table_ops.get("xformOp:translate", ()), EXPECTED_TABLE_TRANSLATE)
        and _tuple_close(table_ops.get("xformOp:scale", ()), EXPECTED_TABLE_SCALE)
        and _tuple_close(table_bbox["min"], EXPECTED_TABLE_BBOX_MIN)
        and _tuple_close(table_bbox["max"], EXPECTED_TABLE_BBOX_MAX)
        and marker_path_set == EXPECTED_MARKERS
        and all_markers_size_one
        and all_marker_widths_visual
        and type_counts.get("Cube", 0) == 5
        and combined_bbox is not None
        and _tuple_close(combined_bbox["min"], EXPECTED_BBOX_MIN)
        and _tuple_close(combined_bbox["max"], EXPECTED_BBOX_MAX)
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
        "table_path": TABLE_PATH,
        "table_type": table.GetTypeName() if table.IsValid() else None,
        "table_size": table_size,
        "table_ops": table_ops,
        "table_bbox": table_bbox,
        "expected_table_bbox": {
            "min": EXPECTED_TABLE_BBOX_MIN,
            "max": EXPECTED_TABLE_BBOX_MAX,
        },
        "support_frame_root": SUPPORT_FRAME_ROOT,
        "marker_paths": sorted(marker_paths),
        "marker_details": marker_details,
        "combined_bbox": combined_bbox,
        "expected_combined_bbox": {"min": EXPECTED_BBOX_MIN, "max": EXPECTED_BBOX_MAX},
        "outer_dimensions": {
            "x_length_m": EXPECTED_SUPPORT_FRAME_OUTER_LENGTH_M,
            "y_width_m": EXPECTED_SUPPORT_FRAME_OUTER_WIDTH_M,
            "marker_width_m": EXPECTED_MARKER_WIDTH_M,
            "marker_height_m": EXPECTED_MARKER_HEIGHT_M,
        },
        "axis_convention": {
            "x": "support frame outer length",
            "y": "support frame outer width",
            "z": "up; visual marker height only",
        },
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
