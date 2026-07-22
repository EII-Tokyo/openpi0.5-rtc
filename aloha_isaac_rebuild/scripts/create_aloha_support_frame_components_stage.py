#!/usr/bin/env python3
"""Create A5 clean ALOHA Y-extension support-frame visual stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom

from create_aloha_camera_marker_visual_stage import ROOT
from create_aloha_camera_marker_visual_stage import create_stage as create_a4_stage


PROFILE_WIDTH_M = 0.020
ORIGINAL_CAM_LOW_OUTER_Y_M = 0.3125
EXTENSION_INNER_GAP_M = 0.260
EXTENSION_OUTER_RAIL_CENTER_Y_M = ORIGINAL_CAM_LOW_OUTER_Y_M + EXTENSION_INNER_GAP_M + PROFILE_WIDTH_M / 2
EXTENSION_SUPPORT_PIPE_CENTER_Y_M = ORIGINAL_CAM_LOW_OUTER_Y_M + EXTENSION_INNER_GAP_M / 2
CAM_LOW_X_M = 0.030
CAM_LOW_VERTICAL_SUPPORT_HEIGHT_M = 0.100
OUTER_LENGTH_M = 1.220
EDGE_CROSS_MEMBER_CENTER_X_M = OUTER_LENGTH_M / 2.0 - PROFILE_WIDTH_M / 2.0
Y_EXTENSION_LAYOUT_ROOT = f"{ROOT}/support_frame/visuals/y_extension_layout"
BASE_ALIGNMENT_GUIDE_ROOT = f"{ROOT}/support_frame/visuals/base_alignment_guides"

# The cam_low extension is a horizontal frame extension in the original
# support-frame plane. Only the small cam_low mount post rises along +Z.
RAIL_Z_M = PROFILE_WIDTH_M / 2.0
VERTICAL_POST_CENTER_Z_M = PROFILE_WIDTH_M + CAM_LOW_VERTICAL_SUPPORT_HEIGHT_M / 2.0

COMPONENTS = {
    "original_cam_low_side_rail": {
        "center": (0.0, ORIGINAL_CAM_LOW_OUTER_Y_M - PROFILE_WIDTH_M / 2, RAIL_Z_M),
        "size": (1.220, PROFILE_WIDTH_M, PROFILE_WIDTH_M),
        "axis": "x",
        "color": (0.16, 0.16, 0.16),
        "source_type": "MEASURED",
        "source_kind": "user_measured_20mm_square_tube",
    },
    "extension_outer_camera_rail": {
        "center": (0.0, EXTENSION_OUTER_RAIL_CENTER_Y_M, RAIL_Z_M),
        "size": (1.220, PROFILE_WIDTH_M, PROFILE_WIDTH_M),
        "axis": "x",
        "color": (0.0, 0.45, 0.16),
        "source_type": "MEASURED",
        "source_kind": "user_measured_20mm_square_tube_and_260mm_inner_gap",
    },
    "y_extension_cross_member_260mm_1": {
        "center": (-EDGE_CROSS_MEMBER_CENTER_X_M, EXTENSION_SUPPORT_PIPE_CENTER_Y_M, RAIL_Z_M),
        "size": (PROFILE_WIDTH_M, EXTENSION_INNER_GAP_M, PROFILE_WIDTH_M),
        "axis": "y",
        "color": (0.18, 0.18, 0.18),
        "source_type": "DERIVED",
        "source_kind": "derived_flush_with_1220mm_outer_rail_and_20mm_square_tube",
    },
    "y_extension_cross_member_260mm_2": {
        "center": (-0.433554, EXTENSION_SUPPORT_PIPE_CENTER_Y_M, RAIL_Z_M),
        "size": (PROFILE_WIDTH_M, EXTENSION_INNER_GAP_M, PROFILE_WIDTH_M),
        "axis": "y",
        "color": (0.18, 0.18, 0.18),
        "source_type": "MEASURED",
        "source_kind": "user_measured_20mm_square_tube_and_260mm_inner_gap",
    },
    "y_extension_cross_member_260mm_3": {
        "center": (0.433554, EXTENSION_SUPPORT_PIPE_CENTER_Y_M, RAIL_Z_M),
        "size": (PROFILE_WIDTH_M, EXTENSION_INNER_GAP_M, PROFILE_WIDTH_M),
        "axis": "y",
        "color": (0.18, 0.18, 0.18),
        "source_type": "MEASURED",
        "source_kind": "user_measured_20mm_square_tube_and_260mm_inner_gap",
    },
    "y_extension_cross_member_260mm_4": {
        "center": (EDGE_CROSS_MEMBER_CENTER_X_M, EXTENSION_SUPPORT_PIPE_CENTER_Y_M, RAIL_Z_M),
        "size": (PROFILE_WIDTH_M, EXTENSION_INNER_GAP_M, PROFILE_WIDTH_M),
        "axis": "y",
        "color": (0.18, 0.18, 0.18),
        "source_type": "DERIVED",
        "source_kind": "derived_flush_with_1220mm_outer_rail_and_20mm_square_tube",
    },
    "cam_low_vertical_mount_post_100mm": {
        "center": (
            CAM_LOW_X_M,
            EXTENSION_OUTER_RAIL_CENTER_Y_M,
            VERTICAL_POST_CENTER_Z_M,
        ),
        "size": (PROFILE_WIDTH_M, PROFILE_WIDTH_M, CAM_LOW_VERTICAL_SUPPORT_HEIGHT_M),
        "axis": "z",
        "color": (0.0, 0.62, 0.20),
        "source_type": "MEASURED",
        "source_kind": "user_measured_20mm_square_tube_and_100mm_vertical_support",
    },
}

ALIGNMENT_GUIDES = {
    "base_edge_near_cam_low_y_guide": {
        "center": (0.0, 0.1325, RAIL_Z_M),
        "size": (1.220, 0.008, 0.008),
        "axis": "x",
        "color": (1.0, 0.62, 0.0),
        "source_kind": "user_measured_base_y_edges",
        "description": "Base outer edge nearer cam_low is 180 mm inward from original cam_low-side frame edge.",
    },
    "base_edge_near_cam_high_y_guide": {
        "center": (0.0, -0.0775, RAIL_Z_M),
        "size": (1.220, 0.008, 0.008),
        "axis": "x",
        "color": (1.0, 0.62, 0.0),
        "source_kind": "user_measured_base_y_edges",
        "description": "Base outer edge nearer cam_high is 235 mm inward from original cam_high-side frame edge.",
    },
}


def _set_string_attr(prim: Usd.Prim, name: str, value: str) -> None:
    prim.CreateAttribute(name, Sdf.ValueTypeNames.String).Set(value)


def _set_bool_attr(prim: Usd.Prim, name: str, value: bool) -> None:
    prim.CreateAttribute(name, Sdf.ValueTypeNames.Bool).Set(value)


def _add_component_cube(stage: Usd.Stage, name: str, spec: dict) -> Usd.Prim:
    path = f"{Y_EXTENSION_LAYOUT_ROOT}/{name}"
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*spec["color"])])
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*spec["center"]))
    xform.AddScaleOp().Set(Gf.Vec3d(*spec["size"]))
    prim = cube.GetPrim()
    _set_string_attr(prim, "aloha:componentName", name)
    _set_string_attr(prim, "aloha:sourceType", spec.get("source_type", "TUNED"))
    _set_string_attr(prim, "aloha:sourceKind", spec.get("source_kind", "aloha1_visual_layout_v0"))
    if spec.get("source_type") == "MEASURED":
        measurement_status = "MEASURED"
    elif spec.get("source_type") == "DERIVED":
        measurement_status = "DERIVED_FROM_MEASURED"
    else:
        measurement_status = "PENDING_MEASUREMENT"
    _set_string_attr(prim, "aloha:measurementStatus", measurement_status)
    _set_string_attr(prim, "aloha:stageRole", "A5_support_frame_visual_component")
    _set_string_attr(prim, "aloha:axis", spec["axis"])
    _set_bool_attr(prim, "aloha:visualOnly", True)
    _set_bool_attr(prim, "aloha:measuredCadReady", False)
    _set_bool_attr(prim, "aloha:physicsEligible", False)
    _set_bool_attr(prim, "aloha:collisionEligible", False)
    _set_bool_attr(prim, "aloha:cameraExtrinsicsEligible", False)
    _set_bool_attr(prim, "aloha:trainingEligible", False)
    prim.SetCustomDataByKey("warning", "Visual-only support-frame proxy; do not use as final CAD, collision, or camera extrinsics.")
    return prim


def _add_alignment_guide_cube(stage: Usd.Stage, name: str, spec: dict) -> Usd.Prim:
    path = f"{BASE_ALIGNMENT_GUIDE_ROOT}/{name}"
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*spec["color"])])
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*spec["center"]))
    xform.AddScaleOp().Set(Gf.Vec3d(*spec["size"]))
    prim = cube.GetPrim()
    _set_string_attr(prim, "aloha:guideName", name)
    _set_string_attr(prim, "aloha:sourceType", "MEASURED")
    _set_string_attr(prim, "aloha:sourceKind", spec["source_kind"])
    _set_string_attr(prim, "aloha:measurementStatus", "MEASURED")
    _set_string_attr(prim, "aloha:stageRole", "A5_base_y_alignment_guide")
    _set_string_attr(prim, "aloha:axis", spec["axis"])
    _set_string_attr(prim, "aloha:description", spec["description"])
    _set_bool_attr(prim, "aloha:visualOnly", True)
    _set_bool_attr(prim, "aloha:baseGeometryComplete", False)
    _set_bool_attr(prim, "aloha:physicsEligible", False)
    _set_bool_attr(prim, "aloha:collisionEligible", False)
    _set_bool_attr(prim, "aloha:trainingEligible", False)
    prim.SetCustomDataByKey("warning", "Measured Y-edge guide only; not a full ALOHA base mesh or collider.")
    return prim


def _add_support_frame_components(stage: Usd.Stage) -> None:
    scope = UsdGeom.Scope.Define(stage, Y_EXTENSION_LAYOUT_ROOT).GetPrim()
    _set_bool_attr(scope, "aloha:visualOnly", True)
    _set_string_attr(scope, "aloha:sourceKind", "aloha1_y_extension_visual_layout_v1")
    _set_string_attr(scope, "aloha:layoutDirection", "positive_y")
    _set_bool_attr(scope, "aloha:measuredCadReady", False)
    _set_bool_attr(scope, "aloha:physicsEligible", False)
    _set_bool_attr(scope, "aloha:collisionEligible", False)
    for name, spec in COMPONENTS.items():
        _add_component_cube(stage, name, spec)

    guides = UsdGeom.Scope.Define(stage, BASE_ALIGNMENT_GUIDE_ROOT).GetPrim()
    _set_bool_attr(guides, "aloha:visualOnly", True)
    _set_string_attr(guides, "aloha:sourceKind", "user_measured_base_y_edges")
    for name, spec in ALIGNMENT_GUIDES.items():
        _add_alignment_guide_cube(stage, name, spec)


def create_stage(output_path: Path) -> None:
    create_a4_stage(output_path)
    stage = Usd.Stage.Open(str(output_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not reopen generated stage: {output_path}")

    _add_support_frame_components(stage)

    root = stage.GetPrimAtPath(ROOT)
    root.SetCustomDataByKey("aloha1_rebuild_stage", "A5_support_frame_components_visual")
    root.SetCustomDataByKey("support_frame_components_source", "TUNED:aloha1_y_extension_visual_layout_v1")
    root.SetCustomDataByKey("support_frame_components_visual_only", True)
    root.SetCustomDataByKey("support_frame_final_cad_ready", False)
    root.SetCustomDataByKey("support_frame_physics_ready", False)
    root.SetCustomDataByKey("support_frame_collision_ready", False)
    root.SetCustomDataByKey("base_y_edges_source", "MEASURED:user_measured_base_y_edges")
    root.SetCustomDataByKey("support_frame_measurement_status", "MEASURED")
    root.SetCustomDataByKey("base_x_extent_ready", False)

    stage.GetRootLayer().Save()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("aloha_isaac_rebuild/scenes/aloha_support_frame_components_visual.usda"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_stage(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
