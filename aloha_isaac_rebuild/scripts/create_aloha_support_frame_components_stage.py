#!/usr/bin/env python3
"""Create A5 clean ALOHA support-frame component visual stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom

from create_aloha_camera_marker_visual_stage import ROOT
from create_aloha_camera_marker_visual_stage import create_stage as create_a4_stage


COMPONENTS = {
    "top_steel_rail_proxy": {
        "center": (0.0, 0.3125, 0.610),
        "size": (1.220, 0.020, 0.020),
        "axis": "x",
        "color": (0.16, 0.16, 0.16),
    },
    "extension_outer_camera_rail_proxy": {
        "center": (0.0, 0.5725, 0.610),
        "size": (1.220, 0.020, 0.020),
        "axis": "x",
        "color": (0.0, 0.45, 0.16),
    },
    "support_pipe_260mm_1": {
        "center": (-0.604959, 0.4425, 0.610),
        "size": (0.020, 0.260, 0.020),
        "axis": "y",
        "color": (0.18, 0.18, 0.18),
    },
    "support_pipe_260mm_2": {
        "center": (-0.433554, 0.4425, 0.610),
        "size": (0.020, 0.260, 0.020),
        "axis": "y",
        "color": (0.18, 0.18, 0.18),
    },
    "support_pipe_260mm_3": {
        "center": (0.433554, 0.4425, 0.610),
        "size": (0.020, 0.260, 0.020),
        "axis": "y",
        "color": (0.18, 0.18, 0.18),
    },
    "support_pipe_260mm_4": {
        "center": (0.604959, 0.4425, 0.610),
        "size": (0.020, 0.260, 0.020),
        "axis": "y",
        "color": (0.18, 0.18, 0.18),
    },
}

ALIGNMENT_GUIDES = {
    "base_edge_near_cam_low_y_guide": {
        "center": (0.0, 0.1325, 0.615),
        "size": (1.220, 0.008, 0.008),
        "axis": "x",
        "color": (1.0, 0.62, 0.0),
        "source_kind": "user_measured_base_y_edges",
        "description": "Base outer edge nearer cam_low is 180 mm inward from original cam_low-side frame edge.",
    },
    "base_edge_near_cam_high_y_guide": {
        "center": (0.0, -0.0775, 0.615),
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
    path = f"{ROOT}/support_frame/visuals/components/{name}"
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*spec["color"])])
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*spec["center"]))
    xform.AddScaleOp().Set(Gf.Vec3d(*spec["size"]))
    prim = cube.GetPrim()
    _set_string_attr(prim, "aloha:componentName", name)
    _set_string_attr(prim, "aloha:sourceType", "TUNED")
    _set_string_attr(prim, "aloha:sourceKind", "aloha1_visual_layout_v0")
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
    path = f"{ROOT}/support_frame/visuals/alignment_guides/{name}"
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
    scope = UsdGeom.Scope.Define(stage, f"{ROOT}/support_frame/visuals/components").GetPrim()
    _set_bool_attr(scope, "aloha:visualOnly", True)
    _set_string_attr(scope, "aloha:sourceKind", "aloha1_visual_layout_v0")
    _set_bool_attr(scope, "aloha:measuredCadReady", False)
    _set_bool_attr(scope, "aloha:physicsEligible", False)
    _set_bool_attr(scope, "aloha:collisionEligible", False)
    for name, spec in COMPONENTS.items():
        _add_component_cube(stage, name, spec)

    guides = UsdGeom.Scope.Define(stage, f"{ROOT}/support_frame/visuals/alignment_guides").GetPrim()
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
    root.SetCustomDataByKey("support_frame_components_source", "TUNED:aloha1_visual_layout_v0")
    root.SetCustomDataByKey("support_frame_components_visual_only", True)
    root.SetCustomDataByKey("support_frame_final_cad_ready", False)
    root.SetCustomDataByKey("support_frame_physics_ready", False)
    root.SetCustomDataByKey("support_frame_collision_ready", False)
    root.SetCustomDataByKey("base_y_edges_source", "MEASURED:user:cam_low_edge_180mm_cam_high_edge_235mm")
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
