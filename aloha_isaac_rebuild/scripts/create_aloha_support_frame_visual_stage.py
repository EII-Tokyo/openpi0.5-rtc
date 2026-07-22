#!/usr/bin/env python3
"""Create A2 visual-only ALOHA support frame footprint stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Gf, Usd, UsdGeom


ROOT = "/aloha"
TABLE_REFERENCE_LENGTH_M = 1.2192
TABLE_REFERENCE_WIDTH_M = 0.7490
TABLE_REFERENCE_THICKNESS_M = 0.0200
SUPPORT_FRAME_OUTER_LENGTH_M = 1.220
SUPPORT_FRAME_OUTER_WIDTH_M = 0.625
MARKER_WIDTH_M = 0.010
MARKER_HEIGHT_M = 0.010
MARKER_Z_M = 0.0


def _create_base_skeleton(stage: Usd.Stage) -> None:
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, ROOT).GetPrim()
    stage.SetDefaultPrim(root)

    for path in ["/Render", "/meshes", "/visuals", "/colliders"]:
        UsdGeom.Scope.Define(stage, path)

    for name, prim_type in {
        "Looks": "Scope",
        "joints": "Scope",
        "table_link": "Xform",
        "support_frame": "Xform",
        "cam_high_link": "Xform",
        "cam_low_link": "Xform",
        "left_base_link": "Xform",
        "right_base_link": "Xform",
        "left_camera_link": "Xform",
        "right_camera_link": "Xform",
    }.items():
        path = f"{ROOT}/{name}"
        if prim_type == "Scope":
            UsdGeom.Scope.Define(stage, path)
        else:
            UsdGeom.Xform.Define(stage, path)


def _add_cube_marker(
    stage: Usd.Stage,
    path: str,
    translate: tuple[float, float, float],
    scale: tuple[float, float, float],
    color: tuple[float, float, float],
) -> None:
    marker = UsdGeom.Cube.Define(stage, path)
    marker.CreateSizeAttr(1.0)
    marker.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    xform = UsdGeom.Xformable(marker.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xform.AddScaleOp().Set(Gf.Vec3d(*scale))


def create_stage(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    stage = Usd.Stage.CreateNew(str(output_path))
    _create_base_skeleton(stage)

    UsdGeom.Scope.Define(stage, f"{ROOT}/table_link/visuals")
    _add_cube_marker(
        stage,
        f"{ROOT}/table_link/visuals/tabletop_reference_proxy",
        (0.0, 0.0, -TABLE_REFERENCE_THICKNESS_M / 2.0),
        (TABLE_REFERENCE_LENGTH_M, TABLE_REFERENCE_WIDTH_M, TABLE_REFERENCE_THICKNESS_M),
        (0.72, 0.66, 0.54),
    )

    UsdGeom.Scope.Define(stage, f"{ROOT}/support_frame/visuals")
    base = f"{ROOT}/support_frame/visuals/outer_footprint"
    UsdGeom.Xform.Define(stage, base)

    half_x = SUPPORT_FRAME_OUTER_LENGTH_M / 2.0
    half_y = SUPPORT_FRAME_OUTER_WIDTH_M / 2.0
    half_marker = MARKER_WIDTH_M / 2.0
    marker_center_z = MARKER_Z_M + MARKER_HEIGHT_M / 2.0

    # These are visual guides for the outer support frame footprint, not physical steel profiles.
    _add_cube_marker(
        stage,
        f"{base}/front_outer_edge",
        (0.0, -half_y + half_marker, marker_center_z),
        (SUPPORT_FRAME_OUTER_LENGTH_M, MARKER_WIDTH_M, MARKER_HEIGHT_M),
        (0.05, 0.85, 0.20),
    )
    _add_cube_marker(
        stage,
        f"{base}/back_outer_edge",
        (0.0, half_y - half_marker, marker_center_z),
        (SUPPORT_FRAME_OUTER_LENGTH_M, MARKER_WIDTH_M, MARKER_HEIGHT_M),
        (0.05, 0.85, 0.20),
    )
    _add_cube_marker(
        stage,
        f"{base}/left_outer_edge",
        (-half_x + half_marker, 0.0, marker_center_z),
        (MARKER_WIDTH_M, SUPPORT_FRAME_OUTER_WIDTH_M, MARKER_HEIGHT_M),
        (0.05, 0.85, 0.20),
    )
    _add_cube_marker(
        stage,
        f"{base}/right_outer_edge",
        (half_x - half_marker, 0.0, marker_center_z),
        (MARKER_WIDTH_M, SUPPORT_FRAME_OUTER_WIDTH_M, MARKER_HEIGHT_M),
        (0.05, 0.85, 0.20),
    )

    root = stage.GetPrimAtPath(ROOT)
    root.SetCustomDataByKey("aloha1_rebuild_stage", "A2_support_frame_visual_only")
    root.SetCustomDataByKey("table_reference_source", "TROSSEN_STATIONARY_AI_VISUAL_BBOX")
    root.SetCustomDataByKey("support_frame_outer_length_source", "MEASURED:user:1220mm")
    root.SetCustomDataByKey("support_frame_outer_width_source", "MEASURED:user:625mm")
    root.SetCustomDataByKey("support_frame_marker_source", "ESTIMATED_VISUAL_ONLY")

    stage.GetRootLayer().Save()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("aloha_isaac_rebuild/scenes/aloha_support_frame_visual.usda"),
        help="Output USDA path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_stage(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
