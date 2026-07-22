#!/usr/bin/env python3
"""Create A3 clean ALOHA camera semantic skeleton stage.

A3 adds true UsdGeom.Camera prims for the four canonical ALOHA/OpenPI camera
channels, but explicitly leaves them uncalibrated and ineligible for rendering,
ROS, OpenPI capture, or training. Mount visuals are intentionally absent here:
camera sensor identity must not depend on D405/mount meshes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom


ROOT = "/aloha"
TABLE_REFERENCE_LENGTH_M = 1.2192
TABLE_REFERENCE_WIDTH_M = 0.7490
TABLE_REFERENCE_THICKNESS_M = 0.0200
SUPPORT_FRAME_OUTER_LENGTH_M = 1.220
SUPPORT_FRAME_OUTER_WIDTH_M = 0.625
MARKER_WIDTH_M = 0.010
MARKER_HEIGHT_M = 0.010
MARKER_Z_M = 0.0

CAMERAS = {
    "cam_high": {
        "semantic_role": "fixed_high_third_person",
        "parent_link": f"{ROOT}/cam_high_link",
        "color_frame": f"{ROOT}/cam_high_link/cam_high_color_frame",
        "sensor": f"{ROOT}/cam_high_link/cam_high_color_frame/cam_high",
        "openpi_obs_key": "cam_high",
    },
    "cam_low": {
        "semantic_role": "fixed_low_third_person",
        "parent_link": f"{ROOT}/cam_low_link",
        "color_frame": f"{ROOT}/cam_low_link/cam_low_color_frame",
        "sensor": f"{ROOT}/cam_low_link/cam_low_color_frame/cam_low",
        "openpi_obs_key": "cam_low",
    },
    "cam_left_wrist": {
        "semantic_role": "left_wrist",
        "parent_link": f"{ROOT}/left_camera_link",
        "color_frame": f"{ROOT}/left_camera_link/cam_left_wrist_color_frame",
        "sensor": f"{ROOT}/left_camera_link/cam_left_wrist_color_frame/cam_left_wrist",
        "openpi_obs_key": "cam_left_wrist",
    },
    "cam_right_wrist": {
        "semantic_role": "right_wrist",
        "parent_link": f"{ROOT}/right_camera_link",
        "color_frame": f"{ROOT}/right_camera_link/cam_right_wrist_color_frame",
        "sensor": f"{ROOT}/right_camera_link/cam_right_wrist_color_frame/cam_right_wrist",
        "openpi_obs_key": "cam_right_wrist",
    },
}


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


def _add_a2_visual_context(stage: Usd.Stage) -> None:
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


def _set_string_attr(prim: Usd.Prim, name: str, value: str) -> None:
    prim.CreateAttribute(name, Sdf.ValueTypeNames.String).Set(value)


def _set_bool_attr(prim: Usd.Prim, name: str, value: bool) -> None:
    prim.CreateAttribute(name, Sdf.ValueTypeNames.Bool).Set(value)


def _add_camera_semantics(stage: Usd.Stage) -> None:
    UsdGeom.Scope.Define(stage, f"{ROOT}/camera_registry")
    for name, spec in CAMERAS.items():
        UsdGeom.Xform.Define(stage, spec["color_frame"])
        camera = UsdGeom.Camera.Define(stage, spec["sensor"])
        prim = camera.GetPrim()
        _set_string_attr(prim, "aloha:semanticName", name)
        _set_string_attr(prim, "aloha:semanticRole", spec["semantic_role"])
        _set_string_attr(prim, "aloha:openpiObsKey", spec["openpi_obs_key"])
        _set_string_attr(prim, "aloha:calibrationStatus", "MISSING")
        _set_string_attr(prim, "aloha:intrinsicsStatus", "MISSING")
        _set_string_attr(prim, "aloha:extrinsicsStatus", "MISSING")
        _set_string_attr(prim, "aloha:sourceType", "MISSING")
        _set_string_attr(prim, "aloha:stageRole", "A3_semantic_sensor_placeholder")
        _set_bool_attr(prim, "aloha:renderEligible", False)
        _set_bool_attr(prim, "aloha:trainingEligible", False)
        _set_bool_attr(prim, "aloha:rosBridgeEligible", False)
        _set_bool_attr(prim, "aloha:openpiCaptureEligible", False)
        prim.SetCustomDataByKey("sensor_semantic_note", "UsdGeom.Camera exists; intrinsics and extrinsics are not calibrated.")
        prim.SetCustomDataByKey("mount_visual_note", "D405/mount visual meshes must remain separate from this sensor prim.")


def create_stage(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    stage = Usd.Stage.CreateNew(str(output_path))
    _create_base_skeleton(stage)
    _add_a2_visual_context(stage)
    _add_camera_semantics(stage)

    root = stage.GetPrimAtPath(ROOT)
    root.SetCustomDataByKey("aloha1_rebuild_stage", "A3_camera_semantic_skeleton")
    root.SetCustomDataByKey("camera_semantic_source", "CANONICAL_ALOHA_OPENPI_NAMES")
    root.SetCustomDataByKey("camera_calibration_ready", False)
    root.SetCustomDataByKey("render_products_created", False)
    root.SetCustomDataByKey("training_eligible", False)

    stage.GetRootLayer().Save()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("aloha_isaac_rebuild/scenes/aloha_camera_semantic_skeleton.usda"),
        help="Output USDA path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_stage(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
