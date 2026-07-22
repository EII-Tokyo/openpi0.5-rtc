#!/usr/bin/env python3
"""Create A4 clean ALOHA camera marker visual stage.

This stage keeps the A3 `UsdGeom.Camera` semantic prims unchanged and adds a
separate visual-only marker scope. The marker coordinates are schematic
inspection aids, not measured camera extrinsics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom

from create_aloha_camera_semantic_stage import CAMERAS, ROOT
from create_aloha_camera_semantic_stage import create_stage as create_a3_stage


MARKER_SIZE_M = (0.045, 0.045, 0.045)
DIRECTION_BAR_SIZE_M = (0.012, 0.110, 0.012)

MARKERS = {
    "cam_high": {
        "position": (0.0, -0.360, 0.100),
        "direction_position": (0.0, -0.300, 0.100),
        "direction_scale": DIRECTION_BAR_SIZE_M,
        "color": (0.15, 0.35, 1.0),
        "hint": "positive_y",
    },
    "cam_low": {
        "position": (0.0, 0.360, 0.100),
        "direction_position": (0.0, 0.300, 0.100),
        "direction_scale": DIRECTION_BAR_SIZE_M,
        "color": (0.0, 0.80, 0.25),
        "hint": "negative_y",
    },
    "cam_left_wrist": {
        "position": (-0.360, 0.0, 0.100),
        "direction_position": (-0.300, 0.0, 0.100),
        "direction_scale": (DIRECTION_BAR_SIZE_M[1], DIRECTION_BAR_SIZE_M[0], DIRECTION_BAR_SIZE_M[2]),
        "color": (1.0, 0.55, 0.05),
        "hint": "toward_center",
    },
    "cam_right_wrist": {
        "position": (0.360, 0.0, 0.100),
        "direction_position": (0.300, 0.0, 0.100),
        "direction_scale": (DIRECTION_BAR_SIZE_M[1], DIRECTION_BAR_SIZE_M[0], DIRECTION_BAR_SIZE_M[2]),
        "color": (0.80, 0.15, 1.0),
        "hint": "toward_center",
    },
}


def _set_string_attr(prim: Usd.Prim, name: str, value: str) -> None:
    prim.CreateAttribute(name, Sdf.ValueTypeNames.String).Set(value)


def _set_bool_attr(prim: Usd.Prim, name: str, value: bool) -> None:
    prim.CreateAttribute(name, Sdf.ValueTypeNames.Bool).Set(value)


def _add_visual_cube(
    stage: Usd.Stage,
    path: str,
    translate: tuple[float, float, float],
    scale: tuple[float, float, float],
    color: tuple[float, float, float],
) -> Usd.Prim:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xform.AddScaleOp().Set(Gf.Vec3d(*scale))
    return cube.GetPrim()


def _mark_visual_only(prim: Usd.Prim, camera_name: str, sensor_prim: str, stage_role: str) -> None:
    _set_string_attr(prim, "aloha:cameraName", camera_name)
    _set_string_attr(prim, "aloha:linkedSensorPrim", sensor_prim)
    _set_string_attr(prim, "aloha:sourceType", "TUNED")
    _set_string_attr(prim, "aloha:sourceKind", "manual_visual_layout_only")
    _set_string_attr(prim, "aloha:poseSource", "MANUAL_SCHEMATIC")
    _set_string_attr(prim, "aloha:calibrationStatus", "MISSING")
    _set_string_attr(prim, "aloha:intrinsicsSource", "MISSING")
    _set_string_attr(prim, "aloha:extrinsicsSource", "MISSING")
    _set_string_attr(prim, "aloha:extrinsicsStatus", "MISSING_REAL_EXTRINSICS")
    _set_string_attr(prim, "aloha:stageRole", stage_role)
    _set_bool_attr(prim, "aloha:visualMarkerOnly", True)
    _set_bool_attr(prim, "aloha:measuredPose", False)
    _set_bool_attr(prim, "aloha:forbiddenAsExtrinsics", True)
    _set_bool_attr(prim, "aloha:renderEligible", False)
    _set_bool_attr(prim, "aloha:trainingEligible", False)
    _set_bool_attr(prim, "aloha:replayEligible", False)
    _set_bool_attr(prim, "aloha:rlReady", False)
    _set_bool_attr(prim, "aloha:rosBridgeEligible", False)
    _set_bool_attr(prim, "aloha:openpiCaptureEligible", False)
    _set_bool_attr(prim, "aloha:physicsEligible", False)
    _set_bool_attr(prim, "aloha:collisionEligible", False)
    prim.SetCustomDataByKey("warning", "Schematic visual marker only; do not use as camera extrinsics.")


def _add_camera_markers(stage: Usd.Stage) -> None:
    marker_scope = UsdGeom.Scope.Define(stage, f"{ROOT}/visuals/camera_markers").GetPrim()
    _set_bool_attr(marker_scope, "aloha:visualMarkerOnly", True)
    _set_string_attr(marker_scope, "aloha:sourceKind", "manual_visual_layout_only")
    _set_bool_attr(marker_scope, "aloha:forbiddenAsExtrinsics", True)

    for camera_name, marker_spec in MARKERS.items():
        sensor_prim = CAMERAS[camera_name]["sensor"]
        marker_path = f"{ROOT}/visuals/camera_markers/{camera_name}_marker"
        marker_prim = _add_visual_cube(
            stage,
            marker_path,
            marker_spec["position"],
            MARKER_SIZE_M,
            marker_spec["color"],
        )
        _mark_visual_only(marker_prim, camera_name, sensor_prim, "A4_camera_marker_body")
        _set_string_attr(marker_prim, "aloha:directionHint", marker_spec["hint"])

        direction_prim = _add_visual_cube(
            stage,
            f"{marker_path}/view_direction_hint",
            marker_spec["direction_position"],
            marker_spec["direction_scale"],
            marker_spec["color"],
        )
        _mark_visual_only(direction_prim, camera_name, sensor_prim, "A4_camera_marker_direction_hint")


def create_stage(output_path: Path) -> None:
    create_a3_stage(output_path)
    stage = Usd.Stage.Open(str(output_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not reopen generated stage: {output_path}")

    _add_camera_markers(stage)

    root = stage.GetPrimAtPath(ROOT)
    root.SetCustomDataByKey("aloha1_rebuild_stage", "A4_camera_marker_visual")
    root.SetCustomDataByKey("camera_markers_source", "TUNED:manual_visual_layout_only")
    root.SetCustomDataByKey("camera_marker_positions_are_extrinsics", False)
    root.SetCustomDataByKey("camera_render_ready", False)
    root.SetCustomDataByKey("camera_calibration_ready", False)
    root.SetCustomDataByKey("openpi_observation_ready", False)
    root.SetCustomDataByKey("rlt_replay_ready", False)
    root.SetCustomDataByKey("rl_ready", False)
    root.SetCustomDataByKey("render_products_created", False)
    root.SetCustomDataByKey("training_eligible", False)

    stage.GetRootLayer().Save()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("aloha_isaac_rebuild/scenes/aloha_camera_marker_visual.usda"),
        help="Output USDA path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_stage(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
