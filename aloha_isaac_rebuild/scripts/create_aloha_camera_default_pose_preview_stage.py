#!/usr/bin/env python3
"""Create A7 clean ALOHA camera default-pose preview stage."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from pxr import Gf, Sdf, Usd, UsdGeom

from create_aloha_support_frame_components_stage import ROOT
from create_aloha_support_frame_components_stage import create_stage as create_a5_stage


DEFAULT_CONFIG = Path("aloha_isaac_rebuild/configs/physical_reconstruction/camera_default_pose_preview.yaml")
DEFAULT_OUTPUT = Path("aloha_isaac_rebuild/scenes/aloha_camera_default_pose_preview.usda")

COLORS = {
    "cam_high": (0.15, 0.35, 1.0),
    "cam_low": (0.0, 0.80, 0.25),
    "cam_left_wrist": (1.0, 0.55, 0.05),
    "cam_right_wrist": (0.80, 0.15, 1.0),
}


def _set_string_attr(prim: Usd.Prim, name: str, value: str) -> None:
    prim.CreateAttribute(name, Sdf.ValueTypeNames.String).Set(value)


def _set_bool_attr(prim: Usd.Prim, name: str, value: bool) -> None:
    prim.CreateAttribute(name, Sdf.ValueTypeNames.Bool).Set(value)


def _set_double3_attr(prim: Usd.Prim, name: str, value: list[float]) -> None:
    prim.CreateAttribute(name, Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(*value))


def _add_visual_cube(
    stage: Usd.Stage,
    path: str,
    translate: list[float],
    scale: tuple[float, float, float] | list[float],
    color: tuple[float, float, float],
) -> Usd.Prim:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xform.AddScaleOp().Set(Gf.Vec3d(*scale))
    return cube.GetPrim()


def _direction_bar_scale(direction: list[float], base_size: list[float]) -> tuple[float, float, float]:
    if abs(direction[0]) >= abs(direction[1]):
        return (base_size[1], base_size[0], base_size[2])
    return tuple(base_size)


def _direction_bar_center(center: list[float], direction: list[float], length_m: float) -> list[float]:
    return [center[i] + direction[i] * length_m / 2.0 for i in range(3)]


def _mark_preview_prim(prim: Usd.Prim, camera_name: str, sensor_prim: str, role: str) -> None:
    _set_string_attr(prim, "aloha:cameraName", camera_name)
    _set_string_attr(prim, "aloha:linkedSensorPrim", sensor_prim)
    _set_string_attr(prim, "aloha:sourceType", "TUNED")
    _set_string_attr(prim, "aloha:sourceKind", "default_placeholder_until_measured")
    _set_string_attr(prim, "aloha:stageRole", role)
    _set_string_attr(prim, "aloha:calibrationStatus", "DEFAULT_PLACEHOLDER_NOT_MEASURED")
    _set_string_attr(prim, "aloha:extrinsicsStatus", "DEFAULT_PLACEHOLDER_NOT_MEASURED")
    _set_bool_attr(prim, "aloha:visualPreviewOnly", True)
    _set_bool_attr(prim, "aloha:measuredPose", False)
    _set_bool_attr(prim, "aloha:renderEligible", False)
    _set_bool_attr(prim, "aloha:trainingEligible", False)
    _set_bool_attr(prim, "aloha:rosBridgeEligible", False)
    _set_bool_attr(prim, "aloha:openpiCaptureEligible", False)
    _set_bool_attr(prim, "aloha:physicsEligible", False)
    _set_bool_attr(prim, "aloha:collisionEligible", False)
    prim.SetCustomDataByKey("warning", "Default camera pose preview only; replace with measured extrinsics before rendering/training.")


def _annotate_camera_prim(prim: Usd.Prim, camera_name: str, center: list[float], direction: list[float]) -> None:
    _set_string_attr(prim, "aloha:defaultPoseSource", "TUNED:default_placeholder_until_measured")
    _set_string_attr(prim, "aloha:extrinsicsStatus", "DEFAULT_PLACEHOLDER_NOT_MEASURED")
    _set_bool_attr(prim, "aloha:defaultPosePreviewOnly", True)
    _set_bool_attr(prim, "aloha:measuredPose", False)
    _set_bool_attr(prim, "aloha:renderEligible", False)
    _set_bool_attr(prim, "aloha:trainingEligible", False)
    _set_double3_attr(prim, "aloha:defaultOpticalCenterWorld", center)
    _set_double3_attr(prim, "aloha:defaultDirectionWorld", direction)
    prim.SetCustomDataByKey("default_pose_warning", f"{camera_name} default pose is a placeholder, not measured extrinsics.")


def create_stage(output_path: Path, config_path: Path) -> None:
    create_a5_stage(output_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stage = Usd.Stage.Open(str(output_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not reopen generated stage: {output_path}")

    preview_scope = UsdGeom.Scope.Define(stage, f"{ROOT}/visuals/camera_default_pose_preview").GetPrim()
    _set_string_attr(preview_scope, "aloha:sourceType", "TUNED")
    _set_string_attr(preview_scope, "aloha:sourceKind", "default_placeholder_until_measured")
    _set_bool_attr(preview_scope, "aloha:visualPreviewOnly", True)
    _set_bool_attr(preview_scope, "aloha:measuredPose", False)
    _set_bool_attr(preview_scope, "aloha:renderEligible", False)
    _set_bool_attr(preview_scope, "aloha:trainingEligible", False)

    marker_size = config["preview_marker_size_m"]
    direction_base_size = config["preview_direction_bar_size_m"]
    length_m = max(direction_base_size[0], direction_base_size[1], direction_base_size[2])

    for camera_name, spec in config["cameras"].items():
        center = spec["default_optical_center_m"]
        direction = spec["default_direction_vector_world"]
        sensor_prim_path = spec["sensor_prim"]
        color = COLORS[camera_name]
        marker_path = f"{ROOT}/visuals/camera_default_pose_preview/{camera_name}_default_optical_center"
        marker_prim = _add_visual_cube(stage, marker_path, center, marker_size, color)
        _mark_preview_prim(marker_prim, camera_name, sensor_prim_path, "A7_default_optical_center_marker")

        bar_center = _direction_bar_center(center, direction, length_m)
        bar_scale = _direction_bar_scale(direction, direction_base_size)
        bar_prim = _add_visual_cube(stage, f"{marker_path}/default_direction_hint", bar_center, bar_scale, color)
        _mark_preview_prim(bar_prim, camera_name, sensor_prim_path, "A7_default_direction_hint")
        _set_double3_attr(bar_prim, "aloha:defaultDirectionWorld", direction)

        camera_prim = stage.GetPrimAtPath(sensor_prim_path)
        if not camera_prim:
            raise RuntimeError(f"Missing camera prim for {camera_name}: {sensor_prim_path}")
        _annotate_camera_prim(camera_prim, camera_name, center, direction)

    root = stage.GetPrimAtPath(ROOT)
    root.SetCustomDataByKey("aloha1_rebuild_stage", "A7_camera_default_pose_preview")
    root.SetCustomDataByKey("camera_default_pose_preview_only", True)
    root.SetCustomDataByKey("camera_default_pose_source", "TUNED:default_placeholder_until_measured")
    root.SetCustomDataByKey("camera_calibration_ready", False)
    root.SetCustomDataByKey("camera_render_ready", False)
    root.SetCustomDataByKey("openpi_observation_ready", False)
    root.SetCustomDataByKey("training_eligible", False)
    root.SetCustomDataByKey("rl_ready", False)
    stage.GetRootLayer().Save()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_stage(args.output, args.config)
    print(args.output)


if __name__ == "__main__":
    main()
