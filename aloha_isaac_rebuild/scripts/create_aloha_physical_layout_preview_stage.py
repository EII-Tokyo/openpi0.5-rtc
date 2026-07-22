#!/usr/bin/env python3
"""Create A9 visual-only non-camera physical-layout preview stage."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from pxr import Gf, Sdf, Usd, UsdGeom

from create_aloha_support_frame_components_stage import ROOT
from create_aloha_camera_default_pose_preview_stage import create_stage as create_a7_stage


DEFAULT_CONFIG = Path("aloha_isaac_rebuild/configs/physical_reconstruction/physical_layout.yaml")
DEFAULT_OUTPUT = Path("aloha_isaac_rebuild/scenes/aloha_physical_layout_preview.usda")

COLORS = {
    "table": (0.38, 0.38, 0.34),
    "base_y": (1.0, 0.58, 0.05),
    "origin": (0.95, 0.95, 0.95),
    "x_axis": (0.90, 0.10, 0.10),
    "y_axis": (0.05, 0.70, 0.18),
    "z_axis": (0.10, 0.35, 0.95),
}


def _set_string_attr(prim: Usd.Prim, name: str, value: str) -> None:
    prim.CreateAttribute(name, Sdf.ValueTypeNames.String).Set(value)


def _set_bool_attr(prim: Usd.Prim, name: str, value: bool) -> None:
    prim.CreateAttribute(name, Sdf.ValueTypeNames.Bool).Set(value)


def _set_double3_attr(prim: Usd.Prim, name: str, value: list[float] | tuple[float, float, float]) -> None:
    prim.CreateAttribute(name, Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(*value))


def _add_visual_cube(
    stage: Usd.Stage,
    path: str,
    translate: list[float] | tuple[float, float, float],
    scale: list[float] | tuple[float, float, float],
    color: tuple[float, float, float],
    stage_role: str,
    source_type: str,
    source_kind: str,
    *,
    visibility: str = "inherited",
    opacity: float | None = None,
) -> Usd.Prim:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if opacity is not None:
        cube.CreateDisplayOpacityAttr([opacity])
    if visibility == "hidden":
        UsdGeom.Imageable(cube.GetPrim()).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xform.AddScaleOp().Set(Gf.Vec3d(*scale))
    prim = cube.GetPrim()
    _set_string_attr(prim, "aloha:stageRole", stage_role)
    _set_string_attr(prim, "aloha:sourceType", source_type)
    _set_string_attr(prim, "aloha:sourceKind", source_kind)
    _set_bool_attr(prim, "aloha:visualOnly", True)
    _set_bool_attr(prim, "aloha:physicsEligible", False)
    _set_bool_attr(prim, "aloha:collisionEligible", False)
    _set_bool_attr(prim, "aloha:renderEligible", False)
    _set_bool_attr(prim, "aloha:trainingEligible", False)
    prim.SetCustomDataByKey("warning", "A9 visual-only physical-layout reference; not physics, collision, replay, or training geometry.")
    return prim


def _add_table_outline(stage: Usd.Stage, config: dict) -> None:
    spec = config["table_reference"]
    root = config["visuals_root"] + "/table_reference_outline"
    UsdGeom.Scope.Define(stage, root)
    sx, sy, _sz = spec["outer_size_m"]
    width = spec["outline_width_m"]
    z = spec["outline_z_m"]
    half_x = sx / 2.0
    half_y = sy / 2.0
    half_width = width / 2.0
    role = "A9_table_reference_outline"
    for name, center, scale in [
        ("x_min_edge", (-half_x + half_width, 0.0, z), (width, sy, width)),
        ("x_max_edge", (half_x - half_width, 0.0, z), (width, sy, width)),
        ("y_min_edge", (0.0, -half_y + half_width, z), (sx, width, width)),
        ("y_max_edge", (0.0, half_y - half_width, z), (sx, width, width)),
    ]:
        prim = _add_visual_cube(
            stage,
            f"{root}/{name}",
            center,
            scale,
            COLORS["table"],
            role,
            spec["source_type"],
            spec["source_kind"],
            visibility=spec.get("visibility", "inherited"),
            opacity=0.18,
        )
        _set_bool_attr(prim, "aloha:referenceOnly", True)
        _set_bool_attr(prim, "aloha:supportFrameMeasurement", False)
        _set_bool_attr(prim, "aloha:mainFrameMeasurement", False)


def _add_world_frame(stage: Usd.Stage, config: dict) -> None:
    spec = config["world_frame_visual"]
    root = config["visuals_root"] + "/world_frame_reference"
    UsdGeom.Scope.Define(stage, root)
    origin = spec["origin_m"]
    length = spec["axis_length_m"]
    width = spec["axis_width_m"]
    _add_visual_cube(stage, f"{root}/origin", origin, (width * 1.8, width * 1.8, width * 1.8), COLORS["origin"], "A9_world_frame_origin", "TUNED", "visual_axis_reference")
    axes = [
        ("positive_x_axis", (origin[0] + length / 2.0, origin[1], origin[2]), (length, width, width), COLORS["x_axis"], "support_frame_outer_length_direction"),
        ("positive_y_axis", (origin[0], origin[1] + length / 2.0, origin[2]), (width, length, width), COLORS["y_axis"], "toward_cam_low_extension"),
        ("positive_z_axis", (origin[0], origin[1], origin[2] + length / 2.0), (width, width, length), COLORS["z_axis"], "up"),
    ]
    for name, center, scale, color, meaning in axes:
        prim = _add_visual_cube(stage, f"{root}/{name}", center, scale, color, "A9_world_frame_axis", "TUNED", "visual_axis_reference")
        _set_string_attr(prim, "aloha:axisMeaning", meaning)


def _add_base_y_reference(stage: Usd.Stage, config: dict) -> None:
    spec = config["base_y_reference"]
    root = config["visuals_root"] + "/base_y_reference"
    UsdGeom.Scope.Define(stage, root)
    size = spec["guide_size_m"]
    z = spec["guide_z_m"]
    for name, y_value, description in [
        ("near_cam_low_y_edge", spec["near_cam_low_y_m"], "User-measured base edge nearer cam_low side; Y reference only."),
        ("near_cam_high_y_edge", spec["near_cam_high_y_m"], "User-measured base edge nearer cam_high side; Y reference only."),
    ]:
        prim = _add_visual_cube(
            stage,
            f"{root}/{name}",
            (0.0, y_value, z),
            size,
            COLORS["base_y"],
            "A9_base_y_reference_guide",
            spec["source_type"],
            spec["source_kind"],
        )
        _set_string_attr(prim, "aloha:description", description)
        _set_bool_attr(prim, "aloha:fullBaseFootprint", False)
        _set_bool_attr(prim, "aloha:xExtentMeasured", False)


def _add_future_measurement_scopes(stage: Usd.Stage, config: dict) -> None:
    root = UsdGeom.Scope.Define(stage, config["physical_layout_root"] + "/future_measurements").GetPrim()
    _set_bool_attr(root, "aloha:visualOnly", True)
    _set_bool_attr(root, "aloha:allFutureMeasurementValuesMissing", True)
    for name, spec in config["future_measurements"].items():
        prim = UsdGeom.Scope.Define(stage, f"{config['physical_layout_root']}/future_measurements/{name}").GetPrim()
        _set_string_attr(prim, "aloha:status", spec["status"])
        _set_bool_attr(prim, "aloha:visualOnly", True)
        _set_bool_attr(prim, "aloha:readyForScenePlacement", False)


def create_stage(output_path: Path, config_path: Path) -> None:
    create_a7_stage(output_path, Path("aloha_isaac_rebuild/configs/physical_reconstruction/camera_default_pose_preview.yaml"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stage = Usd.Stage.Open(str(output_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not reopen generated stage: {output_path}")

    root = UsdGeom.Xform.Define(stage, config["physical_layout_root"]).GetPrim()
    _set_string_attr(root, "aloha:stageRole", "A9_non_camera_physical_layout_checkpoint")
    _set_bool_attr(root, "aloha:cameraWorkDeferred", True)
    _set_bool_attr(root, "aloha:visualOnly", True)
    _set_string_attr(root, "aloha:measurementStatus", "MIXED_MEASURED_AND_MISSING")
    _set_string_attr(root, "aloha:layoutParameterSource", "MEASURED:user_measured_support_frame_and_base_y_edges")
    _set_bool_attr(root, "aloha:physicsEligible", False)
    _set_bool_attr(root, "aloha:collisionEligible", False)
    _set_bool_attr(root, "aloha:renderEligible", False)
    _set_bool_attr(root, "aloha:trainingEligible", False)
    UsdGeom.Scope.Define(stage, config["visuals_root"])

    _add_table_outline(stage, config)
    _add_world_frame(stage, config)
    _add_base_y_reference(stage, config)
    _add_future_measurement_scopes(stage, config)

    aloha_root = stage.GetPrimAtPath(ROOT)
    aloha_root.SetCustomDataByKey("aloha1_rebuild_stage", "A9_physical_layout_preview")
    aloha_root.SetCustomDataByKey("a9_scope", "non_camera_physical_layout_visual_checkpoint")
    aloha_root.SetCustomDataByKey("a9_layout_measurement_status", "MIXED_MEASURED_AND_MISSING")
    aloha_root.SetCustomDataByKey("a9_layout_parameter_source", "MEASURED:user_measured_support_frame_and_base_y_edges")
    aloha_root.SetCustomDataByKey("trossen_table_reference_visibility", config["table_reference"].get("visibility", "inherited"))
    aloha_root.SetCustomDataByKey("camera_work_deferred", True)
    aloha_root.SetCustomDataByKey("camera_calibration_ready", False)
    aloha_root.SetCustomDataByKey("camera_render_ready", False)
    aloha_root.SetCustomDataByKey("support_frame_physics_ready", False)
    aloha_root.SetCustomDataByKey("support_frame_collision_ready", False)
    aloha_root.SetCustomDataByKey("physical_layout_physics_ready", False)
    aloha_root.SetCustomDataByKey("physical_layout_collision_ready", False)
    aloha_root.SetCustomDataByKey("openpi_observation_ready", False)
    aloha_root.SetCustomDataByKey("training_eligible", False)
    aloha_root.SetCustomDataByKey("rl_ready", False)
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
