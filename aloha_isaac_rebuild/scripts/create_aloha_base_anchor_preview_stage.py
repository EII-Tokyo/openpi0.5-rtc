#!/usr/bin/env python3
"""Create A10 visual-only robot/base-anchor measurement checkpoint stage."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from pxr import Gf, Sdf, Usd, UsdGeom

from create_aloha_physical_layout_preview_stage import create_stage as create_a9_stage
from create_aloha_support_frame_components_stage import ROOT


DEFAULT_CONFIG = Path("aloha_isaac_rebuild/configs/physical_reconstruction/base_anchor_preview.yaml")
DEFAULT_OUTPUT = Path("aloha_isaac_rebuild/scenes/aloha_base_anchor_preview.usda")

COLORS = {
    "measured_band": (1.0, 0.64, 0.05),
    "edge": (1.0, 0.35, 0.02),
    "left_zone": (0.15, 0.42, 0.95),
    "right_zone": (0.68, 0.32, 0.95),
    "center": (0.98, 0.98, 0.98),
    "x_axis": (0.90, 0.10, 0.10),
    "y_axis": (0.05, 0.70, 0.18),
}


def _set_string_attr(prim: Usd.Prim, name: str, value: str) -> None:
    prim.CreateAttribute(name, Sdf.ValueTypeNames.String).Set(value)


def _set_bool_attr(prim: Usd.Prim, name: str, value: bool) -> None:
    prim.CreateAttribute(name, Sdf.ValueTypeNames.Bool).Set(value)


def _set_double_attr(prim: Usd.Prim, name: str, value: float) -> None:
    prim.CreateAttribute(name, Sdf.ValueTypeNames.Double).Set(value)


def _add_visual_cube(
    stage: Usd.Stage,
    path: str,
    center: tuple[float, float, float] | list[float],
    size: tuple[float, float, float] | list[float],
    color: tuple[float, float, float],
    stage_role: str,
    source_type: str,
    source_kind: str,
    *,
    opacity: float | None = None,
) -> Usd.Prim:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if opacity is not None:
        cube.CreateDisplayOpacityAttr([opacity])
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*center))
    xform.AddScaleOp().Set(Gf.Vec3d(*size))
    prim = cube.GetPrim()
    _set_string_attr(prim, "aloha:stageRole", stage_role)
    _set_string_attr(prim, "aloha:sourceType", source_type)
    _set_string_attr(prim, "aloha:sourceKind", source_kind)
    _set_bool_attr(prim, "aloha:visualOnly", True)
    _set_bool_attr(prim, "aloha:physicsEligible", False)
    _set_bool_attr(prim, "aloha:collisionEligible", False)
    _set_bool_attr(prim, "aloha:cameraExtrinsicsEligible", False)
    _set_bool_attr(prim, "aloha:renderEligible", False)
    _set_bool_attr(prim, "aloha:trainingEligible", False)
    prim.SetCustomDataByKey("warning", "A10 visual-only base-anchor guide; not full base CAD, collision, physics, or robot placement.")
    return prim


def _add_measured_y_band(stage: Usd.Stage, config: dict) -> None:
    spec = config["measured_base_y_band"]
    root = config["visuals_root"] + "/measured_base_y_band"
    UsdGeom.Scope.Define(stage, root)

    center_y = float(spec["center_y_m"])
    depth_y = float(spec["depth_y_m"])
    support_x = float(spec["support_frame_outer_length_m"])
    z = float(spec["z_m"])
    band = _add_visual_cube(
        stage,
        f"{root}/shared_y_band_not_full_footprint",
        (0.0, center_y, z),
        (support_x, depth_y, 0.006),
        COLORS["measured_band"],
        "A10_measured_base_y_band",
        spec["source_type"],
        spec["source_kind"],
        opacity=0.28,
    )
    _set_bool_attr(band, "aloha:baseYEdgesMeasured", True)
    _set_bool_attr(band, "aloha:fullBaseFootprint", False)
    _set_bool_attr(band, "aloha:xExtentMeasured", False)
    _set_double_attr(band, "aloha:baseDepthYMeters", depth_y)

    edge_z = float(spec["edge_guide_z_m"])
    for name, y_value, measured_offset in [
        ("near_cam_low_measured_y_edge", float(spec["near_cam_low_y_m"]), "180mm_from_cam_low_side_outer_edge"),
        ("near_cam_high_measured_y_edge", float(spec["near_cam_high_y_m"]), "235mm_from_cam_high_side_outer_edge"),
    ]:
        edge = _add_visual_cube(
            stage,
            f"{root}/{name}",
            (0.0, y_value, edge_z),
            (support_x, 0.010, 0.010),
            COLORS["edge"],
            "A10_measured_base_y_edge",
            spec["source_type"],
            spec["source_kind"],
        )
        _set_string_attr(edge, "aloha:measuredOffset", measured_offset)
        _set_bool_attr(edge, "aloha:fullBaseFootprint", False)
        _set_bool_attr(edge, "aloha:xExtentMeasured", False)


def _add_measurement_zones(stage: Usd.Stage, config: dict) -> None:
    spec = config["measurement_zones"]
    root = config["visuals_root"] + "/x_extent_measurement_zones"
    UsdGeom.Scope.Define(stage, root)
    for name, center, color in [
        ("left_base_x_extent_missing_zone", spec["left_zone_center_m"], COLORS["left_zone"]),
        ("right_base_x_extent_missing_zone", spec["right_zone_center_m"], COLORS["right_zone"]),
    ]:
        zone = _add_visual_cube(
            stage,
            f"{root}/{name}",
            center,
            spec["zone_size_m"],
            color,
            "A10_pending_base_x_extent_measurement_zone",
            spec["source_type"],
            spec["source_kind"],
            opacity=0.22,
        )
        _set_string_attr(zone, "aloha:measurementStatus", "PENDING_MEASUREMENT")
        _set_bool_attr(zone, "aloha:baseGeometryComplete", False)
        _set_bool_attr(zone, "aloha:isActualBaseFootprint", False)


def _add_direction_hints(stage: Usd.Stage, config: dict) -> None:
    spec = config["direction_hints"]
    root = config["visuals_root"] + "/direction_hints"
    UsdGeom.Scope.Define(stage, root)
    center = tuple(float(v) for v in spec["center_m"])
    center_marker = _add_visual_cube(
        stage,
        f"{root}/base_y_band_center_marker",
        center,
        (0.030, 0.030, 0.030),
        COLORS["center"],
        "A10_base_y_band_center_marker",
        "DERIVED",
        "derived_from_two_user_measured_y_edges",
    )
    _set_string_attr(center_marker, "aloha:measurementStatus", "DERIVED_FROM_MEASURED")
    _set_bool_attr(center_marker, "aloha:fullBaseFootprint", False)

    x_axis = _add_visual_cube(
        stage,
        f"{root}/support_frame_x_direction_hint",
        (center[0], center[1], center[2] + 0.020),
        spec["x_axis_size_m"],
        COLORS["x_axis"],
        "A10_base_anchor_direction_hint",
        "TUNED",
        "visual_axis_reference",
    )
    _set_string_attr(x_axis, "aloha:axisMeaning", spec["x_axis_meaning"])

    y_axis = _add_visual_cube(
        stage,
        f"{root}/measured_base_y_depth_direction_hint",
        (center[0], center[1], center[2] + 0.040),
        spec["y_axis_size_m"],
        COLORS["y_axis"],
        "A10_base_anchor_direction_hint",
        "TUNED",
        "visual_axis_reference",
    )
    _set_string_attr(y_axis, "aloha:axisMeaning", spec["y_axis_meaning"])


def _add_missing_measurement_scopes(stage: Usd.Stage, config: dict) -> None:
    root_path = config["base_anchor_root"] + "/missing_measurements"
    root = UsdGeom.Scope.Define(stage, root_path).GetPrim()
    _set_bool_attr(root, "aloha:visualOnly", True)
    _set_bool_attr(root, "aloha:baseGeometryComplete", False)
    missing = config["measured_base_y_band"].get("missing", [])
    for item in missing:
        prim = UsdGeom.Scope.Define(stage, f"{root_path}/{item}").GetPrim()
        _set_string_attr(prim, "aloha:status", "MISSING")
        _set_bool_attr(prim, "aloha:requiredBeforePhysics", True)
        _set_bool_attr(prim, "aloha:requiredBeforeRobotPlacement", True)


def create_stage(output_path: Path, config_path: Path) -> None:
    create_a9_stage(output_path, Path("aloha_isaac_rebuild/configs/physical_reconstruction/physical_layout.yaml"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stage = Usd.Stage.Open(str(output_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not reopen generated stage: {output_path}")

    root = UsdGeom.Xform.Define(stage, config["base_anchor_root"]).GetPrim()
    _set_string_attr(root, "aloha:stageRole", "A10_robot_base_anchor_measurement_checkpoint")
    _set_string_attr(root, "aloha:measurementStatus", "MEASURED_Y_BAND_ONLY")
    _set_string_attr(root, "aloha:sourceKind", "user_measured_base_y_edges_and_pending_base_x_measurements")
    _set_bool_attr(root, "aloha:visualOnly", True)
    _set_bool_attr(root, "aloha:baseYEdgesMeasured", True)
    _set_bool_attr(root, "aloha:baseGeometryComplete", False)
    _set_bool_attr(root, "aloha:measuredCadReady", False)
    _set_bool_attr(root, "aloha:physicsEligible", False)
    _set_bool_attr(root, "aloha:collisionEligible", False)
    _set_bool_attr(root, "aloha:cameraExtrinsicsEligible", False)
    _set_bool_attr(root, "aloha:renderEligible", False)
    _set_bool_attr(root, "aloha:trainingEligible", False)
    UsdGeom.Scope.Define(stage, config["visuals_root"])

    _add_measured_y_band(stage, config)
    _add_measurement_zones(stage, config)
    _add_direction_hints(stage, config)
    _add_missing_measurement_scopes(stage, config)

    aloha_root = stage.GetPrimAtPath(ROOT)
    aloha_root.SetCustomDataByKey("aloha1_rebuild_stage", "A10_robot_base_anchor_preview")
    aloha_root.SetCustomDataByKey("a10_scope", "robot_base_anchor_measurement_checkpoint")
    aloha_root.SetCustomDataByKey("base_y_edges_measured", True)
    aloha_root.SetCustomDataByKey("base_geometry_complete", False)
    aloha_root.SetCustomDataByKey("base_x_extent_ready", False)
    aloha_root.SetCustomDataByKey("robot_base_placement_ready", False)
    aloha_root.SetCustomDataByKey("support_frame_collision_ready", False)
    aloha_root.SetCustomDataByKey("physical_layout_collision_ready", False)
    aloha_root.SetCustomDataByKey("camera_work_deferred", True)
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
