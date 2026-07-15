from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "examples/aloha_isaac/config/workcell_minimal.yaml"


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_site_model(cfg: dict[str, Any]) -> dict[str, Any] | None:
    site_model_json = cfg.get("stage", {}).get("site_model_json")
    if not site_model_json:
        return None
    path = (REPO_ROOT / site_model_json).resolve()
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _require_isaac() -> None:
    try:
        import isaacsim  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "Isaac Sim is not importable in this Python environment. "
            "Run examples/aloha_isaac/scripts/check_isaac_install.py and install Isaac first."
        ) from exc


def _rpy_deg_to_quat(rpy_deg: list[float]):
    from pxr import Gf

    roll, pitch, yaw = [math.radians(v) for v in rpy_deg]
    qx = Gf.Quatf(math.cos(roll / 2), math.sin(roll / 2), 0, 0)
    qy = Gf.Quatf(math.cos(pitch / 2), 0, math.sin(pitch / 2), 0)
    qz = Gf.Quatf(math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
    return qz * qy * qx


def _set_xform(prim, translation: list[float], rotation_rpy_deg: list[float] | None = None) -> None:
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    if rotation_rpy_deg is not None:
        xform.AddOrientOp().Set(_rpy_deg_to_quat(rotation_rpy_deg))


def _apply_collision(prim) -> None:
    from pxr import UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)


def _add_cube(
    stage,
    path: str,
    translation: list[float],
    size: list[float],
    color: list[float],
    *,
    collision: bool = False,
) -> None:
    from pxr import Gf, Sdf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, Sdf.Path(path))
    cube.CreateSizeAttr(1.0)
    _set_xform(cube.GetPrim(), translation)
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collision:
        _apply_collision(cube.GetPrim())


def _add_sphere(stage, path: str, translation: list[float], radius: float, color: list[float]) -> None:
    from pxr import Gf, Sdf, UsdGeom

    sphere = UsdGeom.Sphere.Define(stage, Sdf.Path(path))
    sphere.CreateRadiusAttr(radius)
    _set_xform(sphere.GetPrim(), translation)
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def _add_cylinder_between(
    stage,
    path: str,
    start: list[float],
    end: list[float],
    radius: float,
    color: list[float],
    *,
    collision: bool = False,
) -> None:
    from pxr import Gf, Sdf, UsdGeom

    start_v = Gf.Vec3d(*start)
    end_v = Gf.Vec3d(*end)
    midpoint = (start_v + end_v) * 0.5
    direction = end_v - start_v
    height = direction.GetLength()
    if height <= 1e-8:
        raise ValueError(f"zero-length cylinder: {path}")

    cylinder = UsdGeom.Cylinder.Define(stage, Sdf.Path(path))
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(height)
    cylinder.CreateAxisAttr("Z")
    cylinder.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    z_axis = Gf.Vec3d(0, 0, 1)
    direction_norm = direction.GetNormalized()
    rotation = Gf.Rotation(z_axis, direction_norm)
    xform = UsdGeom.Xformable(cylinder.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(midpoint)
    xform.AddOrientOp().Set(Gf.Quatf(rotation.GetQuat()))
    if collision:
        _apply_collision(cylinder.GetPrim())


def _hex_to_rgb(value: str) -> list[float]:
    value = value.strip().lstrip("#")
    return [int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]


def _normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 1e-8:
        raise ValueError("zero-length vector")
    return [v / norm for v in values]


def _set_custom_data_json(prim, key: str, value: Any) -> None:
    """Store provenance values using USD-safe scalar metadata."""
    if isinstance(value, (dict, list, tuple)):
        prim.SetCustomDataByKey(key, json.dumps(value, ensure_ascii=True))
        return
    prim.SetCustomDataByKey(key, value)


def _round_vec(values: list[float], digits: int = 3) -> list[float]:
    return [round(float(v), digits) for v in values]


def _resolve_pipe_placeholder(cfg: dict[str, Any]) -> dict[str, Any]:
    pipe_cfg = dict(cfg["pipe_placeholder"])
    measurement = pipe_cfg.get("measurement")
    if not measurement:
        return pipe_cfg

    table_edge = measurement.get("table_edge")
    if table_edge not in {"w0", "w1"}:
        raise ValueError(f"unsupported pipe table edge: {table_edge!r}")

    table_size = cfg["table"]["size"]
    table_translation = cfg["table"]["pose"]["translation"]
    left_edge_x = float(table_translation[0]) - float(table_size[0]) / 2.0
    edge_y = float(table_translation[1]) + (
        float(table_size[1]) / 2.0 if table_edge == "w1" else -float(table_size[1]) / 2.0
    )

    a_point = [
        left_edge_x + float(measurement["a_distance_from_left_edge_m"]),
        edge_y,
        0.0,
    ]
    outside_sign = 1.0 if table_edge == "w1" else -1.0
    base_y = a_point[1] + outside_sign * float(measurement["base_offset_outside_table_m"])
    start = [a_point[0], base_y, float(measurement["mount_height_m"])]

    length = float(measurement["pipe_length_m"])
    tilt_rad = math.radians(float(measurement["side_tilt_deg"]))
    horizontal = length * math.cos(tilt_rad)
    vertical = length * math.sin(tilt_rad)
    plan_direction = measurement.get("plan_direction", "toward_table")
    if plan_direction == "parallel_to_table_edge_toward_left_arm":
        end = [start[0] - horizontal, start[1], start[2] + vertical]
    elif plan_direction == "toward_table":
        toward_table_sign = -1.0 if table_edge == "w1" else 1.0
        end = [start[0], start[1] + toward_table_sign * horizontal, start[2] + vertical]
    else:
        raise ValueError(f"unsupported pipe plan direction: {plan_direction!r}")

    pipe_cfg["start"] = _round_vec(start)
    pipe_cfg["end"] = _round_vec(end)
    pipe_cfg["radius"] = round(float(measurement["pipe_diameter_m"]) / 2.0, 6)
    pipe_cfg["measurement_a_point"] = _round_vec(a_point)
    pipe_cfg["measurement_base_offset_line_start"] = _round_vec(a_point)
    pipe_cfg["measurement_base_offset_line_end"] = _round_vec([start[0], start[1], 0.0])
    return pipe_cfg


def _add_workspace_areas(stage, cfg: dict[str, Any], site_model: dict[str, Any] | None) -> None:
    area_cfg = cfg.get("workspace_areas", {})
    if not area_cfg.get("enabled", False) or site_model is None:
        return
    base = area_cfg["prim_path"]
    height = float(area_cfg.get("height", 0.006))
    for area in site_model.get("workspace_areas", []):
        cx, cy, cz = area["center_m"]
        sx, sy = area["size_m"]
        _add_cube(stage, f"{base}/{area['id']}", [cx, cy, cz], [sx, sy, height], _hex_to_rgb(area["color"]))


def _add_robot_mount_hints(stage, cfg: dict[str, Any], site_model: dict[str, Any] | None) -> None:
    if site_model is not None:
        mounts = site_model.get("robot_mounts", {})
        radius = float(mounts.get("turntable_radius_m", 0.055))
        for name, path, color in (
            ("left_shoulder_turntable_center_m", "/World/Aloha/left_shoulder_turntable_measured", [0.38, 0.78, 1.0]),
            ("right_shoulder_turntable_center_m", "/World/Aloha/right_shoulder_turntable_measured", [1.0, 0.65, 0.35]),
        ):
            center = mounts.get(name)
            if center is None:
                continue
            _add_cylinder_between(
                stage,
                path,
                [center[0], center[1], center[2] - 0.01],
                [center[0], center[1], center[2] + 0.01],
                radius,
                color,
            )
        return

    for base in cfg["robot_layout"].values():
        _add_cube(stage, base["prim_path"], base["translation"], [0.12, 0.12, 0.06], [0.55, 0.55, 0.55])


def _add_legacy_rinse_device(stage, cfg: dict[str, Any], site_model: dict[str, Any] | None) -> None:
    legacy_cfg = cfg.get("legacy_rinse_device", {})
    if not legacy_cfg.get("enabled", False) or site_model is None:
        return
    old = site_model["rinse_device_v1"]
    color = legacy_cfg.get("color", [0.1, 0.9, 0.35])
    base = legacy_cfg["prim_path"]
    center = old["funnel_center_m"]
    _add_cylinder_between(
        stage,
        f"{base}/funnel_opening_hint",
        [center[0], center[1], center[2] - 0.012],
        [center[0], center[1], center[2] + 0.012],
        float(old["funnel_opening_diameter_m"]) / 2.0,
        color,
    )
    _add_cylinder_between(
        stage,
        f"{base}/old_nozzle_axis",
        center,
        old["nozzle_tip_m"],
        float(old["legacy_nozzle_capsule_radius_m"]),
        color,
    )


def _add_pipe_support_and_axis(stage, cfg: dict[str, Any], site_model: dict[str, Any] | None) -> None:
    pipe_cfg = _resolve_pipe_placeholder(cfg)
    _add_cylinder_between(
        stage,
        f"{pipe_cfg['prim_path']}/axis",
        pipe_cfg["start"],
        pipe_cfg["end"],
        float(pipe_cfg["radius"]),
        pipe_cfg["color"],
        collision=True,
    )
    _add_sphere(stage, f"{pipe_cfg['prim_path']}/inlet_center", pipe_cfg["end"], 0.012, pipe_cfg["color"])

    start = pipe_cfg["start"]
    support_center = [start[0], start[1], max(0.02, start[2] - 0.045)]
    _add_cube(
        stage,
        f"{pipe_cfg['prim_path']}/support_base_placeholder",
        support_center,
        [0.12, 0.025, 0.018],
        [0.85, 0.85, 0.82],
        collision=True,
    )

    if "measurement_a_point" in pipe_cfg:
        _add_sphere(
            stage,
            f"{pipe_cfg['prim_path']}/measurement_A_on_w1_edge",
            pipe_cfg["measurement_a_point"],
            0.01,
            [0.1, 0.45, 1.0],
        )
        _add_cylinder_between(
            stage,
            f"{pipe_cfg['prim_path']}/measurement_9p5cm_base_offset",
            pipe_cfg["measurement_base_offset_line_start"],
            pipe_cfg["measurement_base_offset_line_end"],
            0.004,
            [0.1, 0.45, 1.0],
        )

    if site_model is not None:
        axis_prim = stage.GetPrimAtPath(f"{pipe_cfg['prim_path']}/axis")
        for key, value in site_model.get("current_pipe_candidate", {}).items():
            _set_custom_data_json(axis_prim, f"current_pipe_candidate_{key}", value)


def _add_bottle_reference(stage, cfg: dict[str, Any], site_model: dict[str, Any] | None) -> None:
    bottle_cfg = cfg.get("bottle_reference", {})
    if not bottle_cfg.get("enabled", False) or site_model is None:
        return
    bottle = site_model["bottle_default"]
    axis = _normalize(bottle_cfg["axis_unit"])
    center = bottle_cfg["center"]
    total_length = float(bottle["total_length_m"])
    neck_length = float(bottle["neck_length_m"])
    half = [axis[i] * total_length * 0.5 for i in range(3)]
    body_start = [center[i] - half[i] for i in range(3)]
    body_end = [center[i] + half[i] for i in range(3)]
    neck_end = [body_end[i] + axis[i] * neck_length for i in range(3)]
    base = bottle_cfg["prim_path"]
    _add_cylinder_between(stage, f"{base}/body_axis_hint", body_start, body_end, float(bottle["body_diameter_m"]) / 2.0, bottle_cfg["color"])
    _add_cylinder_between(stage, f"{base}/neck_hint", body_end, neck_end, float(bottle["neck_diameter_m"]) / 2.0, [0.2, 0.55, 1.0])
    _add_sphere(stage, f"{base}/mouth_center_hint", neck_end, float(bottle["neck_diameter_m"]) * 0.65, [1.0, 0.95, 0.2])


def _add_table_frame(stage, cfg: dict[str, Any]) -> None:
    origin = cfg["pose"]["translation"]
    length = float(cfg["axis_length"])
    radius = float(cfg["axis_radius"])
    base = cfg["prim_path"]
    _add_cylinder_between(stage, f"{base}/x_axis", origin, [origin[0] + length, origin[1], origin[2]], radius, [1, 0, 0])
    _add_cylinder_between(stage, f"{base}/y_axis", origin, [origin[0], origin[1] + length, origin[2]], radius, [0, 0.8, 0])
    _add_cylinder_between(stage, f"{base}/z_axis", origin, [origin[0], origin[1], origin[2] + length], radius, [0.1, 0.2, 1])
    _add_cube(stage, f"{base}/origin", origin, [0.03, 0.03, 0.03], [0.1, 0.2, 1])


def _add_camera(stage, name: str, cfg: dict[str, Any]) -> None:
    from pxr import Gf, Sdf, UsdGeom

    camera = UsdGeom.Camera.Define(stage, Sdf.Path(cfg["prim_path"]))
    camera.CreateFocalLengthAttr(float(cfg.get("focal_length", 18.0)))
    translation = Gf.Vec3d(*cfg["translation"])
    target = Gf.Vec3d(*cfg["target"])
    direction = (target - translation).GetNormalized()
    # USD cameras look down their local -Z axis.
    rotation = Gf.Rotation(Gf.Vec3d(0, 0, -1), direction)
    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(translation)
    xform.AddOrientOp().Set(Gf.Quatf(rotation.GetQuat()))
    camera.GetPrim().SetCustomDataByKey("aloha_name", name)


def _reference_usd(stage, path: str, usd_path: Path, translation: list[float], rotation_rpy_deg: list[float]) -> None:
    from pxr import Sdf, UsdGeom

    root = UsdGeom.Xform.Define(stage, Sdf.Path(path))
    _set_xform(root.GetPrim(), translation, rotation_rpy_deg)
    root.GetPrim().GetReferences().AddReference(str(usd_path.resolve()))


def _resolve_aloha_reference_targets(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    assets_cfg = cfg["assets"]
    if assets_cfg.get("instance_single_arm_usd_twice", False):
        return [
            {
                "prim_path": arm_cfg["prim_path"],
                "translation": arm_cfg["translation"],
                "rotation_rpy_deg": arm_cfg["rotation_rpy_deg"],
            }
            for arm_cfg in cfg["robot_instances"].values()
        ]

    pose_cfg = assets_cfg.get("aloha_pose", {})
    return [
        {
            "prim_path": assets_cfg.get("aloha_prim_path", "/World/Aloha/StationaryAI"),
            "translation": pose_cfg.get("translation", [0.0, 0.0, 0.0]),
            "rotation_rpy_deg": pose_cfg.get("rotation_rpy_deg", [0.0, 0.0, 0.0]),
        }
    ]


def _reference_aloha_usd(stage, cfg: dict[str, Any], usd_path: Path) -> None:
    """Reference the imported ALOHA asset.

    Preferred path: reference the official Trossen Stationary AI dual-arm USD as
    one robot asset. The older single-arm-instanced-twice path is retained only
    for legacy local MJCF-import experiments.
    """
    from pxr import Sdf, UsdGeom

    UsdGeom.Xform.Define(stage, Sdf.Path("/World/Aloha"))

    for target in _resolve_aloha_reference_targets(cfg):
        _reference_usd(
            stage,
            target["prim_path"],
            usd_path,
            target["translation"],
            target["rotation_rpy_deg"],
        )


def build_stage(config_path: Path, headless: bool) -> Path:
    _require_isaac()

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": headless})
    try:
        from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

        cfg = _load_config(config_path)
        site_model = _load_site_model(cfg)
        output_usd = (REPO_ROOT / cfg["stage"]["output_usd"]).resolve()
        output_usd.parent.mkdir(parents=True, exist_ok=True)
        if output_usd.exists():
            output_usd.unlink()

        stage = Usd.Stage.CreateNew(str(output_usd))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
        UsdPhysics.Scene.Define(stage, Sdf.Path("/World/PhysicsScene"))
        light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight"))
        light.CreateIntensityAttr(800)

        table_cfg = cfg["table"]
        _add_cube(
            stage,
            table_cfg["prim_path"],
            table_cfg["pose"]["translation"],
            table_cfg["size"],
            table_cfg["color"],
            collision=True,
        )
        _add_table_frame(stage, cfg["table_frame"])
        _add_workspace_areas(stage, cfg, site_model)
        _add_legacy_rinse_device(stage, cfg, site_model)
        _add_pipe_support_and_axis(stage, cfg, site_model)
        _add_bottle_reference(stage, cfg, site_model)

        for name, camera_cfg in cfg["cameras"].items():
            _add_camera(stage, name, camera_cfg)

        aloha_usd = (REPO_ROOT / cfg["assets"]["aloha_usd"]).resolve()
        if cfg["assets"].get("load_aloha_if_exists", True) and aloha_usd.exists():
            _reference_aloha_usd(stage, cfg, aloha_usd)
        else:
            _add_robot_mount_hints(stage, cfg, None)
        if site_model is not None:
            _add_robot_mount_hints(stage, cfg, site_model)

        stage.GetRootLayer().Save()
        return output_usd
    finally:
        app.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the minimal Isaac Sim ALOHA workcell stage.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    output = build_stage(args.config, args.headless)
    print(f"usd={output}")


if __name__ == "__main__":
    # Avoid a runtime prompt in scripted runs; users still need to accept NVIDIA's EULA.
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    main()
