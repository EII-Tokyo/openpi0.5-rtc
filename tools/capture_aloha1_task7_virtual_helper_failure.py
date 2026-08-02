#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Capture collision-overlay evidence for the rejected helper-body candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback
from typing import Any

import numpy as np

from tools.aloha1_mapping.isaac_screenshot import look_at_orientation_wxyz
from tools.capture_aloha_viper_cad_finger_task5_numeric_pass_viewport import _capture_viewport_png

ROOT = Path(__file__).resolve().parents[1]
RESOLUTION = (1600, 1000)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--follower", choices=("follower_left", "follower_right"), required=True)
    parser.add_argument("--validator-repeat1", type=Path, required=True)
    parser.add_argument("--validator-repeat2", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def _mesh_world_points(stage: Any, prim: Any, cache: Any) -> np.ndarray:
    from pxr import UsdGeom

    points = UsdGeom.Mesh(prim).GetPointsAttr().Get() or []
    world = cache.GetLocalToWorldTransform(prim)
    return np.asarray(
        [list(world.Transform(point)) for point in points],
        dtype=np.float64,
    )


def _overlay_color(path: str) -> tuple[float, float, float]:
    if "base_link" in path:
        return (1.0, 0.05, 0.05)
    if "gripper" in path or "finger" in path:
        return (1.0, 0.55, 0.02)
    return (0.0, 0.95, 1.0)


def _build_overlay(stage: Any, robot_root: str) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    from pxr import Gf
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics
    from pxr import Vt

    cache = UsdGeom.XformCache()
    overlay_root = "/Task7VirtualHelperFailure/ColliderOverlay"
    UsdGeom.Scope.Define(stage, overlay_root)
    records = []
    grouped: dict[str, list[np.ndarray]] = {"base": [], "gripper": [], "all": []}
    sources = [
        prim
        for prim in stage.Traverse(
            Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
        )
        if prim.IsA(UsdGeom.Mesh)
        and prim.HasAPI(UsdPhysics.CollisionAPI)
        and str(prim.GetPath()).startswith(robot_root + "/")
    ]
    for index, source in enumerate(sorted(sources, key=lambda item: str(item.GetPath()))):
        source_path = str(source.GetPath())
        source_mesh = UsdGeom.Mesh(source)
        counts = source_mesh.GetFaceVertexCountsAttr().Get() or []
        indices = source_mesh.GetFaceVertexIndicesAttr().Get() or []
        world_points = _mesh_world_points(stage, source, cache)
        if not len(world_points) or not counts or not indices:
            continue
        clone = UsdGeom.Mesh.Define(stage, f"{overlay_root}/collider_{index:03d}")
        clone.CreatePointsAttr(
            Vt.Vec3fArray([Gf.Vec3f(*point) for point in world_points])
        )
        clone.CreateFaceVertexCountsAttr(Vt.IntArray([int(value) for value in counts]))
        clone.CreateFaceVertexIndicesAttr(Vt.IntArray([int(value) for value in indices]))
        clone.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        clone.CreateDoubleSidedAttr(True)  # noqa: FBT003
        clone.CreateDisplayColorAttr([Gf.Vec3f(*_overlay_color(source_path))])
        clone.CreateDisplayOpacityAttr([0.62])
        group = "base" if "base_link" in source_path else (
            "gripper" if "gripper" in source_path or "finger" in source_path else "other"
        )
        grouped["all"].append(world_points)
        if group in grouped:
            grouped[group].append(world_points)
        records.append(
            {
                "source_prim": source_path,
                "clone_prim": str(clone.GetPath()),
                "group": group,
                "point_count": len(world_points),
            }
        )
    arrays = {
        key: np.concatenate(values) if values else np.empty((0, 3), dtype=np.float64)
        for key, values in grouped.items()
    }
    return records, arrays


def _visual_points(stage: Any, robot_root: str) -> np.ndarray:
    from pxr import Usd
    from pxr import UsdGeom

    cache = UsdGeom.XformCache()
    clouds = []
    for prim in stage.Traverse(
        Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    ):
        path = str(prim.GetPath())
        if not prim.IsA(UsdGeom.Mesh) or not path.startswith(robot_root + "/"):
            continue
        if "/collisions/" in path or "/cad_derived_collisions/" in path:
            continue
        points = _mesh_world_points(stage, prim, cache)
        if len(points):
            clouds.append(points)
    if not clouds:
        raise RuntimeError("robot visual point cloud is empty")
    return np.concatenate(clouds)


def _projection(camera: Any, points: np.ndarray) -> dict[str, Any]:
    if not len(points):
        return {"minimum_px": None, "maximum_px": None, "finite_point_count": 0}
    pixels = np.asarray(camera.get_image_coords_from_world_points(points), dtype=np.float64)
    finite = pixels[np.isfinite(pixels).all(axis=1)]
    if not len(finite):
        return {"minimum_px": None, "maximum_px": None, "finite_point_count": 0}
    return {
        "minimum_px": finite.min(axis=0).tolist(),
        "maximum_px": finite.max(axis=0).tolist(),
        "finite_point_count": len(finite),
    }


def _main() -> int:
    args = _parse_args()
    stage_path = args.stage.resolve(strict=True)
    repeat_paths = [args.validator_repeat1.resolve(strict=True), args.validator_repeat2.resolve(strict=True)]
    repeats = [json.loads(path.read_text(encoding="utf-8")) for path in repeat_paths]
    signatures = []
    for report in repeats:
        issues = [
            (item["rule"], item["at"], item["message"])
            for item in report["issues"]
            if item["severity"] in {"ERROR", "FAILURE"}
        ]
        signatures.append(hashlib.sha256(json.dumps(issues, sort_keys=True).encode()).hexdigest())
    if signatures[0] != signatures[1]:
        raise RuntimeError("fresh-process failure signatures differ")
    clash_count = sum(
        item["rule"] == "NonAdjacentCollisionMeshesDoNotClash"
        for item in repeats[0]["issues"]
    )
    if clash_count != 57:
        raise RuntimeError(f"expected 57 repeated clash findings, got {clash_count}")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    raw_root = output_root / "screenshots_raw"
    raw_root.mkdir(parents=True)

    from isaacsim.core.utils.stage import get_current_stage
    from isaacsim.core.utils.stage import open_stage
    from isaacsim.sensors.camera import Camera
    from omni.kit.viewport.utility import get_active_viewport
    from pxr import Gf
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdLux

    app = globals()["_SIMULATION_APP"]
    if not open_stage(str(stage_path)):
        raise RuntimeError(f"cannot open {stage_path}")
    stage = get_current_stage()
    root_prim = stage.GetDefaultPrim()
    robot_root = str(root_prim.GetPath())
    if not robot_root.startswith("/vx300s_"):
        raise RuntimeError(f"unexpected robot root: {robot_root}")
    stage.SetEditTarget(stage.GetSessionLayer())
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        dome = UsdLux.DomeLight.Define(stage, "/Task7VirtualHelperFailure/Dome")
        dome.CreateIntensityAttr(900.0)
        key = UsdLux.DistantLight.Define(stage, "/Task7VirtualHelperFailure/Key")
        key.CreateIntensityAttr(1800.0)
        key.AddRotateXYZOp().Set(Gf.Vec3f(30.0, -35.0, -20.0))
        overlay_records, collider_groups = _build_overlay(stage, robot_root)

    cache = UsdGeom.XformCache()
    helper_points = []
    for suffix in ("ee_arm_link", "fingers_link", "ee_gripper_link"):
        prim = stage.GetPrimAtPath(f"{robot_root}/{args.follower}_{suffix}")
        if not prim.IsValid():
            raise RuntimeError(f"missing helper prim: {suffix}")
        helper_points.append(
            list(cache.GetLocalToWorldTransform(prim).ExtractTranslation())
        )
    collider_groups["helpers"] = np.asarray(helper_points, dtype=np.float64)

    visual_points = _visual_points(stage, robot_root)
    framing_points = np.concatenate([visual_points, collider_groups["all"]])
    minimum = framing_points.min(axis=0)
    maximum = framing_points.max(axis=0)
    center = (minimum + maximum) / 2.0
    span = float(np.linalg.norm(maximum - minimum))
    gripper_points = collider_groups["gripper"]
    if not len(gripper_points):
        raise RuntimeError("gripper collider group is empty")
    gripper_center = (gripper_points.min(axis=0) + gripper_points.max(axis=0)) / 2.0
    gripper_span = float(np.linalg.norm(gripper_points.max(axis=0) - gripper_points.min(axis=0)))
    direction = np.asarray([0.75, 1.0, 0.62], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    views = {
        "whole_arm_oblique": {
            "target": center,
            "position": center + direction * max(3.0 * span, 1.8),
        },
        "gripper_failure_closeup": {
            "target": gripper_center,
            "position": gripper_center + direction * max(3.0 * gripper_span, 0.55),
        },
    }
    camera = Camera(
        prim_path="/Task7VirtualHelperFailure/Camera",
        name="task7_virtual_helper_failure_camera",
        resolution=RESOLUTION,
        frequency=60,
    )
    camera.initialize()
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("active viewport unavailable")
    captures = []
    for view_name, view in views.items():
        orientation = look_at_orientation_wxyz(
            view["position"],
            view["target"],
            np.asarray([0.0, 0.0, 1.0]),
        )
        camera.set_clipping_range(0.005, 10.0)
        camera.set_world_pose(
            position=view["position"],
            orientation=orientation,
            camera_axes="usd",
        )
        viewport.camera_path = Sdf.Path(camera.prim_path)
        for _ in range(60):
            app.update()
        raw = raw_root / f"{args.follower}_{view_name}_raw.png"
        _capture_viewport_png(app, viewport, raw)
        captures.append(
            {
                "follower": args.follower,
                "view": view_name,
                "raw_absolute_path": str(raw.resolve(strict=True)),
                "raw_sha256": _sha256(raw),
                "resolution": list(RESOLUTION),
                "camera": {
                    "position_world_m": view["position"].tolist(),
                    "target_world_m": view["target"].tolist(),
                    "orientation_wxyz": np.asarray(orientation).tolist(),
                },
                "projections": {
                    key: _projection(camera, points)
                    for key, points in collider_groups.items()
                },
                "visual_review": "PENDING",
            }
        )
    report = {
        "schema_version": 1,
        "status": "PARTIAL",
        "reason": "PENDING_ANNOTATION_AND_VISUAL_MODEL_REVIEW",
        "stage": {
            "absolute_path": str(stage_path),
            "sha256_before": _sha256(stage_path),
            "sha256_after": _sha256(stage_path),
            "root_prim": robot_root,
        },
        "failure": {
            "hypothesis": "virtual_helpers_without_rigid_body",
            "fresh_process_count": 2,
            "deterministic_signature": signatures[0],
            "non_adjacent_clash_count": clash_count,
            "interpretation": (
                "Removing body semantics from source-empty fixed helpers changes the "
                "validator's rigid-body adjacency/ownership model and introduces 57 "
                "repeatable collision-clash findings."
            ),
        },
        "validator_reports": [
            {"absolute_path": str(path), "sha256": _sha256(path)}
            for path in repeat_paths
        ],
        "overlay": {
            "semantics": "SESSION_ONLY_NON_PHYSICAL_VISUAL_CLONES_OF_AUTHORED_COLLIDERS",
            "count": len(overlay_records),
            "records": overlay_records,
            "colors": {
                "base": "red",
                "gripper_and_fingers": "orange",
                "other_arm_links": "cyan",
            },
        },
        "captures": captures,
        "source_or_final_asset_modified": False,
        "task8": "NOT_RUN",
    }
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "captures": len(captures)}))
    return 0


def run() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "width": RESOLUTION[0],
            "height": RESOLUTION[1],
            "renderer": "RaytracedLighting",
        }
    )
    globals()["_SIMULATION_APP"] = app
    exit_code = 1
    try:
        exit_code = _main()
    except BaseException:
        traceback.print_exc()
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
