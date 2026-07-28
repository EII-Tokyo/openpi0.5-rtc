#!/usr/bin/env python3
"""Capture Isaac runtime evidence for correct-finger open/closed preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np
import yaml

from tools.aloha1_mapping.correct_finger_asset import (
    EXPECTED_RESTART_BOUNDARY,
)
from tools.aloha1_mapping.correct_finger_asset import load_correct_finger_profile
from tools.aloha1_mapping.isaac_screenshot import look_at_orientation_wxyz
from tools.aloha1_mapping.isaac_screenshot import save_camera_rgba_png
from tools.aloha1_mapping.screenshot_manifest import build_screenshot_manifest
from tools.aloha1_mapping.screenshot_manifest import validate_screenshot


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _world_points(stage: Any, prim_path: str) -> np.ndarray:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"correct finger visual mesh missing: {prim_path}")
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get() or []
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    return np.asarray(
        [
            list(transform.Transform(point))
            for point in points
        ],
        dtype=np.float64,
    )


def _geometry_state(stage: Any, robot: str) -> dict[str, Any]:
    from pxr import Gf
    from pxr import UsdGeom

    paths = {
        side: (
            f"/World/Robot/{robot}_{side}_finger_link/visuals/"
            f"correct_custom_finger_{side}/mesh"
        )
        for side in ("left", "right")
    }
    clouds = {
        side: _world_points(stage, path) for side, path in paths.items()
    }
    centers = {side: cloud.mean(axis=0) for side, cloud in clouds.items()}
    closing_axis = centers["right"] - centers["left"]
    closing_axis /= np.linalg.norm(closing_axis)
    center_projection = {
        side: float(np.dot(center, closing_axis))
        for side, center in centers.items()
    }
    low_side, high_side = sorted(
        ("left", "right"),
        key=lambda side: center_projection[side],
    )
    low_projection = clouds[low_side] @ closing_axis
    high_projection = clouds[high_side] @ closing_axis
    surface_gap = float(high_projection.min() - low_projection.max())
    combined = np.concatenate([clouds["left"], clouds["right"]], axis=0)
    target = (combined.min(axis=0) + combined.max(axis=0)) / 2.0

    link = stage.GetPrimAtPath(f"/World/Robot/{robot}_left_finger_link")
    link_transform = UsdGeom.XformCache().GetLocalToWorldTransform(link)
    tool_axis = np.asarray(
        list(link_transform.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))),
        dtype=np.float64,
    )
    tool_axis /= np.linalg.norm(tool_axis)
    return {
        "mesh_paths": paths,
        "centers_world_m": {
            side: centers[side].tolist() for side in ("left", "right")
        },
        "closing_axis_world": closing_axis.tolist(),
        "tool_axis_world": tool_axis.tolist(),
        "surface_gap_m": surface_gap,
        "target_world_m": target.tolist(),
        "combined_aabb_min_m": combined.min(axis=0).tolist(),
        "combined_aabb_max_m": combined.max(axis=0).tolist(),
    }


def _camera_pose(
    geometry: dict[str, Any],
    view: str,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(geometry["target_world_m"], dtype=np.float64)
    closing = np.asarray(geometry["closing_axis_world"], dtype=np.float64)
    tool = np.asarray(geometry["tool_axis_world"], dtype=np.float64)
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if view == "closing_axis":
        position = target - 0.28 * closing + 0.035 * up
    elif view == "isometric":
        position = target - 0.24 * tool - 0.13 * closing + 0.16 * up
    else:
        raise ValueError(f"unsupported camera view: {view}")
    orientation = look_at_orientation_wxyz(position, target)
    return position, orientation


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture correct-finger Isaac 5.1 open/closed evidence."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("configs/aloha1_gripper_correct_finger_profiles.yaml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "reports/aloha1_mapping/"
            "gripper_correct_finger_preflight_screenshots.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    project_root = args.project_root.resolve(strict=True)
    profile_path = (
        args.profile
        if args.profile.is_absolute()
        else project_root / args.profile
    )
    report_path = (
        args.report if args.report.is_absolute() else project_root / args.report
    )
    profile = load_correct_finger_profile(profile_path, project_root)
    preflight_path = (
        project_root
        / "reports/aloha1_mapping/gripper_correct_finger_preflight.json"
    )
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight["status"] != "PASS":
        raise RuntimeError("correct-finger diagnostic USD preflight is not PASS")
    physics = yaml.safe_load(
        (project_root / "configs/aloha1_physics_profiles.yaml").read_text(
            encoding="utf-8"
        )
    )
    home_by_robot = {
        item["name"]: np.asarray(item["home_si"], dtype=np.float64)
        for item in physics["robots"]
    }
    screenshot_root = (
        project_root / profile["diagnostic_directories"]["screenshots"]
    ).resolve()
    screenshot_root.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "width": 1280,
            "height": 900,
        }
    )
    try:
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.utils.stage import create_new_stage
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.sensors.camera import Camera
        from pxr import Gf
        from pxr import UsdLux

        captures = []
        robots: dict[str, Any] = {}
        for robot in ("follower_left", "follower_right"):
            print(f"CAPTURE_PROGRESS robot_start={robot}", flush=True)
            World.clear_instance()
            create_new_stage()
            stage = get_current_stage()
            world_prim = stage.DefinePrim("/World", "Xform")
            stage.SetDefaultPrim(world_prim)
            asset = (
                project_root
                / profile["diagnostic_directories"]["convex_hull"]
                / robot
                / f"{robot}_convex_hull.usd"
            ).resolve(strict=True)
            add_reference_to_stage(str(asset), "/World/Robot")
            dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
            dome.CreateIntensityAttr(650.0)
            dome.CreateColorAttr(Gf.Vec3f(0.85, 0.88, 1.0))
            key = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
            key.CreateIntensityAttr(1100.0)
            key.CreateAngleAttr(1.0)
            world = World(
                stage_units_in_meters=1.0,
                backend="numpy",
                device="cpu",
                physics_dt=1.0 / 60.0,
                rendering_dt=1.0 / 60.0,
            )
            physics_context = world.get_physics_context()
            physics_context.set_solve_articulation_contact_last(True)
            articulation = SingleArticulation(
                prim_path="/World/Robot/root_joint",
                name=f"{robot}_correct_finger_preflight",
                reset_xform_properties=False,
            )
            world.scene.add(articulation)
            camera = Camera(
                prim_path="/World/DiagnosticCamera",
                name=f"{robot}_diagnostic_camera",
                resolution=(1280, 900),
                frequency=60,
            )
            world.scene.add(camera)
            world.reset()
            camera.initialize()
            camera.set_clipping_range(0.01, 10.0)
            print(f"CAPTURE_PROGRESS initialized={robot}", flush=True)
            order = list(articulation.dof_names)
            home = home_by_robot[robot]
            if len(order) != len(home):
                raise RuntimeError(
                    f"home/DOF length mismatch for {robot}: {len(home)} != {len(order)}"
                )
            left_index = order.index("left_finger")
            right_index = order.index("right_finger")
            robot_states: dict[str, Any] = {}
            for state, targets in (
                ("open", (0.057, -0.057)),
                ("closed", (0.021, -0.021)),
            ):
                print(
                    f"CAPTURE_PROGRESS state_start={robot}/{state}",
                    flush=True,
                )
                qpos = home.copy()
                qpos[left_index] = targets[0]
                qpos[right_index] = targets[1]
                articulation.set_joint_positions(qpos)
                articulation.get_articulation_controller().apply_action(
                    ArticulationAction(
                        joint_positions=qpos.astype(np.float32),
                    )
                )
                for _ in range(12):
                    world.step(render=True)
                readback = np.asarray(
                    articulation.get_joint_positions(),
                    dtype=np.float64,
                )
                geometry = _geometry_state(stage, robot)
                print(
                    "CAPTURE_PROGRESS geometry="
                    + json.dumps(
                        {
                            "robot": robot,
                            "state": state,
                            "target": geometry["target_world_m"],
                            "closing_axis": geometry["closing_axis_world"],
                            "tool_axis": geometry["tool_axis_world"],
                            "aabb_min": geometry["combined_aabb_min_m"],
                            "aabb_max": geometry["combined_aabb_max_m"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                state_gate = bool(
                    abs(readback[left_index] - targets[0]) <= 5.0e-4
                    and abs(readback[right_index] - targets[1]) <= 5.0e-4
                )
                state_record = {
                    **geometry,
                    "left_finger_target_m": targets[0],
                    "right_finger_target_m": targets[1],
                    "left_finger_readback_m": float(readback[left_index]),
                    "right_finger_readback_m": float(readback[right_index]),
                    "legal_joint_readback": state_gate,
                    "dof_order": order,
                }
                robot_states[state] = state_record
                for view in ("closing_axis", "isometric"):
                    position, orientation = _camera_pose(geometry, view)
                    print(
                        "CAPTURE_PROGRESS camera="
                        + json.dumps(
                            {
                                "view": view,
                                "position": position.tolist(),
                                "orientation_wxyz": orientation.tolist(),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    camera.set_world_pose(
                        position=position,
                        orientation=orientation,
                        camera_axes="usd",
                    )
                    for _ in range(8):
                        world.step(render=True)
                    capture_name = f"{robot}_{state}_{view}"
                    output = (
                        screenshot_root
                        / "asset_preflight"
                        / f"{capture_name}.png"
                    )
                    render_readback = save_camera_rgba_png(camera, output)
                    print(
                        f"CAPTURE_PROGRESS saved={output.resolve()}",
                        flush=True,
                    )
                    camera_position, camera_orientation = camera.get_world_pose(
                        camera_axes="usd"
                    )
                    captures.append(
                        validate_screenshot(
                            output.resolve(strict=True),
                            artifact_root=screenshot_root,
                            phase="asset_preflight",
                            capture_name=capture_name,
                            gate_status="PASS" if state_gate else "FAIL",
                            camera={
                                "runtime": "isaacsim.sensors.camera.Camera",
                                "position_world_m": np.asarray(
                                    camera_position
                                ).tolist(),
                                "orientation_wxyz": np.asarray(
                                    camera_orientation
                                ).tolist(),
                                "target_world_m": geometry["target_world_m"],
                                "view": view,
                                "resolution": [1280, 900],
                                "render_readback": render_readback,
                            },
                            simulation={
                                "stage_asset": str(asset),
                                "robot": robot,
                                "state": state,
                                "physics_frequency_hz": 60,
                                "solve_articulation_contact_last": bool(
                                    physics_context.get_solve_articulation_contact_last()
                                ),
                                "dof_order": order,
                                "joint_positions": readback.tolist(),
                                "surface_gap_m": geometry["surface_gap_m"],
                            },
                        )
                    )
            robot_states["gates"] = {
                "aperture_monotonic": (
                    robot_states["open"]["surface_gap_m"]
                    > robot_states["closed"]["surface_gap_m"]
                ),
                "legal_joint_readback": (
                    robot_states["open"]["legal_joint_readback"]
                    and robot_states["closed"]["legal_joint_readback"]
                ),
            }
            robots[robot] = robot_states
            world.stop()

        required = {
            "asset_preflight": profile["screenshots"]["required_captures"][
                "asset_preflight"
            ]
        }
        manifest = build_screenshot_manifest(
            captures=captures,
            required_captures=required,
            artifact_root=screenshot_root,
        )
        numeric_pass = all(
            all(robot["gates"].values()) for robot in robots.values()
        )
        report = {
            "schema_version": 1,
            "status": (
                "PASS"
                if manifest["status"] == "PASS" and numeric_pass
                else "FAIL"
            ),
            "restart_boundary": EXPECTED_RESTART_BOUNDARY,
            "runtime": profile["runtime"],
            "robots": robots,
            "manifest": manifest,
            "screenshot_root_absolute": str(screenshot_root),
            "acceptance_note": (
                "Screenshots are required evidence, but numeric aperture and "
                "joint readback gates determine kinematic PASS."
            ),
        }
        _write_json(report_path, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "report": str(report_path),
                    "screenshot_root": str(screenshot_root),
                    "screenshots": [
                        capture["absolute_path"] for capture in captures
                    ],
                },
                indent=2,
            ),
            flush=True,
        )
    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        _write_json(
            report_path,
            {
                "schema_version": 1,
                "status": "FAIL",
                "restart_boundary": EXPECTED_RESTART_BOUNDARY,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "screenshot_root_absolute": str(screenshot_root),
            },
        )
        raise
    finally:
        app.close()
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
