#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Capture Task 7 left-arm pose evidence without inventing a right arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback
from typing import Any

import numpy as np

from tools.aloha1_mapping.isaac_screenshot import look_at_orientation_wxyz
from tools.aloha1_mapping.screenshot_manifest import build_screenshot_manifest
from tools.aloha1_mapping.screenshot_manifest import validate_screenshot
from tools.capture_aloha_viper_cad_finger_task5_numeric_pass_viewport import _capture_viewport_png
from tools.capture_aloha_viper_cad_finger_task5_numeric_pass_viewport import select_trace_frames

ROOT = Path(__file__).resolve().parents[1]
NUMERIC_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_drive_probe_arm_max_force_over_combined.json"
)
OUTPUT_ROOT = (
    ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
    "task7_robot_scope/pose_evidence_attempt5"
)
OUTPUT_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task7_pose_screenshots_raw.json"
)
PHASE = "task7_certified_left_arm_pose_replay"
RIGHT_ARM_BLOCKER = (
    "HARD_BLOCKER_APPROVED_STAGE_MISSING_FOLLOWER_RIGHT"
)
ROBOT_VISUAL_ROOT = "/workcell/vx300s_left"
ARTICULATION_PATH = "/workcell/vx300s_left/vx300s_left"
FINGER_MESHES = {
    side: (
        f"/workcell/vx300s_left/vx300s_left_{side}_finger_link/"
        f"visuals/diagnostic_supplier_cad_{side}_finger/mesh"
    )
    for side in ("left", "right")
}
PROVEN_CLOSEUP_REVIEW = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_numeric_pass_screenshot_review.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _world_points(stage: Any, prim: Any) -> np.ndarray:
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh(prim)
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    return np.asarray(
        [
            list(transform.Transform(point))
            for point in (mesh.GetPointsAttr().Get() or [])
        ],
        dtype=np.float64,
    )


def _set_visual_scope(stage: Any, *, closeup: bool) -> list[str]:
    """Use the session layer to expose the arm and audited CAD fingers."""

    if closeup:
        from tools.validate_aloha_viper_cad_finger_task5_structure import _hide_non_target_visuals
        from tools.validate_aloha_viper_cad_finger_task5_structure import _set_view_visibility

        hidden = _hide_non_target_visuals(stage)
        hidden.extend(_set_view_visibility(stage, "base_oblique"))
        return sorted(set(hidden))

    from pxr import UsdGeom

    from tools.validate_aloha_viper_cad_finger_task5_structure import _hide_non_target_visuals

    hidden = _hide_non_target_visuals(stage)
    finger_tokens = tuple(
        f"/diagnostic_supplier_cad_{side}_finger/" for side in ("left", "right")
    )
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not prim.IsA(UsdGeom.Imageable):
            continue
        imageable = UsdGeom.Imageable(prim)
        if prim.IsA(UsdGeom.Gprim) and not path.startswith(
            f"{ROBOT_VISUAL_ROOT}/"
        ):
            imageable.MakeInvisible()
            hidden.append(path)
            continue
        if "/collisions" in path or "/sites" in path:
            imageable.MakeInvisible()
            hidden.append(path)
            continue
        if "_finger_link/visuals/" in path and not any(
            token in path for token in finger_tokens
        ):
            imageable.MakeInvisible()
            hidden.append(path)
            continue
        if "_camera_focus" in path or "_gripper_prop_link" in path:
            imageable.MakeInvisible()
            hidden.append(path)
            continue
        if path.startswith(f"{ROBOT_VISUAL_ROOT}/"):
            imageable.MakeVisible()
    return hidden


def _visible_robot_meshes(stage: Any) -> list[Any]:
    from pxr import Usd
    from pxr import UsdGeom

    meshes = []
    root = stage.GetPrimAtPath(ROBOT_VISUAL_ROOT)
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if not (
            prim.IsA(UsdGeom.Mesh)
            and path.startswith(f"{ROBOT_VISUAL_ROOT}/")
            and "/visuals/" in path
        ):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        meshes.append(prim)
    if not meshes:
        raise RuntimeError("no visible robot meshes for screenshot")
    return meshes


def _point_cloud(stage: Any, prims: list[Any]) -> np.ndarray:
    clouds = [_world_points(stage, prim) for prim in prims]
    clouds = [cloud for cloud in clouds if cloud.size]
    if not clouds:
        raise RuntimeError("visible robot meshes contain no points")
    return np.concatenate(clouds)


def _projection(camera: Any, points: np.ndarray) -> dict[str, Any]:
    pixels = np.asarray(
        camera.get_image_coords_from_world_points(points),
        dtype=np.float64,
    )
    finite = pixels[np.isfinite(pixels).all(axis=1)]
    if not len(finite):
        raise RuntimeError("camera projection has no finite points")
    minimum = finite.min(axis=0)
    maximum = finite.max(axis=0)
    return {
        "bbox_min_px": minimum.tolist(),
        "bbox_max_px": maximum.tolist(),
        "bbox_center_px": ((minimum + maximum) / 2.0).tolist(),
        "finite_point_count": len(finite),
        "fully_in_frame": bool(
            minimum[0] >= 0.0
            and minimum[1] >= 0.0
            and maximum[0] < 1280.0
            and maximum[1] < 900.0
        ),
    }


def _camera_specs(
    robot_points: np.ndarray,
    proven_closeup: dict[str, Any],
) -> dict[str, dict[str, list[float]]]:
    robot_min = robot_points.min(axis=0)
    robot_max = robot_points.max(axis=0)
    robot_target = (robot_min + robot_max) / 2.0
    span = float(np.linalg.norm(robot_max - robot_min))
    full_direction = np.asarray([-1.0, -0.85, 0.72])
    full_direction /= np.linalg.norm(full_direction)
    full_position = robot_target + full_direction * (2.2 * span + 0.35)
    return {
        "full_arm_oblique": {
            "position_world_m": full_position.tolist(),
            "target_world_m": robot_target.tolist(),
        },
        "gripper_closeup": {
            "position_world_m": proven_closeup["position_world_m"],
            "target_world_m": proven_closeup["target_world_m"],
            "orientation_wxyz": proven_closeup["orientation_wxyz"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numeric-report", type=Path, default=NUMERIC_REPORT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=OUTPUT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    numeric_path = args.numeric_report.resolve(strict=True)
    numeric_hash = _sha256(numeric_path)
    numeric = json.loads(numeric_path.read_text(encoding="utf-8"))
    if numeric["status"] != "PASS":
        raise RuntimeError("numeric source is not PASS")
    symmetric = next(
        item
        for item in numeric["trajectories"]
        if item["name"] == "symmetric_close"
    )
    states = select_trace_frames(symmetric["trace"])
    proven_review = json.loads(
        PROVEN_CLOSEUP_REVIEW.resolve(strict=True).read_text(
            encoding="utf-8"
        )
    )
    if proven_review["status"] != "PASS":
        raise RuntimeError("proven closeup screenshot review is not PASS")
    proven_closeup = proven_review["captures"][0]["camera"]
    stage_path = Path(numeric["stage"]["absolute_path"]).resolve(strict=True)
    stage_hash = numeric["stage"]["sha256_after"]
    if _sha256(stage_path) != stage_hash:
        raise RuntimeError("protected diagnostic Stage hash mismatch")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    raw_root = output_root / "screenshots_raw"
    raw_root.mkdir(parents=True)

    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import get_current_stage
    from isaacsim.core.utils.stage import open_stage
    from isaacsim.sensors.camera import Camera
    from omni.kit.viewport.utility import get_active_viewport
    from omni.physx import get_physx_interface
    from pxr import Gf
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdLux

    app = globals()["_SIMULATION_APP"]
    if not open_stage(str(stage_path)):
        raise RuntimeError(f"failed to open {stage_path}")
    stage = get_current_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        dome = UsdLux.DomeLight.Define(
            stage, "/workcell/Task7PoseEvidence/Dome"
        )
        dome.CreateIntensityAttr(800.0)
        dome.CreateColorAttr(Gf.Vec3f(0.9, 0.92, 1.0))
        key = UsdLux.DistantLight.Define(
            stage, "/workcell/Task7PoseEvidence/Key"
        )
        key.CreateIntensityAttr(1200.0)

    world = World(
        stage_units_in_meters=1.0,
        backend="numpy",
        device="cpu",
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
    )
    articulation = SingleArticulation(
        prim_path=ARTICULATION_PATH,
        name="task7_pose_evidence",
        reset_xform_properties=False,
    )
    world.scene.add(articulation)
    camera = Camera(
        prim_path="/workcell/Task7PoseEvidence/Camera",
        name="task7_pose_camera",
        resolution=(1280, 900),
        frequency=60,
    )
    world.scene.add(camera)
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("no active Isaac viewport")

    world.reset()
    world.pause()
    first_qpos = np.asarray(
        states[0]["record"]["all_dof_readback"],
        dtype=np.float32,
    )
    articulation.set_joint_positions(first_qpos)
    get_physx_interface().update_transformations(
        True, True, False, False  # noqa: FBT003
    )
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        _set_visual_scope(stage, closeup=False)
    robot_points = _point_cloud(stage, _visible_robot_meshes(stage))
    camera_specs = _camera_specs(robot_points, proven_closeup)

    captures = []
    session_hidden_by_view: dict[str, list[str]] = {}
    for view in ("full_arm_oblique", "gripper_closeup"):
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            session_hidden_by_view[view] = _set_visual_scope(
                stage,
                closeup=view == "gripper_closeup",
            )
        spec = camera_specs[view]
        position = np.asarray(spec["position_world_m"])
        target = np.asarray(spec["target_world_m"])
        orientation = np.asarray(
            spec.get(
                "orientation_wxyz",
                look_at_orientation_wxyz(position, target),
            )
        )
        for selected in states:
            record = selected["record"]
            world.reset()
            world.pause()
            articulation.set_joint_positions(
                np.asarray(record["all_dof_readback"], dtype=np.float32)
            )
            get_physx_interface().update_transformations(
                True, True, False, False  # noqa: FBT003
            )
            camera.set_clipping_range(0.01, 10.0)
            camera.set_world_pose(
                position=position,
                orientation=orientation,
                camera_axes="usd",
            )
            viewport.camera_path = Sdf.Path(camera.prim_path)
            for _ in range(24):
                app.update()
            current_robot_points = _point_cloud(
                stage, _visible_robot_meshes(stage)
            )
            current_finger_clouds = {
                side: _world_points(
                    stage, stage.GetPrimAtPath(FINGER_MESHES[side])
                )
                for side in ("left", "right")
            }
            projections = {
                "robot": _projection(camera, current_robot_points),
                "left_finger": _projection(
                    camera, current_finger_clouds["left"]
                ),
                "right_finger": _projection(
                    camera, current_finger_clouds["right"]
                ),
            }
            capture_name = f"{view}_{selected['phase']}"
            destination = raw_root / f"{capture_name}_raw.png"
            _capture_viewport_png(app, viewport, destination)
            actual_position, actual_orientation = camera.get_world_pose(
                camera_axes="usd"
            )
            captures.append(
                validate_screenshot(
                    destination.resolve(strict=True),
                    artifact_root=output_root,
                    phase=PHASE,
                    capture_name=capture_name,
                    gate_status="PASS",
                    camera={
                        "view": view,
                        "resolution": [1280, 900],
                        "position_world_m": np.asarray(
                            actual_position
                        ).tolist(),
                        "orientation_wxyz": np.asarray(
                            actual_orientation
                        ).tolist(),
                        "target_world_m": target.tolist(),
                        "fixed_camera_within_view": True,
                        "capture_backend": (
                            "omni.kit.viewport.utility."
                            "capture_viewport_to_file"
                        ),
                        "projections": projections,
                    },
                    simulation={
                        "robot": "follower_left",
                        "isaac_sim": "5.1.0.0",
                        "kit": "107.3.3",
                        "physx": "107.3.26",
                        "stage_absolute_path": str(stage_path),
                        "stage_sha256": stage_hash,
                        "source_numeric_report": str(numeric_path),
                        "source_numeric_report_sha256": numeric_hash,
                        "source_trajectory": "symmetric_close",
                        "phase": selected["phase"],
                        "source_runtime_frame": int(record["frame"]),
                        "source_runtime_time_s": float(record["time_s"]),
                        "all_dof_readback": record["all_dof_readback"],
                        "readback_left_m": float(
                            record["readback_left_m"]
                        ),
                        "readback_right_m": float(
                            record["readback_right_m"]
                        ),
                        "capture_method": (
                            "CERTIFIED_RUNTIME_READBACK_REPLAY_AUXILIARY"
                        ),
                        "physics_steps_added": 0,
                        "acceptance_boundary": (
                            "POSE/DIRECTION AUXILIARY EVIDENCE ONLY; RUNTIME "
                            "NUMERIC REPORT REMAINS AUTHORITATIVE"
                        ),
                    },
                )
            )

    manifest = build_screenshot_manifest(
        captures=captures,
        required_captures={
            PHASE: [item["capture_name"] for item in captures]
        },
        artifact_root=output_root,
    )
    fixed_cameras = {
        view: len(
            {
                (
                    tuple(item["camera"]["position_world_m"]),
                    tuple(item["camera"]["orientation_wxyz"]),
                    tuple(item["camera"]["target_world_m"]),
                )
                for item in captures
                if item["camera"]["view"] == view
            }
        )
        == 1
        for view in ("full_arm_oblique", "gripper_closeup")
    }
    report = {
        "schema_version": 1,
        "status": "PARTIAL",
        "capture_status": (
            "PASS"
            if manifest["status"] == "PASS" and all(fixed_cameras.values())
            else "FAIL"
        ),
        "captures": captures,
        "manifest": manifest,
        "fixed_camera_within_each_view": fixed_cameras,
        "session_hidden_by_view": session_hidden_by_view,
        "visual_model_review": "PENDING",
        "right_arm": {
            "status": "NOT_RUN",
            "blocker": RIGHT_ARM_BLOCKER,
        },
        "source_stage_immutable": _sha256(stage_path) == stage_hash,
        "source_report_immutable": _sha256(numeric_path) == numeric_hash,
        "task7": "PARTIAL",
        "task8": "NOT_RUN",
    }
    _write_json(args.report.resolve(), report)
    print(f"status={report['status']}")
    print(f"capture_status={report['capture_status']}")
    print(f"capture_count={len(captures)}")
    print(f"report={args.report.resolve()}")
    return 0 if report["capture_status"] == "PASS" else 1


def run() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "width": 1280, "height": 900})
    globals()["_SIMULATION_APP"] = app
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
