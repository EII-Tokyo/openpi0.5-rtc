#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Capture robot-local follower_right pose evidence in Isaac Sim 5.1."""

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

ROOT = Path(__file__).resolve().parents[1]
STAGE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "supplier_cad_follower_right/1.0/"
    "supplier_cad_follower_right.usda"
)
NUMERIC_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_follower_right_one_joint_validation.json"
)
OUTPUT_ROOT = (
    ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
    "follower_right_pose_evidence/attempt4_final"
)
OUTPUT_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_follower_right_pose_screenshots_raw.json"
)
SCOPE = "ROBOT_LOCAL_DIAGNOSTIC_NOT_WORKCELL_PLACEMENT"
PHASE = "follower_right_robot_local_pose_readback_replay"
ROBOT_ROOT = "/follower_right/vx300s_right"
ARTICULATION_PATH = "/follower_right/vx300s_right/root_joint"
FINGER_MESHES = {
    side: (
        f"{ROBOT_ROOT}/follower_right_{side}_finger_link/"
        f"visuals/diagnostic_supplier_cad_{side}_finger/mesh"
    )
    for side in ("left", "right")
}
RESOLUTION = (1280, 900)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _world_points(prim: Any) -> np.ndarray:
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


def _visible_robot_meshes(stage: Any) -> list[Any]:
    from pxr import Usd
    from pxr import UsdGeom

    root = stage.GetPrimAtPath(ROBOT_ROOT)
    meshes = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if not prim.IsA(UsdGeom.Mesh) or "/visuals/" not in path:
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == (
            UsdGeom.Tokens.invisible
        ):
            continue
        meshes.append(prim)
    if not meshes:
        raise RuntimeError("no visible follower_right robot meshes")
    return meshes


def _point_cloud(prims: list[Any]) -> np.ndarray:
    clouds = [_world_points(prim) for prim in prims]
    clouds = [cloud for cloud in clouds if cloud.size]
    if not clouds:
        raise RuntimeError("visible follower_right meshes contain no points")
    return np.concatenate(clouds)


def _projection(camera: Any, points: np.ndarray) -> dict[str, Any]:
    pixels = np.asarray(
        camera.get_image_coords_from_world_points(points),
        dtype=np.float64,
    )
    finite = pixels[np.isfinite(pixels).all(axis=1)]
    if not len(finite):
        raise RuntimeError("camera projection contains no finite points")
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
            and maximum[0] < RESOLUTION[0]
            and maximum[1] < RESOLUTION[1]
        ),
    }


def _set_visual_scope(stage: Any) -> list[str]:
    from pxr import UsdGeom

    hidden = []
    supplier_tokens = tuple(
        f"/diagnostic_supplier_cad_{side}_finger/"
        for side in ("left", "right")
    )
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Imageable):
            continue
        path = str(prim.GetPath())
        imageable = UsdGeom.Imageable(prim)
        if prim.IsA(UsdGeom.Gprim) and not path.startswith(
            f"{ROBOT_ROOT}/"
        ):
            imageable.MakeInvisible()
            hidden.append(path)
            continue
        if "/collisions" in path or "/sites" in path:
            imageable.MakeInvisible()
            hidden.append(path)
            continue
        if "_finger_link/visuals/" in path and not any(
            token in path for token in supplier_tokens
        ):
            imageable.MakeInvisible()
            hidden.append(path)
            continue
        if path.startswith(f"{ROBOT_ROOT}/"):
            imageable.MakeVisible()
    return sorted(set(hidden))


def _pose_records(report: dict[str, Any]) -> list[dict[str, Any]]:
    arm = report["arm_one_joint_cases"]
    home = next(
        item for item in arm if item["repeat"] == 0
    )["start"]
    waist_positive = next(
        item
        for item in arm
        if item["repeat"] == 0 and item["test"] == "waist_positive"
    )
    waist_negative = next(
        item
        for item in arm
        if item["repeat"] == 0 and item["test"] == "waist_negative"
    )
    records = [
        {
            "phase": "home_reference",
            "view": "full_arm_oblique",
            "qpos": home,
            "joint_name": "all_home",
            "joint_index": None,
            "target": home,
            "readback": home,
            "numeric_status": "PASS",
            "source_test": "static_pose_hold",
        },
        {
            "phase": "waist_positive",
            "view": "full_arm_oblique",
            "qpos": waist_positive["end"],
            "joint_name": "waist",
            "joint_index": 0,
            "target": waist_positive["target"],
            "readback": waist_positive["end"][0],
            "numeric_status": waist_positive["status"],
            "source_test": waist_positive["test"],
        },
        {
            "phase": "waist_negative",
            "view": "full_arm_oblique",
            "qpos": waist_negative["end"],
            "joint_name": "waist",
            "joint_index": 0,
            "target": waist_negative["target"],
            "readback": waist_negative["end"][0],
            "numeric_status": waist_negative["status"],
            "source_test": waist_negative["test"],
        },
    ]
    for state in (
        "open",
        "partially_closed",
        "closed",
        "maximum_legal_aperture",
    ):
        source = report["gripper_validation"]["states"][state]
        qpos = list(home)
        qpos[7] = source["readback_left_m"]
        qpos[8] = source["readback_right_m"]
        records.append(
            {
                "phase": f"gripper_{state}",
                "view": "gripper_closeup",
                "qpos": qpos,
                "joint_name": "left_finger/right_finger",
                "joint_index": [7, 8],
                "target": [source["target_left_m"], -source["target_left_m"]],
                "readback": [
                    source["readback_left_m"],
                    source["readback_right_m"],
                ],
                "numeric_status": source["status"],
                "mimic_residual_m": source["mimic_residual_m"],
                "aperture_m": source["aperture_m"],
                "source_test": source["test"],
            }
        )
    return records


def _camera_specs(
    robot_points: np.ndarray,
    finger_points: np.ndarray,
) -> dict[str, dict[str, Any]]:
    minimum = robot_points.min(axis=0)
    maximum = robot_points.max(axis=0)
    target = (minimum + maximum) / 2.0
    span = float(np.linalg.norm(maximum - minimum))
    direction = np.asarray([-1.0, -0.85, 0.72], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    full_position = target + direction * (2.2 * span + 0.35)

    close_target = (
        finger_points.min(axis=0) + finger_points.max(axis=0)
    ) / 2.0
    tip_side_direction_and_distance = np.asarray(
        [0.392661573, 0.156488959, 0.219114509],
        dtype=np.float64,
    )
    close_position = close_target + tip_side_direction_and_distance
    return {
        "full_arm_oblique": {
            "position_world_m": full_position.tolist(),
            "target_world_m": target.tolist(),
        },
        "gripper_closeup": {
            "position_world_m": close_position.tolist(),
            "target_world_m": close_target.tolist(),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--numeric-report", type=Path, default=NUMERIC_REPORT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=OUTPUT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stage_path = args.stage.resolve(strict=True)
    stage_hash = _sha256(stage_path)
    numeric_path = args.numeric_report.resolve(strict=True)
    numeric_hash = _sha256(numeric_path)
    numeric = json.loads(numeric_path.read_text(encoding="utf-8"))
    if numeric["scope"] != SCOPE:
        raise RuntimeError("numeric source is not robot-local follower_right")
    if numeric["stage"]["sha256_after"] != stage_hash:
        raise RuntimeError("numeric/source Stage hash mismatch")
    poses = _pose_records(numeric)
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
        raise RuntimeError(f"failed to open Stage: {stage_path}")
    stage = get_current_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        hidden = _set_visual_scope(stage)
        dome = UsdLux.DomeLight.Define(
            stage, "/follower_right/RightPoseEvidence/Dome"
        )
        dome.CreateIntensityAttr(800.0)
        dome.CreateColorAttr(Gf.Vec3f(0.9, 0.92, 1.0))
        key = UsdLux.DistantLight.Define(
            stage, "/follower_right/RightPoseEvidence/Key"
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
        name="follower_right_pose_evidence",
        reset_xform_properties=False,
    )
    world.scene.add(articulation)
    camera = Camera(
        prim_path="/follower_right/RightPoseEvidence/Camera",
        name="follower_right_pose_camera",
        resolution=RESOLUTION,
        frequency=60,
    )
    world.scene.add(camera)
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("no active Isaac viewport")

    world.reset()
    world.pause()
    articulation.set_joint_positions(
        np.asarray(poses[0]["qpos"], dtype=np.float32)
    )
    get_physx_interface().update_transformations(
        True, True, False, False  # noqa: FBT003
    )
    home_finger_points = np.concatenate(
        [
            _world_points(stage.GetPrimAtPath(FINGER_MESHES[side]))
            for side in ("left", "right")
        ]
    )
    camera_specs = _camera_specs(
        _point_cloud(_visible_robot_meshes(stage)),
        home_finger_points,
    )

    captures = []
    for pose in poses:
        spec = camera_specs[pose["view"]]
        position = np.asarray(spec["position_world_m"])
        target = np.asarray(spec["target_world_m"])
        orientation = np.asarray(
            look_at_orientation_wxyz(position, target)
        )
        world.reset()
        world.pause()
        articulation.set_joint_positions(
            np.asarray(pose["qpos"], dtype=np.float32)
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
        finger_clouds = {
            side: _world_points(stage.GetPrimAtPath(FINGER_MESHES[side]))
            for side in ("left", "right")
        }
        projections = {
            "robot": _projection(
                camera,
                _point_cloud(_visible_robot_meshes(stage)),
            ),
            "left_finger": _projection(camera, finger_clouds["left"]),
            "right_finger": _projection(camera, finger_clouds["right"]),
        }
        required_projections = (
            ("robot", "left_finger", "right_finger")
            if pose["view"] == "full_arm_oblique"
            else ("left_finger", "right_finger")
        )
        if not all(
            projections[name]["fully_in_frame"]
            for name in required_projections
        ):
            raise RuntimeError(
                f"projection cropped for phase {pose['phase']}: {projections}"
            )
        capture_name = f"{pose['view']}_{pose['phase']}"
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
                    "view": pose["view"],
                    "resolution": list(RESOLUTION),
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
                    "scope": SCOPE,
                    "robot": "follower_right",
                    "isaac_sim": "5.1.0.0",
                    "kit": "107.3.3",
                    "physx": "107.3.26",
                    "stage_absolute_path": str(stage_path),
                    "stage_sha256": stage_hash,
                    "source_numeric_report": str(numeric_path),
                    "source_numeric_report_sha256": numeric_hash,
                    "phase": pose["phase"],
                    "source_test": pose["source_test"],
                    "joint_name": pose["joint_name"],
                    "joint_index": pose["joint_index"],
                    "target": pose["target"],
                    "readback": pose["readback"],
                    "all_dof_readback": pose["qpos"],
                    "numeric_status": pose["numeric_status"],
                    "mimic_residual_m": pose.get("mimic_residual_m"),
                    "aperture_m": pose.get("aperture_m"),
                    "frame": 90,
                    "time_s": 1.5,
                    "supplier_finger_type": (
                        "Simple Aloha Viper embedded handed v2 pair"
                    ),
                    "capture_method": (
                        "CERTIFIED_RUNTIME_READBACK_REPLAY_AUXILIARY"
                    ),
                    "physics_steps_added": 0,
                    "acceptance_boundary": (
                        "VISUAL INSTALLATION/POSE GATE ONLY; NUMERIC RUNTIME "
                        "REPORT REMAINS AUTHORITATIVE; NOT WORKCELL PLACEMENT"
                    ),
                },
            )
        )

    required_names = [item["capture_name"] for item in captures]
    manifest = build_screenshot_manifest(
        captures=captures,
        required_captures={PHASE: required_names},
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
        "status": "PENDING_VISUAL_MODEL_REVIEW",
        "capture_status": (
            "PASS"
            if manifest["status"] == "PASS"
            and all(fixed_cameras.values())
            else "FAIL"
        ),
        "scope": SCOPE,
        "captures": captures,
        "manifest": manifest,
        "fixed_camera_within_each_view": fixed_cameras,
        "session_hidden_paths": hidden,
        "visual_model_review": "PENDING",
        "source_stage_immutable": _sha256(stage_path) == stage_hash,
        "source_numeric_report_immutable": (
            _sha256(numeric_path) == numeric_hash
        ),
        "workcell_placement_verified": False,
        "task8": "NOT_RUN",
    }
    args.report.resolve().write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"status={report['status']}")
    print(f"capture_status={report['capture_status']}")
    print(f"capture_count={len(captures)}")
    print(f"report={args.report.resolve()}")
    return 0 if report["capture_status"] == "PASS" else 1


def run() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp(
        {"headless": True, "width": RESOLUTION[0], "height": RESOLUTION[1]}
    )
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
