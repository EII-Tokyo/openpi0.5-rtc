#!/usr/bin/env python3
"""Run a no-bottle numerical drive probe on one isolated CAD-finger Stage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback
from typing import Any

import numpy as np

from tools.aloha1_mapping.cad_finger_task5_structure import FINGER_DOF_NAMES
from tools.aloha1_mapping.isaac_screenshot import save_camera_rgba_png

ROOT = Path(__file__).resolve().parents[1]
ARTICULATION_PATH = "/workcell/vx300s_left/vx300s_left"
ARM_DOF_NAMES = (
    "vx300s_left_waist",
    "vx300s_left_shoulder",
    "vx300s_left_elbow",
    "vx300s_left_forearm_roll",
    "vx300s_left_wrist_angle",
    "vx300s_left_wrist_rotate",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _drive_snapshot(stage: Any) -> dict[str, Any]:
    from pxr import UsdPhysics

    result = {}
    for side in ("left", "right"):
        path = f"/workcell/joints/vx300s_left_{side}_finger"
        prim = stage.GetPrimAtPath(path)
        drive = UsdPhysics.DriveAPI(prim, "linear")
        result[side] = {
            "path": path,
            "type": drive.GetTypeAttr().Get(),
            "stiffness": float(drive.GetStiffnessAttr().Get()),
            "damping": float(drive.GetDampingAttr().Get()),
            "max_force": float(drive.GetMaxForceAttr().Get()),
            "target_position": float(
                drive.GetTargetPositionAttr().Get()
            ),
        }
    return result


def _arm_joint_paths() -> dict[str, str]:
    return {
        name: f"/workcell/joints/{name}" for name in ARM_DOF_NAMES
    }


def _arm_drive_snapshot(stage: Any) -> dict[str, Any]:
    from pxr import UsdPhysics

    result = {}
    for name, path in _arm_joint_paths().items():
        prim = stage.GetPrimAtPath(path)
        drive = UsdPhysics.DriveAPI(prim, "angular")
        result[name] = {
            "path": path,
            "type": drive.GetTypeAttr().Get(),
            "stiffness": float(drive.GetStiffnessAttr().Get()),
            "damping": float(drive.GetDampingAttr().Get()),
            "max_force": float(drive.GetMaxForceAttr().Get()),
            "target_position": float(
                drive.GetTargetPositionAttr().Get()
            ),
        }
    return result


def _world_pose(articulation: Any) -> dict[str, list[float]]:
    position, orientation = articulation.get_world_pose()
    return {
        "position_m": np.asarray(position, dtype=float).tolist(),
        "orientation_wxyz": np.asarray(
            orientation,
            dtype=float,
        ).tolist(),
    }


def _pose_translation_delta(
    first: dict[str, list[float]],
    second: dict[str, list[float]],
) -> float:
    return float(
        np.linalg.norm(
            np.asarray(second["position_m"])
            - np.asarray(first["position_m"])
        )
    )


def _trajectory(
    *,
    world: Any,
    articulation: Any,
    order: list[str],
    left_index: int,
    right_index: int,
    start: tuple[float, float],
    target: tuple[float, float],
    name: str,
    steps: int,
    capture: Any | None = None,
) -> dict[str, Any]:
    from isaacsim.core.utils.types import ArticulationAction

    world.reset()
    base_by_name = {
        "vx300s_left_waist": 0.0,
        "vx300s_left_shoulder": -0.96,
        "vx300s_left_elbow": 1.16,
        "vx300s_left_forearm_roll": 0.0,
        "vx300s_left_wrist_angle": -0.3,
        "vx300s_left_wrist_rotate": 0.0,
    }
    start_q = np.asarray(
        [base_by_name.get(joint, 0.0) for joint in order],
        dtype=np.float32,
    )
    start_q[left_index], start_q[right_index] = start
    target_q = start_q.copy()
    target_q[left_index], target_q[right_index] = target
    articulation.set_joint_positions(start_q)
    start_capture = None
    if capture is not None:
        start_capture = capture(
            phase="start",
            frame=0,
            command=start_q,
        )
        world.play()
    injected = np.asarray(
        articulation.get_joint_positions(),
        dtype=np.float64,
    )
    base_pose_start = _world_pose(articulation)
    trace = []
    for frame in range(1, steps + 1):
        alpha = frame / steps
        command = start_q + alpha * (target_q - start_q)
        articulation.get_articulation_controller().apply_action(
            ArticulationAction(joint_positions=command)
        )
        world.step(render=False)
        readback = np.asarray(
            articulation.get_joint_positions(),
            dtype=np.float64,
        )
        trace.append(
            {
                "frame": frame,
                "time_s": frame / 60.0,
                "command_left_m": float(command[left_index]),
                "command_right_m": float(command[right_index]),
                "readback_left_m": float(readback[left_index]),
                "readback_right_m": float(readback[right_index]),
                "all_dof_readback": readback.tolist(),
            }
        )
    final = np.asarray(
        articulation.get_joint_positions(),
        dtype=np.float64,
    )
    base_pose_end = _world_pose(articulation)
    end_capture = None
    if capture is not None:
        end_capture = capture(
            phase="end",
            frame=steps,
            command=target_q,
        )
    intended = []
    non_target_errors = []
    for index, side in ((left_index, "left"), (right_index, "right")):
        displacement = target_q[index] - start_q[index]
        actual = final[index] - injected[index]
        if abs(displacement) > 1.0e-12:
            intended.append(
                {
                    "side": side,
                    "commanded_displacement_m": float(displacement),
                    "actual_displacement_m": float(actual),
                    "direction_correct": bool(
                        np.sign(actual) == np.sign(displacement)
                    ),
                    "final_error_m": float(
                        abs(final[index] - target_q[index])
                    ),
                }
            )
        else:
            non_target_errors.append(
                {
                    "side": side,
                    "drift_m": float(abs(actual)),
                }
            )
    return {
        "name": name,
        "start_target_m": list(start),
        "end_target_m": list(target),
        "steps": steps,
        "fresh_world_reset": True,
        "injected_readback_m": [
            float(injected[left_index]),
            float(injected[right_index]),
        ],
        "final_readback_m": [
            float(final[left_index]),
            float(final[right_index]),
        ],
        "intended_joint_results": intended,
        "non_target_finger_results": non_target_errors,
        "base_pose_start": base_pose_start,
        "base_pose_end": base_pose_end,
        "base_translation_drift_m": _pose_translation_delta(
            base_pose_start,
            base_pose_end,
        ),
        "maximum_arm_dof_drift": float(
            np.max(np.abs(final[:6] - injected[:6]))
        ),
        "trace": trace,
        "screenshots": (
            []
            if capture is None
            else [start_capture, end_capture]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--screenshot-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stage_path = args.stage.resolve(strict=True)
    stage_hash_before = _sha256(stage_path)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    exit_code = 1
    try:
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.core.utils.stage import open_stage
        from isaacsim.sensors.camera import Camera
        from omni.physx import get_physx_interface
        from pxr import Gf
        from pxr import Usd
        from pxr import UsdLux

        from tools.validate_aloha_viper_cad_finger_task5_structure import _hide_non_target_visuals
        from tools.validate_aloha_viper_cad_finger_task5_structure import _read_nonblank_rgba
        from tools.validate_aloha_viper_cad_finger_task5_structure import _set_view_visibility

        if not open_stage(str(stage_path)):
            raise RuntimeError(f"failed to open {stage_path}")
        stage = get_current_stage()
        hidden_visuals: list[str] = []
        if args.screenshot_root is not None:
            stage.SetEditTarget(stage.GetSessionLayer())
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                hidden_visuals = _hide_non_target_visuals(stage)
                dome = UsdLux.DomeLight.Define(
                    stage, "/workcell/Task5DriveProbeSession/Dome"
                )
                dome.CreateIntensityAttr(700.0)
                dome.CreateColorAttr(Gf.Vec3f(0.9, 0.92, 1.0))
                key = UsdLux.DistantLight.Define(
                    stage, "/workcell/Task5DriveProbeSession/Key"
                )
                key.CreateIntensityAttr(1100.0)
                key.CreateAngleAttr(1.0)
        world = World(
            stage_units_in_meters=1.0,
            backend="numpy",
            device="cpu",
            physics_dt=1.0 / 60.0,
            rendering_dt=1.0 / 60.0,
        )
        world.get_physics_context().set_solve_articulation_contact_last(True)
        articulation = SingleArticulation(
            prim_path=ARTICULATION_PATH,
            name=f"cad_finger_drive_probe_{args.profile}",
            reset_xform_properties=False,
        )
        world.scene.add(articulation)
        camera = None
        camera_pose = None
        screenshot_root = None
        if args.screenshot_root is not None:
            screenshot_root = args.screenshot_root.resolve()
            screenshot_root.mkdir(parents=True, exist_ok=True)
            structure_report = json.loads(
                (
                    ROOT
                    / "reports/aloha1_mapping/"
                    "aloha_viper_cad_finger_task5_structure.json"
                ).read_text(encoding="utf-8")
            )
            camera_pose = structure_report["camera_poses"]["base_oblique"]
            camera = Camera(
                prim_path="/workcell/Task5DriveProbeSession/Camera",
                name=f"cad_finger_drive_probe_camera_{args.profile}",
                resolution=(1280, 900),
                frequency=60,
            )
            world.scene.add(camera)
        world.reset()
        if camera is not None:
            camera.initialize()
            camera.set_clipping_range(0.01, 10.0)
            _set_view_visibility(stage, "base_oblique")
            camera.set_world_pose(
                position=np.asarray(camera_pose["position_world_m"]),
                orientation=np.asarray(camera_pose["orientation_wxyz"]),
                camera_axes="usd",
            )
        order = list(articulation.dof_names)
        left_index = order.index(FINGER_DOF_NAMES[0])
        right_index = order.index(FINGER_DOF_NAMES[1])
        drive = _drive_snapshot(stage)
        physx_interface = get_physx_interface()

        class Capture:
            def __call__(
                self,
                *,
                phase: str,
                frame: int,
                command: np.ndarray,
            ) -> dict[str, Any]:
                if (
                    camera is None
                    or camera_pose is None
                    or screenshot_root is None
                ):
                    raise RuntimeError("capture requested without camera")
                world.pause()
                camera.initialize()
                camera.set_clipping_range(0.01, 10.0)
                camera.set_world_pose(
                    position=np.asarray(camera_pose["position_world_m"]),
                    orientation=np.asarray(
                        camera_pose["orientation_wxyz"]
                    ),
                    camera_axes="usd",
                )
                physx_interface.update_transformations(
                    True, True, False, False  # noqa: FBT003
                )
                for _ in range(8):
                    world.render()
                pixels = _read_nonblank_rgba(world, camera)
                path = (
                    screenshot_root
                    / f"{args.profile}_symmetric_{phase}_raw.png"
                )
                readback = np.asarray(
                    articulation.get_joint_positions(),
                    dtype=np.float64,
                )
                render = save_camera_rgba_png(
                    camera,
                    path,
                    rgba=pixels,
                )
                return {
                    "phase": phase,
                    "frame": frame,
                    "time_s": frame / 60.0,
                    "absolute_path": str(path.resolve()),
                    "sha256": _sha256(path),
                    "resolution": [1280, 900],
                    "render_readback": render,
                    "command_left_m": float(command[left_index]),
                    "command_right_m": float(command[right_index]),
                    "readback_left_m": float(readback[left_index]),
                    "readback_right_m": float(readback[right_index]),
                    "base_pose": _world_pose(articulation),
                    "camera": camera_pose,
                    "visual_review": "PENDING_VISUAL_MODEL_REVIEW",
                    "acceptance_boundary": (
                        "NUMERIC DRIVE FAILURE AUXILIARY EVIDENCE; "
                        "NO BOTTLE OR GRASP CLAIM"
                    ),
                }

        capture = Capture() if camera is not None else None
        trajectories = [
            _trajectory(
                world=world,
                articulation=articulation,
                order=order,
                left_index=left_index,
                right_index=right_index,
                start=(0.057, -0.057),
                target=(0.021, -0.057),
                name="left_only_close",
                steps=args.steps,
            ),
            _trajectory(
                world=world,
                articulation=articulation,
                order=order,
                left_index=left_index,
                right_index=right_index,
                start=(0.057, -0.057),
                target=(0.057, -0.021),
                name="right_only_close",
                steps=args.steps,
            ),
            _trajectory(
                world=world,
                articulation=articulation,
                order=order,
                left_index=left_index,
                right_index=right_index,
                start=(0.057, -0.057),
                target=(0.021, -0.021),
                name="symmetric_close",
                steps=args.steps,
                capture=capture,
            ),
        ]
        intended_results = [
            result
            for trajectory in trajectories
            for result in trajectory["intended_joint_results"]
        ]
        non_target_results = [
            result
            for trajectory in trajectories
            for result in trajectory["non_target_finger_results"]
        ]
        gates = {
            "all_intended_directions_correct": all(
                result["direction_correct"] for result in intended_results
            ),
            "all_intended_final_errors_within_1mm": all(
                result["final_error_m"] <= 0.001
                for result in intended_results
            ),
            "all_non_target_finger_drift_within_1mm": all(
                result["drift_m"] <= 0.001
                for result in non_target_results
            ),
            "all_arm_dof_drift_within_1mm": all(
                trajectory["maximum_arm_dof_drift"] <= 0.001
                for trajectory in trajectories
            ),
            "stage_immutable": _sha256(stage_path) == stage_hash_before,
            "no_bottle": True,
        }
        report = {
            "schema_version": 1,
            "status": "PASS" if all(gates.values()) else "FAIL",
            "gate": "NUMERIC_DIAGNOSTIC_PROBE_NOT_ACCEPTANCE_TEST",
            "profile": args.profile,
            "stage": {
                "absolute_path": str(stage_path),
                "sha256_before": stage_hash_before,
                "sha256_after": _sha256(stage_path),
            },
            "runtime": {
                "isaac_sim": "5.1.0.0",
                "kit": "107.3.3",
                "physx": "107.3.26",
                "physics_frequency_hz": 60,
                "solve_articulation_contact_last": True,
            },
            "articulation": {
                "path": ARTICULATION_PATH,
                "dof_order": order,
            },
            "drive_readback": drive,
            "arm_drive_readback": _arm_drive_snapshot(stage),
            "trajectories": trajectories,
            "gates": gates,
            "scope": {
                "screenshot_acceptance": (
                    "PENDING_VISUAL_MODEL_REVIEW"
                    if camera is not None
                    else "NOT_RUN"
                ),
                "collision_contact": "NOT_RUN",
                "bottle_contact_grasp": "NOT_RUN",
                "task8": "NOT_RUN",
            },
            "session_only_hidden_visuals": hidden_visuals,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"status={report['status']}")
        print(f"profile={args.profile}")
        print(f"report={args.output.resolve()}")
        exit_code = 0
    except Exception:
        traceback.print_exc()
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
