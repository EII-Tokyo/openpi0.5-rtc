#!/usr/bin/env python3
"""Capture static replays of a Task 5 runtime drive trace.

This deliberately runs in a process separate from the numerical physics
probe.  The approved Stage has no useful animation interval, and attaching a
Camera after the active physics trace repeatedly produced blank RGB frames.
The images are therefore auxiliary visualizations of recorded runtime qpos,
not same-frame physics or acceptance evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback
from typing import Any

import numpy as np

from tools.aloha1_mapping.isaac_screenshot import save_camera_rgba_png
from tools.aloha1_mapping.screenshot_manifest import build_screenshot_manifest
from tools.aloha1_mapping.screenshot_manifest import validate_screenshot

ROOT = Path(__file__).resolve().parents[1]
PHASE = "task5_runtime_readback_replay_auxiliary"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_symmetric_trace(
    report: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if report["status"] != "FAIL":
        raise RuntimeError("expected preserved numerical drive FAIL")
    if report["scope"]["bottle_contact_grasp"] != "NOT_RUN":
        raise RuntimeError("numeric report unexpectedly includes bottle work")
    matches = [
        item
        for item in report["trajectories"]
        if item["name"] == "symmetric_close"
    ]
    if len(matches) != 1:
        raise RuntimeError("expected exactly one symmetric_close trajectory")
    trajectory = matches[0]
    trace = trajectory["trace"]
    if len(trace) < 2:
        raise RuntimeError("symmetric trace is too short to replay")
    return trajectory, [trace[0], trace[-1]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numeric-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    numeric_report_path = args.numeric_report.resolve(strict=True)
    numeric_hash_before = _sha256(numeric_report_path)
    numeric_report = json.loads(
        numeric_report_path.read_text(encoding="utf-8")
    )
    trajectory, replay_frames = _load_symmetric_trace(numeric_report)
    stage_path = Path(
        numeric_report["stage"]["absolute_path"]
    ).resolve(strict=True)
    expected_stage_hash = numeric_report["stage"]["sha256_after"]
    if _sha256(stage_path) != expected_stage_hash:
        raise RuntimeError("diagnostic Stage hash drift before replay")

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "width": 1280, "height": 900})
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

        from tools.validate_aloha_viper_cad_finger_task5_structure import ARTICULATION_PATH
        from tools.validate_aloha_viper_cad_finger_task5_structure import FINGER_MESHES
        from tools.validate_aloha_viper_cad_finger_task5_structure import _hide_non_target_visuals
        from tools.validate_aloha_viper_cad_finger_task5_structure import _read_nonblank_rgba
        from tools.validate_aloha_viper_cad_finger_task5_structure import _set_view_visibility
        from tools.validate_aloha_viper_cad_finger_task5_structure import _world_points
        from tools.validate_aloha_viper_cad_finger_task5_structure import summarize_image_projection

        if not open_stage(str(stage_path)):
            raise RuntimeError(f"failed to open {stage_path}")
        stage = get_current_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            hidden_visuals = _hide_non_target_visuals(stage)
            dome = UsdLux.DomeLight.Define(
                stage, "/workcell/Task5DriveReplaySession/Dome"
            )
            dome.CreateIntensityAttr(700.0)
            dome.CreateColorAttr(Gf.Vec3f(0.9, 0.92, 1.0))
            key = UsdLux.DistantLight.Define(
                stage, "/workcell/Task5DriveReplaySession/Key"
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
        articulation = SingleArticulation(
            prim_path=ARTICULATION_PATH,
            name="cad_finger_drive_readback_replay",
            reset_xform_properties=False,
        )
        world.scene.add(articulation)
        camera = Camera(
            prim_path="/workcell/Task5DriveReplaySession/Camera",
            name="cad_finger_drive_readback_replay_camera",
            resolution=(1280, 900),
            frequency=60,
        )
        world.scene.add(camera)
        structure_report = json.loads(
            (
                ROOT
                / "reports/aloha1_mapping/"
                "aloha_viper_cad_finger_task5_structure.json"
            ).read_text(encoding="utf-8")
        )
        camera_pose = structure_report["camera_poses"]["base_oblique"]
        output_root = args.output_root.resolve()
        if output_root.exists():
            raise FileExistsError(
                f"replay screenshot output already exists: {output_root}"
            )
        raw_root = output_root / "screenshots_raw"
        raw_root.mkdir(parents=True)
        physx_interface = get_physx_interface()
        captures = []
        for trace_record in replay_frames:
            world.reset()
            camera.initialize()
            camera.set_clipping_range(0.01, 10.0)
            world.pause()
            qpos = np.asarray(
                trace_record["all_dof_readback"],
                dtype=np.float32,
            )
            articulation.set_joint_positions(qpos)
            physx_interface.update_transformations(
                True, True, False, False  # noqa: FBT003
            )
            _set_view_visibility(stage, "base_oblique")
            camera.set_world_pose(
                position=np.asarray(camera_pose["position_world_m"]),
                orientation=np.asarray(camera_pose["orientation_wxyz"]),
                camera_axes="usd",
            )
            for _ in range(8):
                world.render()
            pixels = _read_nonblank_rgba(world, camera)
            actual_position, actual_orientation = camera.get_world_pose(
                camera_axes="usd"
            )
            finger_points = np.concatenate(
                [
                    _world_points(stage, FINGER_MESHES[side])
                    for side in ("left", "right")
                ]
            )
            projection = summarize_image_projection(
                camera.get_image_coords_from_world_points(
                    finger_points
                ).tolist(),
                width=1280,
                height=900,
            )
            frame = int(trace_record["frame"])
            capture_name = f"root_frame_only_symmetric_frame_{frame:04d}"
            output = raw_root / f"{capture_name}_raw.png"
            render_readback = save_camera_rgba_png(
                camera,
                output,
                rgba=pixels,
            )
            captures.append(
                validate_screenshot(
                    output.resolve(strict=True),
                    artifact_root=output_root,
                    phase=PHASE,
                    capture_name=capture_name,
                    gate_status="PASS",
                    camera={
                        **camera_pose,
                        "view": "base_oblique",
                        "resolution": [1280, 900],
                        "runtime": "isaacsim.sensors.camera.Camera",
                        "render_readback": render_readback,
                        "actual_position_world_m": (
                            np.asarray(actual_position).tolist()
                        ),
                        "actual_orientation_wxyz": (
                            np.asarray(actual_orientation).tolist()
                        ),
                        "finger_projection": projection,
                    },
                    simulation={
                        "isaac_sim": "5.1.0.0",
                        "kit": "107.3.3",
                        "physx": "107.3.26",
                        "stage_absolute_path": str(stage_path),
                        "stage_sha256": expected_stage_hash,
                        "source_numeric_report": str(numeric_report_path),
                        "source_numeric_report_sha256": numeric_hash_before,
                        "source_trajectory": "symmetric_close",
                        "source_runtime_frame": frame,
                        "source_runtime_time_s": float(
                            trace_record["time_s"]
                        ),
                        "command_left_m": float(
                            trace_record["command_left_m"]
                        ),
                        "command_right_m": float(
                            trace_record["command_right_m"]
                        ),
                        "readback_left_m": float(
                            trace_record["readback_left_m"]
                        ),
                        "readback_right_m": float(
                            trace_record["readback_right_m"]
                        ),
                        "all_dof_readback": (
                            trace_record["all_dof_readback"]
                        ),
                        "replay_world_frame": int(
                            world.current_time_step_index
                        ),
                        "replay_world_time_s": float(world.current_time),
                        "capture_method": (
                            "RUNTIME_READBACK_REPLAY_AUXILIARY"
                        ),
                        "capture_physics_step": False,
                        "bottle_present": False,
                        "acceptance_boundary": (
                            "AUXILIARY QPOS REPLAY; NOT SAME-FRAME PHYSICS, "
                            "CONTACT, COLLISION, OR GRASP EVIDENCE"
                        ),
                    },
                )
            )

        manifest = build_screenshot_manifest(
            captures=captures,
            required_captures={
                PHASE: [
                    f"root_frame_only_symmetric_frame_{int(item['frame']):04d}"
                    for item in replay_frames
                ]
            },
            artifact_root=output_root,
        )
        gates = {
            "raw_screenshot_acquisition": manifest["status"] == "PASS",
            "two_runtime_trace_endpoints_replayed": len(captures) == 2,
            "stage_immutable": _sha256(stage_path) == expected_stage_hash,
            "numeric_report_immutable": (
                _sha256(numeric_report_path) == numeric_hash_before
            ),
            "no_bottle": True,
            "no_physics_step_in_replay": True,
            "dynamic_failure_preserved": numeric_report["status"] == "FAIL",
        }
        report = {
            "schema_version": 1,
            "status": "PARTIAL" if all(gates.values()) else "FAIL",
            "gate": "RUNTIME_READBACK_REPLAY_AUXILIARY",
            "numeric_report": {
                "absolute_path": str(numeric_report_path),
                "sha256": numeric_hash_before,
                "status": numeric_report["status"],
            },
            "stage": {
                "absolute_path": str(stage_path),
                "sha256_before": expected_stage_hash,
                "sha256_after": _sha256(stage_path),
            },
            "trajectory_summary": {
                "name": trajectory["name"],
                "base_translation_drift_m": trajectory[
                    "base_translation_drift_m"
                ],
                "maximum_arm_dof_drift": trajectory[
                    "maximum_arm_dof_drift"
                ],
                "final_readback_m": trajectory["final_readback_m"],
            },
            "captures": captures,
            "screenshot_manifest": manifest,
            "session_only_hidden_visuals": hidden_visuals,
            "gates": gates,
            "visual_model_review": "PENDING_VISUAL_MODEL_REVIEW",
            "scope": {
                "dynamic_drive_gate": "FAIL",
                "collision_contact": "NOT_RUN",
                "bottle_contact_grasp": "NOT_RUN",
                "task8": "NOT_RUN",
            },
        }
        args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.report.resolve().write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"status={report['status']}")
        print(f"report={args.report.resolve()}")
        print(f"raw_root={raw_root}")
        exit_code = 0
    except Exception:
        traceback.print_exc()
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
