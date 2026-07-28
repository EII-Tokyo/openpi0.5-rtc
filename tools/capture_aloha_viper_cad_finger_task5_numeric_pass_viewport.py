#!/usr/bin/env python3
"""Capture Isaac 5.1 viewport evidence for the numeric-pass Task 5 replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback
from typing import Any

import numpy as np

from tools.aloha1_mapping.screenshot_manifest import build_screenshot_manifest
from tools.aloha1_mapping.screenshot_manifest import validate_screenshot

ROOT = Path(__file__).resolve().parents[1]
PHASE = "task5_numeric_pass_runtime_readback_replay"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_trace_frames(
    trace: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select distinct open, partial, and closed runtime readback records."""

    if len(trace) < 3:
        raise ValueError("symmetric-close trace needs at least three records")
    indices = (0, len(trace) // 2, len(trace) - 1)
    phases = (
        "open_maximum_legal_aperture",
        "partially_closed",
        "closed",
    )
    return [
        {"phase": phase, "record": trace[index]}
        for phase, index in zip(phases, indices, strict=True)
    ]


def _capture_viewport_png(
    app: Any,
    viewport: Any,
    destination: Path,
) -> None:
    from omni.kit.viewport.utility import capture_viewport_to_file

    capture_helper = capture_viewport_to_file(
        viewport,
        file_path=str(destination),
    )
    previous_size = -1
    stable_updates = 0
    for _ in range(300):
        app.update()
        if not destination.exists():
            continue
        size = destination.stat().st_size
        if size > 0 and size == previous_size:
            stable_updates += 1
        else:
            stable_updates = 0
        previous_size = size
        if stable_updates >= 2:
            break
    if not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError("viewport capture did not create a nonempty PNG")
    del capture_helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numeric-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    numeric_path = args.numeric_report.resolve(strict=True)
    numeric_hash = _sha256(numeric_path)
    numeric = json.loads(numeric_path.read_text(encoding="utf-8"))
    if numeric["status"] != "PASS":
        raise RuntimeError("numeric input must be the isolated PASS profile")
    if numeric["scope"]["bottle_contact_grasp"] != "NOT_RUN":
        raise RuntimeError("numeric input unexpectedly contains bottle work")
    symmetric = next(
        item
        for item in numeric["trajectories"]
        if item["name"] == "symmetric_close"
    )
    replay_frames = select_trace_frames(symmetric["trace"])
    stage_path = Path(
        numeric["stage"]["absolute_path"]
    ).resolve(strict=True)
    stage_hash = numeric["stage"]["sha256_after"]
    if _sha256(stage_path) != stage_hash:
        raise RuntimeError("diagnostic Stage hash mismatch")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    raw_root = output_root / "screenshots_raw"
    raw_root.mkdir(parents=True)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "width": 1280, "height": 900})
    exit_code = 1
    try:
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

        from tools.validate_aloha_viper_cad_finger_task5_structure import ARTICULATION_PATH
        from tools.validate_aloha_viper_cad_finger_task5_structure import FINGER_MESHES
        from tools.validate_aloha_viper_cad_finger_task5_structure import _hide_non_target_visuals
        from tools.validate_aloha_viper_cad_finger_task5_structure import _set_view_visibility
        from tools.validate_aloha_viper_cad_finger_task5_structure import _world_points

        if not open_stage(str(stage_path)):
            raise RuntimeError(f"failed to open {stage_path}")
        stage = get_current_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            hidden_visuals = _hide_non_target_visuals(stage)
            dome = UsdLux.DomeLight.Define(
                stage, "/workcell/Task5NumericPassReplay/Dome"
            )
            dome.CreateIntensityAttr(700.0)
            dome.CreateColorAttr(Gf.Vec3f(0.9, 0.92, 1.0))
            key = UsdLux.DistantLight.Define(
                stage, "/workcell/Task5NumericPassReplay/Key"
            )
            key.CreateIntensityAttr(1100.0)

        world = World(
            stage_units_in_meters=1.0,
            backend="numpy",
            device="cpu",
            physics_dt=1.0 / 60.0,
            rendering_dt=1.0 / 60.0,
        )
        articulation = SingleArticulation(
            prim_path=ARTICULATION_PATH,
            name="task5_numeric_pass_replay",
            reset_xform_properties=False,
        )
        world.scene.add(articulation)
        camera = Camera(
            prim_path="/workcell/Task5NumericPassReplay/Camera",
            name="task5_numeric_pass_replay_camera",
            resolution=(1280, 900),
            frequency=60,
        )
        world.scene.add(camera)
        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("no active Isaac viewport")
        structure = json.loads(
            (
                ROOT
                / "reports/aloha1_mapping/"
                "aloha_viper_cad_finger_task5_structure.json"
            ).read_text(encoding="utf-8")
        )
        source_camera = structure["camera_poses"]["base_oblique"]
        source_target = np.asarray(source_camera["target_world_m"])
        source_position = np.asarray(source_camera["position_world_m"])
        fixed_camera_position: np.ndarray | None = None
        fixed_camera_target: np.ndarray | None = None
        captures = []
        for selected in replay_frames:
            record = selected["record"]
            world.reset()
            world.pause()
            camera.set_clipping_range(0.01, 10.0)
            qpos = np.asarray(
                record["all_dof_readback"],
                dtype=np.float32,
            )
            articulation.set_joint_positions(qpos)
            get_physx_interface().update_transformations(
                True, True, False, False  # noqa: FBT003
            )
            _set_view_visibility(stage, "base_oblique")
            points = np.concatenate(
                [
                    _world_points(stage, FINGER_MESHES[side])
                    for side in ("left", "right")
                ]
            )
            runtime_finger_center_world_m = np.mean(points, axis=0)
            if fixed_camera_position is None:
                fixed_camera_target = runtime_finger_center_world_m
                fixed_camera_position = (
                    fixed_camera_target + source_position - source_target
                )
            camera.set_world_pose(
                position=fixed_camera_position,
                orientation=np.asarray(
                    source_camera["orientation_wxyz"]
                ),
                camera_axes="usd",
            )
            viewport.camera_path = Sdf.Path(camera.prim_path)
            for _ in range(20):
                app.update()
            capture_name = (
                f"{selected['phase']}_frame_{int(record['frame']):04d}"
            )
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
                        "view": "base_oblique",
                        "resolution": [1280, 900],
                        "capture_backend": (
                            "omni.kit.viewport.utility."
                            "capture_viewport_to_file"
                        ),
                        "fixed_camera_for_all_phases": True,
                        "position_world_m": np.asarray(
                            actual_position
                        ).tolist(),
                        "orientation_wxyz": np.asarray(
                            actual_orientation
                        ).tolist(),
                        "target_world_m": np.asarray(
                            fixed_camera_target
                        ).tolist(),
                    },
                    simulation={
                        "isaac_sim": "5.1.0.0",
                        "kit": "107.3.3",
                        "physx": "107.3.26",
                        "stage_absolute_path": str(stage_path),
                        "stage_sha256": stage_hash,
                        "source_numeric_report": str(numeric_path),
                        "source_numeric_report_sha256": numeric_hash,
                        "diagnostic_profile": numeric["profile"],
                        "source_trajectory": symmetric["name"],
                        "phase": selected["phase"],
                        "source_runtime_frame": int(record["frame"]),
                        "source_runtime_time_s": float(record["time_s"]),
                        "command_left_m": float(record["command_left_m"]),
                        "command_right_m": float(
                            record["command_right_m"]
                        ),
                        "readback_left_m": float(
                            record["readback_left_m"]
                        ),
                        "readback_right_m": float(
                            record["readback_right_m"]
                        ),
                        "all_dof_readback": record["all_dof_readback"],
                        "runtime_finger_center_world_m": (
                            runtime_finger_center_world_m.tolist()
                        ),
                        "capture_method": (
                            "RUNTIME_READBACK_REPLAY_AUXILIARY"
                        ),
                        "capture_physics_step": False,
                        "bottle_present": False,
                        "acceptance_boundary": (
                            "AUXILIARY QPOS REPLAY OF NUMERIC-PASS TRACE; "
                            "NOT SAME-FRAME CONTACT OR GRASP EVIDENCE"
                        ),
                    },
                )
            )
        required = [
            item["capture_name"] for item in captures
        ]
        manifest = build_screenshot_manifest(
            captures=captures,
            required_captures={PHASE: required},
            artifact_root=output_root,
        )
        gates = {
            "raw_screenshot_acquisition": manifest["status"] == "PASS",
            "three_distinct_runtime_readbacks_replayed": len(captures) == 3,
            "fixed_camera_for_all_phases": len(
                {
                    tuple(item["camera"]["position_world_m"])
                    for item in captures
                }
            )
            == 1,
            "source_stage_immutable": _sha256(stage_path) == stage_hash,
            "numeric_report_immutable": _sha256(numeric_path) == numeric_hash,
            "no_bottle": True,
            "no_physics_step_in_replay": True,
        }
        report = {
            "schema_version": 1,
            "status": "PARTIAL" if all(gates.values()) else "FAIL",
            "gate": "RUNTIME_READBACK_REPLAY_AUXILIARY",
            "numeric_structure_gate": "PASS",
            "captures": captures,
            "screenshot_manifest": manifest,
            "gates": gates,
            "session_only_hidden_visuals": hidden_visuals,
            "visual_model_review": "PENDING_VISUAL_MODEL_REVIEW",
            "scope": {
                "dynamic_drive_gate": "PASS_NUMERIC_ONLY",
                "collision_contact": "NOT_RUN",
                "bottle_contact_grasp": "NOT_RUN",
                "task7": "NOT_RUN",
                "task8": "NOT_RUN",
                "default_or_final_asset_modified": False,
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
