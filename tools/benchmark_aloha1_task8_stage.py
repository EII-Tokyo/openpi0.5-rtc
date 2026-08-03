#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np

from tools.aloha1_mapping.task8_optimization import summarize_numeric_samples
from tools.audit_aloha1_task8_baseline import start_usd_runtime_if_needed

ROOT = Path(__file__).resolve().parents[1]
WORKLOAD_SOURCE = ROOT / (
    "reports/aloha1_mapping/"
    "aloha1_grasp_20cm_five_pose_cad_collision_replan_preflight.json"
)
ARM_JOINT_ORDER = (
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _measurement_dict(data: Any) -> dict[str, dict[str, Any]]:
    return {
        str(item.name): {
            "value": item.value,
            "unit": str(getattr(item, "unit", "")),
        }
        for item in data.measurements
    }


def align_physics_samples(
    app_samples: list[float], physics_samples_raw: list[float]
) -> tuple[list[float], int]:
    """Drop only the leading PhysX profile history not paired with app updates."""

    history_count = max(0, len(physics_samples_raw) - len(app_samples))
    aligned = physics_samples_raw[-len(app_samples) :] if app_samples else []
    return aligned, history_count


def select_target_waypoint(
    sequence: list[list[float]], frame_index: int
) -> list[float]:
    if not sequence:
        raise ValueError("benchmark target sequence is empty")
    return sequence[min(frame_index, len(sequence) - 1)]


def _load_accepted_workload() -> tuple[list[list[float]], dict[str, Any]]:
    payload = json.loads(WORKLOAD_SOURCE.read_text(encoding="utf-8"))
    sample = payload["selected_samples"][0]
    if sample["sample_id"] != "sample_01":
        raise RuntimeError("accepted Task 8 workload sample order drift")
    sequence = [
        [float(value) for value in sample["initial_arm_q_rad"]],
        *[
            [float(value) for value in waypoint["joint_positions_rad"]]
            for waypoint in sample["ik"]["waypoints"]
        ],
    ]
    if any(len(values) != len(ARM_JOINT_ORDER) for values in sequence):
        raise RuntimeError("accepted Task 8 workload DOF order drift")
    return sequence, {
        "source": str(WORKLOAD_SOURCE.resolve()),
        "source_sha256": _sha256(WORKLOAD_SOURCE),
        "sample_id": sample["sample_id"],
        "source_waypoint_count": len(sequence),
        "joint_order": list(ARM_JOINT_ORDER),
        "provenance": "FROZEN_ACCEPTED_TASK7_809_WAYPOINT_SWEEP_INPUT",
    }


def _apply_workload_target(
    articulations: list[Any], sequence: list[list[float]], frame_index: int
) -> None:
    from isaacsim.core.utils.types import ArticulationAction

    target = select_target_waypoint(sequence, frame_index)
    action = ArticulationAction(
        joint_positions=np.asarray(target, dtype=np.float64),
        joint_indices=np.arange(len(ARM_JOINT_ORDER), dtype=np.int64),
    )
    for articulation in articulations:
        articulation.get_articulation_controller().apply_action(action)


def benchmark(
    *,
    app: Any,
    stage_path: Path,
    expected_sha256: str | None,
    profile: str,
    environment_count: int,
    warmup_frames: int,
    measured_frames: int,
) -> dict[str, Any]:
    import omni.kit.app
    import omni.timeline
    import omni.usd
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    extension_manager = omni.kit.app.get_app().get_extension_manager()
    extension_manager.set_extension_enabled_immediate(
        "isaacsim.benchmark.services", True  # noqa: FBT003 - C++ binding is positional
    )
    app.update()

    from isaacsim.benchmark.services.base_isaac_benchmark import set_sync_mode
    from isaacsim.benchmark.services.datarecorders.interface import InputContext
    from isaacsim.benchmark.services.datarecorders.memory import MemoryRecorder
    from isaacsim.benchmark.services.recorders import IsaacFrameTimeRecorder
    from isaacsim.benchmark.services.utils import wait_until_stage_is_fully_loaded
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import open_stage

    stage_path = stage_path.resolve(strict=True)
    stage_hash = _sha256(stage_path)
    set_sync_mode()
    load_start = time.perf_counter_ns()
    open_stage(str(stage_path))
    wait_until_stage_is_fully_loaded()
    load_ms = (time.perf_counter_ns() - load_start) / 1_000_000.0

    stage = omni.usd.get_context().get_stage()
    default_prim = stage.GetDefaultPrim() if stage else None
    collision_prims = []
    collision_mesh_points = 0
    collision_mesh_faces = 0
    articulation_count = 0
    joint_count = 0
    rigid_body_count = 0
    robot_articulation_paths = []
    if stage is not None:
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_count += 1
                if str(prim.GetPath()).endswith("/root_joint"):
                    robot_articulation_paths.append(str(prim.GetPath()))
            if prim.IsA(UsdPhysics.Joint):
                joint_count += 1
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_body_count += 1
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_prims.append(str(prim.GetPath()))
                if prim.IsA(UsdGeom.Mesh):
                    mesh = UsdGeom.Mesh(prim)
                    collision_mesh_points += len(mesh.GetPointsAttr().Get() or [])
                    collision_mesh_faces += len(
                        mesh.GetFaceVertexCountsAttr().Get() or []
                    )
    target_sequence, workload = _load_accepted_workload()
    robot_articulation_paths.sort()
    if len(robot_articulation_paths) != 2 * environment_count:
        raise RuntimeError(
            "benchmark robot articulation count mismatch: "
            f"{len(robot_articulation_paths)} != {2 * environment_count}"
        )
    world = World(
        stage_units_in_meters=1.0,
        backend="numpy",
        device="cpu",
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
    )
    articulations = []
    for index, articulation_path in enumerate(robot_articulation_paths):
        articulation = SingleArticulation(
            prim_path=articulation_path,
            name=f"task8_benchmark_articulation_{index:03d}",
            reset_xform_properties=False,
        )
        world.scene.add(articulation)
        articulations.append(articulation)
    world.reset()
    for articulation in articulations:
        if list(articulation.dof_names[:6]) != list(ARM_JOINT_ORDER):
            raise RuntimeError(f"benchmark DOF order drift: {articulation.dof_names}")
    workload["robot_articulation_paths"] = robot_articulation_paths
    workload["runtime_articulation_count"] = len(articulations)
    workload["runtime_dof_names"] = list(articulations[0].dof_names)
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    context = InputContext(
        artifact_prefix="aloha1_task8",
        kit_version="107.3.3",
        phase="fixed_frame_playing",
    )
    recorder = IsaacFrameTimeRecorder(context, gpu_frametime=True)
    memory_before = _measurement_dict(MemoryRecorder().get_data())
    workload_frame = 0
    for _ in range(warmup_frames):
        _apply_workload_target(articulations, target_sequence, workload_frame)
        workload_frame += 1
        app.update()
    readback_after_warmup = [
        np.asarray(item.get_joint_positions(), dtype=np.float64).tolist()
        for item in articulations
    ]
    recorder.start_collecting()
    wall_start = time.perf_counter_ns()
    for _ in range(measured_frames):
        _apply_workload_target(articulations, target_sequence, workload_frame)
        workload_frame += 1
        app.update()
    readback_after_measurement = [
        np.asarray(item.get_joint_positions(), dtype=np.float64).tolist()
        for item in articulations
    ]
    measured_wall_ms = (time.perf_counter_ns() - wall_start) / 1_000_000.0
    recorder.stop_collecting()
    timeline.pause()
    frame_metrics = _measurement_dict(recorder.get_data())
    memory_after = _measurement_dict(MemoryRecorder().get_data())

    app_samples = frame_metrics.get("App_Update Frametime Samples", {}).get("value", [])
    physics_samples_raw = frame_metrics.get("Physics Frametime Samples", {}).get(
        "value", []
    )
    physics_samples, physics_history_count = align_physics_samples(
        app_samples, physics_samples_raw
    )
    readback_motion_max_abs_rad = max(
        float(
            np.max(
                np.abs(np.asarray(after[:6]) - np.asarray(before[:6]))
            )
        )
        for before, after in zip(
            readback_after_warmup, readback_after_measurement, strict=True
        )
    )
    status = (
        "PASS"
        if (expected_sha256 is None or expected_sha256 == stage_hash)
        and stage is not None
        and default_prim
        and len(app_samples) > 0
        and readback_motion_max_abs_rad > 0.0
        and _sha256(stage_path) == stage_hash
        else "FAIL"
    )
    return {
        "schema_version": 1,
        "status": status,
        "classification": "TASK8_FRESH_PROCESS_STAGE_BENCHMARK",
        "profile": profile,
        "environment_count": environment_count,
        "stage": {
            "absolute_path": str(stage_path),
            "sha256": stage_hash,
            "expected_sha256": expected_sha256,
            "default_prim": str(default_prim.GetPath()) if default_prim else None,
        },
        "runtime": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
            "benchmark_source": "isaacsim.benchmark.services 5.1 local installation",
            "warmup_frames": warmup_frames,
            "warmup_timeline_playing": True,
            "measured_frames": measured_frames,
            "timeline_playing_during_measurement": True,
        },
        "workload": {
            **workload,
            "warmup_target_frames": warmup_frames,
            "measured_target_frames": measured_frames,
            "total_target_frames": workload_frame,
            "first_target_arm_q_rad": target_sequence[0],
            "last_measured_target_arm_q_rad": select_target_waypoint(
                target_sequence, workload_frame - 1
            ),
            "joint_readback_after_warmup_rad": readback_after_warmup,
            "joint_readback_after_measurement_rad": readback_after_measurement,
            "readback_motion_max_abs_rad": readback_motion_max_abs_rad,
            "runtime_articulation_action_only": True,
            "usd_drive_target_authored_per_frame": False,
            "source_stage_sha256_after": _sha256(stage_path),
            "source_stage_unchanged": _sha256(stage_path) == stage_hash,
        },
        "inventory": {
            "articulation_count": articulation_count,
            "joint_count": joint_count,
            "rigid_body_count": rigid_body_count,
            "collider_prim_count": len(collision_prims),
            "collision_mesh_point_count": collision_mesh_points,
            "collision_mesh_face_count": collision_mesh_faces,
            "upper_arm_collider_prim_count": sum(
                "_upper_arm_link/cad_derived_collisions/cad_derived_upper_arm_link/"
                in path
                for path in collision_prims
            ),
        },
        "metrics": {
            "stage_load_ms": load_ms,
            "measured_wall_ms": measured_wall_ms,
            "app_frame_sample_count": len(app_samples),
            "physics_frame_sample_count": len(physics_samples),
            "physics_frame_sample_count_raw": len(physics_samples_raw),
            "physics_history_sample_count_excluded": physics_history_count,
            "physics_sample_alignment": (
                "KEEP_LAST_N_PHYSX_PROFILE_SAMPLES_MATCHING_N_APP_UPDATE_SAMPLES; "
                "local 5.1 PhysX benchmark subscription returns pre-subscription "
                "warmup history in its first profile-stats callback"
            ),
            "app_update_ms_summary": summarize_numeric_samples(app_samples),
            "physics_step_ms_summary": summarize_numeric_samples(physics_samples),
            "official_frame_recorder": frame_metrics,
            "memory_before": memory_before,
            "memory_after": memory_after,
        },
        "boundaries": {
            "physics_parameters_modified": False,
            "final_or_default_asset_modified": False,
            "grasp_acceptance_test": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument(
        "--profile", choices=("fidelity_profile", "throughput_profile"), required=True
    )
    parser.add_argument("--environment-count", type=int, required=True)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--measured-frames", type=int, default=180)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = start_usd_runtime_if_needed()
    result = 1
    try:
        report = benchmark(
            app=app,
            stage_path=args.stage,
            expected_sha256=args.expected_sha256,
            profile=args.profile,
            environment_count=args.environment_count,
            warmup_frames=args.warmup_frames,
            measured_frames=args.measured_frames,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "stage": report["stage"]["absolute_path"],
                    "load_ms": report["metrics"]["stage_load_ms"],
                    "app_samples": report["metrics"]["app_frame_sample_count"],
                    "physics_samples": report["metrics"]["physics_frame_sample_count"],
                    "output": str(args.output.resolve()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        result = 0 if report["status"] == "PASS" else 1
    except Exception:
        print("TASK8_BENCHMARK_EXCEPTION", flush=True)
        traceback.print_exc()
    finally:
        if app is not None:
            app.close()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
