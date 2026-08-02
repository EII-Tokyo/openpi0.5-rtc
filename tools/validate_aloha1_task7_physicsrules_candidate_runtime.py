#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Probe one isolated Task 7 PhysicsRules candidate in a fresh Isaac process."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import traceback
from typing import Any

import numpy as np

from tools.aloha1_mapping.task7_physicsrules_root_cause import summarize_runtime_trace

EXPECTED_DOF_ORDER = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
    "gripper",
    "left_finger",
    "right_finger",
]
FIRST_FRAME_ARM_GATE_RAD = 0.020


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--follower", choices=("follower_left", "follower_right"), required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--repeat-index", type=int, required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _runtime_versions(app: Any) -> dict[str, str]:
    import carb

    manager = app.get_extension_manager()
    physx_id = manager.get_enabled_extension_id("omni.physx")
    record = manager.get_extension_dict(physx_id) if physx_id else {}
    return {
        "isaac_sim": version("isaacsim"),
        "kit": str(carb.tokens.get_tokens_interface().resolve("${kit_version}")).split("+", maxsplit=1)[0],
        "physx": str(record.get("package", {}).get("version", "")).split("+", maxsplit=1)[0],
    }


def _active_collision_paths(stage: Any) -> list[str]:
    from pxr import Usd
    from pxr import UsdPhysics

    return sorted(
        str(prim.GetPath())
        for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
        if prim.IsActive() and prim.HasAPI(UsdPhysics.CollisionAPI)
    )


def _articulation_roots(stage: Any) -> list[str]:
    from pxr import Usd
    from pxr import UsdPhysics

    return sorted(
        str(prim.GetPath())
        for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
        if prim.IsActive() and prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    )


def _sample(articulation: Any, frame: int) -> dict[str, Any]:
    positions = np.asarray(articulation.get_joint_positions(), dtype=np.float64)
    velocities = np.asarray(articulation.get_joint_velocities(), dtype=np.float64)
    return {
        "frame": frame,
        "time_s": frame / 60.0,
        "positions": positions.tolist(),
        "velocities": velocities.tolist(),
        "finite": bool(np.all(np.isfinite(positions)) and np.all(np.isfinite(velocities))),
    }


def _trace_signature(*, dof_names: list[str], collision_paths: list[str], samples: list[dict[str, Any]]) -> str:
    payload = {
        "dof_names": dof_names,
        "collision_paths": collision_paths,
        "samples": [
            {
                "frame": sample["frame"],
                "positions": [round(float(value), 12) for value in sample["positions"]],
                "velocities": [round(float(value), 12) for value in sample["velocities"]],
            }
            for sample in samples
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main(args: argparse.Namespace) -> int:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import open_stage
    import omni.kit.app
    import omni.usd
    from pxr import UsdGeom

    stage_path = args.stage.resolve(strict=True)
    hash_before = _sha256(stage_path)
    if args.frames < 2:
        raise ValueError("--frames must be at least 2")
    if not open_stage(str(stage_path)):
        raise RuntimeError(f"failed to open Stage: {stage_path}")
    app = omni.kit.app.get_app()
    for _ in range(20):
        app.update()
    stage = omni.usd.get_context().get_stage()
    runtime = _runtime_versions(app)
    expected_runtime = {
        "isaac_sim": "5.1.0.0",
        "kit": "107.3.3",
        "physx": "107.3.26",
    }
    if runtime != expected_runtime:
        raise RuntimeError(f"runtime version drift: {runtime}")
    roots = _articulation_roots(stage)
    if len(roots) != 1:
        raise RuntimeError(f"expected exactly one articulation root, got {roots}")
    collision_paths = _active_collision_paths(stage)
    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    world = World(
        stage_units_in_meters=1.0,
        backend="numpy",
        device="cpu",
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
    )
    articulation = SingleArticulation(
        prim_path=roots[0],
        name=f"task7_{args.profile}_{args.follower}_{args.repeat_index}",
        reset_xform_properties=False,
    )
    world.scene.add(articulation)
    world.reset()
    dof_names = list(articulation.dof_names)
    samples = [_sample(articulation, 0)]
    for frame in range(1, args.frames + 1):
        world.step(render=False)
        samples.append(_sample(articulation, frame))
    summary = summarize_runtime_trace(
        dof_names=dof_names,
        expected_dof_names=EXPECTED_DOF_ORDER,
        samples=samples,
        first_frame_arm_gate_rad=FIRST_FRAME_ARM_GATE_RAD,
    )
    signature = _trace_signature(
        dof_names=dof_names,
        collision_paths=collision_paths,
        samples=samples,
    )
    hash_after = _sha256(stage_path)
    if hash_after != hash_before:
        raise RuntimeError("candidate Stage changed during runtime validation")
    report = {
        "schema_version": 1,
        "status": summary["status"],
        "scope": "ISOLATED_ROBOT_LOCAL_CANDIDATE_RUNTIME",
        "profile": args.profile,
        "follower": args.follower,
        "repeat_index": args.repeat_index,
        "runtime": runtime,
        "stage": {
            "absolute_path": str(stage_path),
            "sha256_before": hash_before,
            "sha256_after": hash_after,
            "default_prim": str(stage.GetDefaultPrim().GetPath()),
            "up_axis": up_axis,
            "meters_per_unit": meters_per_unit,
        },
        "articulation_roots": roots,
        "articulation_count": len(roots),
        "dof_order": dof_names,
        "expected_dof_order": EXPECTED_DOF_ORDER,
        "physics_frequency_hz": 60,
        "sample_count": len(samples),
        "samples": samples,
        "summary": summary,
        "collision_count": len(collision_paths),
        "collision_paths": collision_paths,
        "deterministic_signature": signature,
        "uncommanded_probe": True,
        "physics_parameters_modified": False,
        "final_or_default_asset_modified": False,
        "real_robot_connected": False,
        "remote_192_168_1_103_accessed": False,
        "task8": "NOT_RUN",
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "profile": args.profile,
                "follower": args.follower,
                "signature": signature,
                "first_frame_arm_jump": summary["first_frame_arm_jump_max_abs_rad"],
                "static_arm_drift": summary["static_arm_drift_max_abs_rad"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    world.stop()
    return 0 if report["status"] == "PASS" else 2


def run() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "create_new_stage": False,
            "disable_viewport_updates": True,
        }
    )
    exit_code = 1
    try:
        exit_code = main(_parse_args())
    except BaseException:
        traceback.print_exc()
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
