#!/usr/bin/env python3
"""Headless end-to-end validation for the episode-0 kinematic replay."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[5]
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, default=repository / "remote_isaac_assets/aloha1_bottle_server/attempt1/remote_stream_cap_stage.usda")
    parser.add_argument("--bundle-dir", type=Path, default=repository / "remote_isaac_assets/aloha1_bottle_server/attempt1/replays/episode_0")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def load_core(path: Path):
    spec = importlib.util.spec_from_file_location("aloha_episode0_replay_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load replay core: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "create_new_stage": False, "disable_viewport_updates": True, "multi_gpu": False, "limit_cpu_threads": 8})
    report = {"status": "FAIL", "classification": "KINEMATIC_VISUAL_REPLAY_NOT_PHYSICS_ACCEPTANCE"}
    try:
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
        from isaacsim.core.utils.stage import open_stage
        from omni.physx import get_physx_interface
        from pxr import Sdf, Usd

        core = load_core(args.bundle_dir / "episode0_replay_core.py")
        manifest, payload = core.load_bundle(args.bundle_dir)
        stage_hash_before = core.sha256(args.stage.resolve())
        if not open_stage(str(args.stage.resolve())):
            raise RuntimeError("stage open failed")
        for _ in range(30):
            app.update()
        stage = omni.usd.get_context().get_stage()
        layer = Sdf.Layer.CreateAnonymous("episode_0_headless_validation.usda")
        stage.GetSessionLayer().subLayerPaths.append(layer.identifier)
        stage.SetEditTarget(Usd.EditTarget(layer))
        core.prepare_kinematic_objects(stage)
        world = World(stage_units_in_meters=1.0, backend="numpy", device="cpu", physics_dt=1.0 / 50.0, rendering_dt=1.0 / 50.0)
        left = world.scene.add(SingleArticulation("/World/follower_left/vx300s_left/root_joint", name="episode0_validation_left", reset_xform_properties=False))
        right = world.scene.add(SingleArticulation("/World/follower_right/vx300s_right/root_joint", name="episode0_validation_right", reset_xform_properties=False))
        world.reset()
        # Do not register Kinematic replay objects with World.scene: its reset
        # path writes Dynamic-body default velocities and PhysX rejects those
        # writes for Kinematic bodies. This mirrors the Script Editor entry.
        bottle = SingleRigidPrim(core.BOTTLE, name="episode0_validation_bottle", reset_xform_properties=False)
        cap = SingleRigidPrim(core.CAP, name="episode0_validation_cap", reset_xform_properties=False)
        bottle.initialize()
        cap.initialize()
        runner = core.Episode0Replay(stage, left, right, bottle, cap, payload)
        runner.reset()
        max_active_error = 0.0
        state_counts: dict[str, int] = {}
        finite_object_poses = True
        for frame in range(918):
            runner.apply_robot_frame(frame)
            get_physx_interface().update_transformations(True, True, False, False)
            runner.apply_objects_and_metadata(frame)
            for articulation, side in ((left, "left"), (right, "right")):
                actual = np.asarray(articulation.get_joint_positions(), dtype=np.float64)[core.ACTIVE_DOF_INDICES]
                expected = runner.commanded_active_positions(frame, side)
                max_active_error = max(max_active_error, float(np.max(np.abs(actual - expected))))
            finite_object_poses &= bool(np.all(np.isfinite(runner.last_bottle)) and np.all(np.isfinite(runner.last_cap)))
            state = str(payload["state"][frame])
            state_counts[state] = state_counts.get(state, 0) + 1

        checks = {
            "all_918_frames_applied": runner.frames_applied == 918,
            "frequency_is_50_hz": float(payload["frequency_hz"]) == 50.0,
            "active_joint_readback_exact": max_active_error <= 1.0e-5,
            "object_transforms_finite": finite_object_poses,
            "attach_transitions_continuous": runner.max_attach_jump_m <= 1.0e-8,
            "manual_label_coverage_exact": state_counts == {"PICK_UP": 174, "UNSCREW_CAP": 476, "DISPOSE": 150, "RETURN": 118},
            "session_layer_only": core.sha256(args.stage.resolve()) == stage_hash_before,
            "no_ros": manifest["safety"]["uses_ros"] is False,
            "no_real_robot": manifest["safety"]["touches_real_robot"] is False,
        }
        report.update({
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "metrics": {"max_active_joint_readback_error": max_active_error, "max_attach_jump_m": runner.max_attach_jump_m, "state_frame_counts": state_counts},
            "stage": str(args.stage.resolve()),
            "payload_sha256": manifest["payload"]["sha256"],
            "stage_sha256_before_after": stage_hash_before,
            "stage_saved": False,
        })
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    finally:
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        app.close()
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
