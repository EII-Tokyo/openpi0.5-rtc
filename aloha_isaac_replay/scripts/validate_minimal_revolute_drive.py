from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _get_gains
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _get_limits
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _json_safe
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _set_full_state
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _set_full_target


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase26_minimal_revolute_drive_20260718"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n")


def _update(path: Path, payload: dict[str, Any], **updates: Any) -> None:
    payload.update(updates)
    _write_json(path, payload)


def _timeout_handler(signum: int, frame: Any) -> None:
    raise TimeoutError("minimal revolute validation timed out")


def _define_minimal_revolute(stage: Any, *, stiffness: float, damping: float, max_force: float) -> None:
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    UsdGeom.Xform.Define(stage, "/World/Mock")

    def cube(path: str, pos: tuple[float, float, float], scale: tuple[float, float, float], mass: float) -> Any:
        prim = UsdGeom.Cube.Define(stage, path).GetPrim()
        UsdGeom.Cube(prim).CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xform.AddScaleOp().Set(Gf.Vec3d(*scale))
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(float(mass))
        return prim

    base = cube("/World/Mock/base", (0.0, 0.0, 0.0), (0.10, 0.10, 0.10), 1.0)
    link = cube("/World/Mock/link", (0.35, 0.0, 0.0), (0.50, 0.06, 0.06), 0.2)

    root_joint = UsdPhysics.FixedJoint.Define(stage, "/World/Mock/root_joint")
    root_joint.CreateBody1Rel().SetTargets([base.GetPath()])
    UsdPhysics.ArticulationRootAPI.Apply(root_joint.GetPrim())
    try:
        PhysxSchema.PhysxArticulationAPI.Apply(root_joint.GetPrim())
    except Exception:
        pass

    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/World/Mock/joints/hinge")
    hinge.CreateBody0Rel().SetTargets([base.GetPath()])
    hinge.CreateBody1Rel().SetTargets([link.GetPath()])
    hinge.CreateAxisAttr().Set("Z")
    hinge.CreateLocalPos0Attr().Set(Gf.Vec3f(0.10, 0.0, 0.0))
    hinge.CreateLocalPos1Attr().Set(Gf.Vec3f(-0.25, 0.0, 0.0))
    hinge.CreateLowerLimitAttr().Set(-180.0)
    hinge.CreateUpperLimitAttr().Set(180.0)
    drive = UsdPhysics.DriveAPI.Apply(hinge.GetPrim(), "angular")
    drive.CreateStiffnessAttr().Set(float(stiffness))
    drive.CreateDampingAttr().Set(float(damping))
    drive.CreateMaxForceAttr().Set(float(max_force))
    drive.CreateTargetPositionAttr().Set(0.0)
    drive.CreateTargetVelocityAttr().Set(0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a minimal Isaac revolute articulation zero-hold gate.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--physics-dt", type=float, default=1.0 / 50.0)
    parser.add_argument("--stiffness", type=float, default=625.0)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--max-force", type=float, default=10.0)
    parser.add_argument("--timeout-seconds", type=int, default=45)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    json_path = output_dir / "minimal_revolute_drive_metrics.json"
    payload: dict[str, Any] = {
        "status": "STARTED",
        "real_robot_touched": False,
        "stage_saved": False,
        "inputs": vars(args),
        "checkpoints": [],
    }
    _write_json(json_path, payload)

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(max(1, int(args.timeout_seconds)))
    try:
        payload["checkpoints"].append("before_simulation_app")
        _write_json(json_path, payload)

        from isaacsim import SimulationApp

        app_config = dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG)
        app_config["fast_shutdown"] = False
        _app = SimulationApp(app_config)
        payload["checkpoints"].append("after_simulation_app")
        _write_json(json_path, payload)

        import isaacsim.core.utils.stage as stage_utils
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation

        World.clear_instance()
        stage_utils.create_new_stage()
        world = World(stage_units_in_meters=1.0, backend="numpy", device="cpu")
        world.set_simulation_dt(physics_dt=args.physics_dt, rendering_dt=args.physics_dt)
        stage = omni.usd.get_context().get_stage()
        _define_minimal_revolute(stage, stiffness=args.stiffness, damping=args.damping, max_force=args.max_force)
        payload["checkpoints"].append("after_stage_authoring")
        _write_json(json_path, payload)

        art = world.scene.add(SingleArticulation(prim_path="/World/Mock/root_joint", name="minimal_revolute"))
        payload["checkpoints"].append("after_articulation_add")
        _write_json(json_path, payload)

        world.reset()
        world.get_physics_context().set_gravity(0.0)
        payload["checkpoints"].append("after_world_reset")
        _write_json(json_path, payload)

        q0 = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1).copy()
        _set_full_state(art, q0)
        _set_full_target(art, q0)
        rows = []
        for step in range(args.steps):
            _set_full_target(art, q0)
            world.step(render=False)
            qpos = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
            qvel = np.asarray(art.get_joint_velocities(), dtype=np.float64).reshape(-1)
            rows.append(
                {
                    "step": step,
                    "target": float(q0[0]),
                    "qpos": float(qpos[0]),
                    "qvel": float(qvel[0]),
                    "error": float(qpos[0] - q0[0]),
                }
            )

        final_abs_error = abs(float(rows[-1]["error"])) if rows else 0.0
        max_abs_qvel = max((abs(float(row["qvel"])) for row in rows), default=0.0)
        payload.update(
            {
                "status": "PASS" if final_abs_error < 0.01 else "FAILED_GATE",
                "overall_pass": final_abs_error < 0.01,
                "articulation_path": art.prim_path,
                "dof_names": list(art.dof_names),
                "num_dof": int(art.num_dof),
                "limits": _get_limits(art),
                "gains": {"stiffness": _get_gains(art)[0], "damping": _get_gains(art)[1]},
                "initial_qpos": q0,
                "final_abs_error": final_abs_error,
                "max_abs_qvel": max_abs_qvel,
                "rows": rows,
            }
        )
        signal.alarm(0)
        _write_json(json_path, payload)
        print(json.dumps({"status": payload["status"], "json": _rel(json_path)}, ensure_ascii=False), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if payload["overall_pass"] else 3)
    except BaseException as exc:
        signal.alarm(0)
        payload.update(
            {
                "status": "EXCEPTION",
                "exception": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc().splitlines()[-25:],
            }
        )
        _write_json(json_path, payload)
        print(json.dumps({"status": payload["status"], "json": _rel(json_path), "exception": payload["exception"]}, ensure_ascii=False), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    raise SystemExit(main())
