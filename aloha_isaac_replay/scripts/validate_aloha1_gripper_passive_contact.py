from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _apply_arm_gains
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _apply_gravity
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _get_limits
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _json_safe
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _set_full_state
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _set_full_target
from aloha_isaac_replay.scripts.validate_aloha1_gripper_proxy_gap import FINGER_PROXY_PATHS
from aloha_isaac_replay.scripts.validate_aloha1_gripper_proxy_gap import _bbox_row
from aloha_isaac_replay.scripts.validate_aloha1_gripper_proxy_gap import _gap_metrics
from aloha_isaac_replay.scripts.validate_aloha1_native_single_joint_response import _safe_target


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE = REPO_ROOT / "local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_bbox_proxy_runtime.usda"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase43_gripper_passive_contact_20260718"


def _rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "phase",
        "step",
        "object_center_x",
        "object_center_y",
        "object_center_z",
        "object_displacement",
        "left_finger_qpos",
        "right_finger_qpos",
        "finger_center_distance",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 43 Gripper Passive Contact Smoke",
        "",
        f"- status: `{payload['status']}`",
        f"- stage: `{payload['inputs']['stage_usd']}`",
        f"- control mode: `{payload['inputs']['control_mode']}`",
        f"- object side length: `{payload.get('object_side_length_stage_units')}` stage units",
        f"- object displacement: `{payload.get('object_displacement')}` stage units",
        f"- finite object motion: `{payload.get('object_motion_finite')}`",
        f"- contact motion lower bound ok: `{payload.get('contact_motion_ok')}`",
        f"- no explosion upper bound ok: `{payload.get('no_explosion_ok')}`",
        "",
        "## Interpretation",
        "",
        "This is a local contact smoke test. It only checks whether a small passive cube between the gripper proxies remains numerically stable and moves within a bounded range during finger closure.",
        "It does not validate grasp success, bottle geometry, friction realism, or full-arm task behavior.",
    ]
    path.write_text("\n".join(lines) + "\n")


def _finger_targets(art: Any, offset: float, limit_margin: float) -> tuple[np.ndarray, dict[str, float]]:
    dof_names = list(art.dof_names)
    limits = _get_limits(art)
    qpos = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
    target = qpos.copy()
    target_values: dict[str, float] = {}
    for name, sign in [("left_finger", 1.0), ("right_finger", -1.0)]:
        idx = dof_names.index(name)
        lower, upper = [float(x) for x in limits[idx]]
        origin = (lower + upper) * 0.5
        target_value, _clipped = _safe_target(origin, offset * sign, lower, upper, limit_margin)
        target[idx] = target_value
        target_values[name] = target_value
    return target, target_values


def _set_finger_target_and_step(world: Any, art: Any, target: np.ndarray, steps: int) -> None:
    for _ in range(steps):
        _set_full_target(art, target)
        world.step(render=False)


def _surface_gap(left_box: dict[str, Any], right_box: dict[str, Any], axis: int) -> float:
    left_min = float(left_box["min"][axis])
    left_max = float(left_box["max"][axis])
    right_min = float(right_box["min"][axis])
    right_max = float(right_box["max"][axis])
    if left_max <= right_min:
        return right_min - left_max
    if right_max <= left_min:
        return left_min - right_max
    return 0.0


def _create_passive_cube(
    *,
    world: Any,
    stage: Any,
    path: str,
    center: np.ndarray,
    side_length: float,
    mass: float,
    creation_mode: str,
) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics

    if creation_mode == "dynamic_cuboid":
        from isaacsim.core.api.objects import DynamicCuboid

        world.scene.add(
            DynamicCuboid(
                prim_path=path,
                name="phase43_passive_contact_cube",
                position=np.asarray(center, dtype=np.float64),
                scale=np.asarray([side_length, side_length, side_length], dtype=np.float64),
                size=1.0,
                mass=float(mass),
                color=np.asarray([0.9, 0.2, 0.1], dtype=np.float64),
            )
        )
        return
    if creation_mode != "raw_usd":
        raise ValueError(f"Unsupported object creation mode: {creation_mode}")
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.2, 0.1)])
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*[float(x) for x in center]))
    xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(side_length, side_length, side_length))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr().Set(True)
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(float(mass))


def _set_collision_offsets(stage: Any, prim_path: str, contact_offset: float | None, rest_offset: float | None) -> dict[str, Any]:
    from pxr import PhysxSchema

    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        return {"path": prim_path, "exists": False, "applied": False}
    author_offsets = contact_offset is not None or rest_offset is not None
    api = PhysxSchema.PhysxCollisionAPI.Apply(prim) if author_offsets else PhysxSchema.PhysxCollisionAPI(prim)
    if contact_offset is not None:
        api.CreateContactOffsetAttr(float(contact_offset)).Set(float(contact_offset))
    if rest_offset is not None:
        api.CreateRestOffsetAttr(float(rest_offset)).Set(float(rest_offset))
    return {
        "path": prim_path,
        "exists": True,
        "applied": author_offsets,
        "contact_offset": api.GetContactOffsetAttr().Get() if api.GetContactOffsetAttr() else None,
        "rest_offset": api.GetRestOffsetAttr().Get() if api.GetRestOffsetAttr() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local passive-object contact smoke test for ALOHA1 gripper proxies.")
    parser.add_argument("--stage-usd", default=str(DEFAULT_STAGE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--open-offset", type=float, default=0.006)
    parser.add_argument("--close-offset", type=float, default=-0.006)
    parser.add_argument("--settle-steps", type=int, default=60)
    parser.add_argument("--close-steps", type=int, default=180)
    parser.add_argument("--physics-dt", type=float, default=1.0 / 50.0)
    parser.add_argument("--gravity", type=float, default=0.0)
    parser.add_argument("--limit-margin", type=float, default=0.001)
    parser.add_argument("--object-fill-fraction", type=float, default=0.6)
    parser.add_argument("--object-creation", choices=("dynamic_cuboid", "raw_usd"), default="dynamic_cuboid")
    parser.add_argument("--object-mass", type=float, default=0.01)
    parser.add_argument("--object-contact-offset", type=float, default=None)
    parser.add_argument("--object-rest-offset", type=float, default=None)
    parser.add_argument("--proxy-contact-offset", type=float, default=None)
    parser.add_argument("--proxy-rest-offset", type=float, default=None)
    parser.add_argument("--min-contact-motion", type=float, default=1e-5)
    parser.add_argument("--max-object-displacement", type=float, default=0.25)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    json_path = output_dir / "gripper_passive_contact_metrics.json"
    csv_path = output_dir / "gripper_passive_contact_timeseries.csv"
    md_path = output_dir / "gripper_passive_contact_metrics.md"
    payload: dict[str, Any] = {
        "status": "STARTED",
        "overall_pass": False,
        "real_robot_touched": False,
        "stage_saved": False,
        "inputs": {
            "stage_usd": _rel(args.stage_usd),
            "side": args.side,
            "control_mode": "opposed_fingers",
            "open_offset": args.open_offset,
            "close_offset": args.close_offset,
            "settle_steps": args.settle_steps,
            "close_steps": args.close_steps,
            "physics_dt": args.physics_dt,
            "gravity": args.gravity,
            "object_fill_fraction": args.object_fill_fraction,
            "object_creation": args.object_creation,
            "object_contact_offset": args.object_contact_offset,
            "object_rest_offset": args.object_rest_offset,
            "proxy_contact_offset": args.proxy_contact_offset,
            "proxy_rest_offset": args.proxy_rest_offset,
            "min_contact_motion": args.min_contact_motion,
            "max_object_displacement": args.max_object_displacement,
        },
        "outputs": {"json": _rel(json_path), "csv": _rel(csv_path), "markdown": _rel(md_path)},
    }
    _write_json(json_path, payload)

    try:
        from isaacsim import SimulationApp

        app_config = dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG)
        app_config["fast_shutdown"] = False
        _app = SimulationApp(app_config)
        import isaacsim.core.utils.stage as stage_utils
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        import omni.usd

        stage_utils.open_stage(str(Path(args.stage_usd).resolve()))
        World.clear_instance()
        world = World(stage_units_in_meters=0.01, backend="numpy", device="cpu")
        world.set_simulation_dt(physics_dt=args.physics_dt, rendering_dt=args.physics_dt)
        stage = omni.usd.get_context().get_stage()
        paths = FINGER_PROXY_PATHS[args.side]
        art = world.scene.add(SingleArticulation(prim_path=paths["articulation"], name=f"{args.side}_vx300s"))
        world.reset()
        _apply_gravity(world, args.gravity)
        _apply_arm_gains(art, None, None)

        open_target, open_values = _finger_targets(art, args.open_offset, args.limit_margin)
        _set_full_state(art, open_target)
        _set_full_target(art, open_target)
        _set_finger_target_and_step(world, art, open_target, args.settle_steps)

        left_box = _bbox_row(stage, paths["left_finger"])
        right_box = _bbox_row(stage, paths["right_finger"])
        gap = _gap_metrics(left_box, right_box)
        if not gap.get("bbox_pair_valid"):
            raise RuntimeError("Finger proxy bbox pair is invalid; cannot place contact object.")
        axis_name = str(gap["dominant_axis"])
        axis = {"x": 0, "y": 1, "z": 2}[axis_name]
        center = (np.asarray(left_box["center"], dtype=np.float64) + np.asarray(right_box["center"], dtype=np.float64)) * 0.5
        surface_gap = _surface_gap(left_box, right_box, axis)
        side_length = max(surface_gap * args.object_fill_fraction, 1e-4)
        object_path = "/World/phase43_passive_contact_cube"
        proxy_offset_rows = [
            _set_collision_offsets(stage, paths["left_finger"], args.proxy_contact_offset, args.proxy_rest_offset),
            _set_collision_offsets(stage, paths["right_finger"], args.proxy_contact_offset, args.proxy_rest_offset),
        ]
        _create_passive_cube(
            world=world,
            stage=stage,
            path=object_path,
            center=center,
            side_length=side_length,
            mass=args.object_mass,
            creation_mode=args.object_creation,
        )
        object_offset_row = _set_collision_offsets(stage, object_path, args.object_contact_offset, args.object_rest_offset)
        world.reset()
        _apply_gravity(world, args.gravity)
        _set_full_state(art, open_target)
        _set_full_target(art, open_target)
        _set_finger_target_and_step(world, art, open_target, args.settle_steps)

        object_initial_box = _bbox_row(stage, object_path)
        object_initial_center = np.asarray(object_initial_box["center"], dtype=np.float64)
        close_target, close_values = _finger_targets(art, args.close_offset, args.limit_margin)
        rows: list[dict[str, Any]] = []
        max_displacement = 0.0
        finite_motion = True
        for step in range(args.close_steps):
            _set_full_target(art, close_target)
            world.step(render=False)
            qpos = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
            dof_names = list(art.dof_names)
            left_box = _bbox_row(stage, paths["left_finger"])
            right_box = _bbox_row(stage, paths["right_finger"])
            object_box = _bbox_row(stage, object_path)
            object_center = np.asarray(object_box.get("center", [np.nan, np.nan, np.nan]), dtype=np.float64)
            displacement = float(np.linalg.norm(object_center - object_initial_center))
            finite_motion = bool(finite_motion and np.all(np.isfinite(object_center)) and np.isfinite(displacement))
            max_displacement = max(max_displacement, displacement if np.isfinite(displacement) else float("inf"))
            rows.append(
                {
                    "phase": "close",
                    "step": step,
                    "object_center_x": float(object_center[0]),
                    "object_center_y": float(object_center[1]),
                    "object_center_z": float(object_center[2]),
                    "object_displacement": displacement,
                    "left_finger_qpos": float(qpos[dof_names.index("left_finger")]),
                    "right_finger_qpos": float(qpos[dof_names.index("right_finger")]),
                    "finger_center_distance": _gap_metrics(left_box, right_box).get("center_distance"),
                }
            )

        object_final_box = _bbox_row(stage, object_path)
        object_final_center = np.asarray(object_final_box["center"], dtype=np.float64)
        object_displacement = float(np.linalg.norm(object_final_center - object_initial_center))
        contact_motion_ok = bool(object_displacement >= args.min_contact_motion)
        no_explosion_ok = bool(finite_motion and max_displacement <= args.max_object_displacement)
        overall_pass = bool(contact_motion_ok and no_explosion_ok)
        payload.update(
            {
                "status": "PASS" if overall_pass else "FAILED_GATE",
                "overall_pass": overall_pass,
                "open_target_values": open_values,
                "close_target_values": close_values,
                "finger_gap_axis": axis_name,
                "finger_surface_gap_open": surface_gap,
                "object_path": object_path,
                "object_side_length_stage_units": side_length,
                "proxy_collision_offsets": proxy_offset_rows,
                "object_collision_offsets": object_offset_row,
                "object_initial_center": object_initial_center.tolist(),
                "object_final_center": object_final_center.tolist(),
                "object_displacement": object_displacement,
                "max_object_displacement": max_displacement,
                "object_motion_finite": finite_motion,
                "contact_motion_ok": contact_motion_ok,
                "no_explosion_ok": no_explosion_ok,
                "csv": _rel(csv_path),
                "markdown": _rel(md_path),
                "next_gate": "gripper_contact_with_task_shape" if overall_pass else "inspect_contact_geometry_or_finger_control",
            }
        )
        _write_csv(csv_path, rows)
        _write_json(json_path, payload)
        _write_markdown(md_path, _json_safe(payload))
        print(json.dumps({"status": payload["status"], "json": _rel(json_path), "markdown": _rel(md_path)}, ensure_ascii=False), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if overall_pass else 3)
    except BaseException as exc:
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
