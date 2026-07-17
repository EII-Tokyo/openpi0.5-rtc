from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _apply_gravity
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _apply_arm_gains
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _json_safe
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _set_full_state
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _set_full_target
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _create_passive_cube
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _finger_targets
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _set_finger_target_and_step
from aloha_isaac_replay.scripts.validate_aloha1_gripper_proxy_gap import FINGER_PROXY_PATHS
from aloha_isaac_replay.scripts.validate_aloha1_gripper_proxy_gap import _bbox_row


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE = REPO_ROOT / "local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_bbox_proxy_runtime.usda"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase44_gripper_contact_runtime_20260718"


def _rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n")


def _applied(prim: Any) -> list[str]:
    return [str(item) for item in prim.GetAppliedSchemas()]


def _has_schema(prim: Any, schema_name: str) -> bool:
    return schema_name in _applied(prim)


def _nearest_schema_ancestor(prim: Any, schema_name: str) -> str | None:
    current = prim.GetParent()
    while current and current.IsValid():
        if _has_schema(current, schema_name):
            return str(current.GetPath())
        current = current.GetParent()
    return None


def _attr_value(api: Any, getter_name: str) -> Any:
    getter = getattr(api, getter_name, None)
    if getter is None:
        return None
    attr = getter()
    if not attr:
        return None
    return attr.Get()


def _prim_physics_row(stage: Any, prim_path: str) -> dict[str, Any]:
    from pxr import PhysxSchema, UsdPhysics

    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        return {"path": prim_path, "exists": False}

    collision_api = UsdPhysics.CollisionAPI(prim)
    rigid_api = UsdPhysics.RigidBodyAPI(prim)
    mass_api = UsdPhysics.MassAPI(prim)
    physx_collision_api = PhysxSchema.PhysxCollisionAPI(prim)
    row: dict[str, Any] = {
        "path": prim_path,
        "exists": True,
        "type_name": prim.GetTypeName(),
        "applied_schemas": _applied(prim),
        "has_collision_api": bool(collision_api),
        "has_rigid_body_api": bool(rigid_api),
        "has_mass_api": bool(mass_api),
        "has_physx_collision_api": bool(physx_collision_api),
        "rigid_body_ancestor": _nearest_schema_ancestor(prim, "PhysicsRigidBodyAPI"),
        "bbox": _bbox_row(stage, prim_path),
    }
    if collision_api:
        row["collision_enabled"] = _attr_value(collision_api, "GetCollisionEnabledAttr")
    if rigid_api:
        row["rigid_body_enabled"] = _attr_value(rigid_api, "GetRigidBodyEnabledAttr")
        row["kinematic_enabled"] = _attr_value(rigid_api, "GetKinematicEnabledAttr")
    if mass_api:
        row["mass"] = _attr_value(mass_api, "GetMassAttr")
        row["density"] = _attr_value(mass_api, "GetDensityAttr")
    if physx_collision_api:
        row["contact_offset"] = _attr_value(physx_collision_api, "GetContactOffsetAttr")
        row["rest_offset"] = _attr_value(physx_collision_api, "GetRestOffsetAttr")
        row["torsional_patch_radius"] = _attr_value(physx_collision_api, "GetTorsionalPatchRadiusAttr")
        row["min_torsional_patch_radius"] = _attr_value(physx_collision_api, "GetMinTorsionalPatchRadiusAttr")
    return row


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 44 Gripper Contact Runtime Inspection",
        "",
        f"- status: `{payload['status']}`",
        f"- stage: `{payload['inputs']['stage_usd']}`",
        f"- object creation: `{payload['inputs']['object_creation']}`",
        f"- object side length: `{payload.get('object_side_length_stage_units')}` stage units",
        f"- object displacement after warmup: `{payload.get('object_displacement_after_warmup')}` stage units",
        "",
        "## Physics Rows",
        "",
        "| prim | collision | rigid body | rigid ancestor | mass | contact offset | rest offset |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for key in ["left_finger_proxy", "right_finger_proxy", "test_object"]:
        row = payload["physics_rows"][key]
        lines.append(
            f"| `{row['path']}` | `{row.get('has_collision_api')}` | `{row.get('has_rigid_body_api')}` | "
            f"`{row.get('rigid_body_ancestor') or ''}` | {row.get('mass')} | "
            f"{row.get('contact_offset')} | {row.get('rest_offset')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This diagnostic separates three questions that Phase 43 conflated:",
            "",
            "1. Whether finger proxy colliders are owned by moving rigid-body links.",
            "2. Whether the inserted object is authored as a dynamic rigid body after reset.",
            "3. Whether the object moves before deliberate finger closure, which indicates immediate penetration/contact instability.",
            "",
            "A stable next gate requires the object to be dynamic, the finger proxies to have a rigid-body ancestor, and the object not to be ejected during warmup.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect runtime physics state for ALOHA1 gripper proxy contact setup.")
    parser.add_argument("--stage-usd", default=str(DEFAULT_STAGE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--object-creation", choices=("dynamic_cuboid", "raw_usd"), default="dynamic_cuboid")
    parser.add_argument("--object-side-length", type=float, default=0.02)
    parser.add_argument("--object-mass", type=float, default=0.01)
    parser.add_argument("--open-offset", type=float, default=0.006)
    parser.add_argument("--settle-steps", type=int, default=60)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--physics-dt", type=float, default=1.0 / 50.0)
    parser.add_argument("--gravity", type=float, default=0.0)
    parser.add_argument("--limit-margin", type=float, default=0.001)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    json_path = output_dir / "gripper_contact_runtime_inspection.json"
    md_path = output_dir / "gripper_contact_runtime_inspection.md"
    payload: dict[str, Any] = {
        "status": "STARTED",
        "real_robot_touched": False,
        "stage_saved": False,
        "inputs": {
            "stage_usd": _rel(args.stage_usd),
            "side": args.side,
            "object_creation": args.object_creation,
            "object_side_length": args.object_side_length,
            "object_mass": args.object_mass,
            "open_offset": args.open_offset,
            "settle_steps": args.settle_steps,
            "warmup_steps": args.warmup_steps,
            "physics_dt": args.physics_dt,
            "gravity": args.gravity,
            "limit_margin": args.limit_margin,
        },
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
        if not left_box.get("bbox_valid") or not right_box.get("bbox_valid"):
            raise RuntimeError("Finger proxy bbox pair is invalid.")
        center = (np.asarray(left_box["center"], dtype=np.float64) + np.asarray(right_box["center"], dtype=np.float64)) * 0.5
        object_path = "/World/phase44_contact_inspection_cube"
        _create_passive_cube(
            world=world,
            stage=stage,
            path=object_path,
            center=center,
            side_length=args.object_side_length,
            mass=args.object_mass,
            creation_mode=args.object_creation,
        )
        rows_before_reset = {
            "left_finger_proxy": _prim_physics_row(stage, paths["left_finger"]),
            "right_finger_proxy": _prim_physics_row(stage, paths["right_finger"]),
            "test_object": _prim_physics_row(stage, object_path),
        }
        world.reset()
        _apply_gravity(world, args.gravity)
        _set_full_state(art, open_target)
        _set_full_target(art, open_target)
        _set_finger_target_and_step(world, art, open_target, args.settle_steps)
        object_initial_box = _bbox_row(stage, object_path)
        object_initial_center = np.asarray(object_initial_box.get("center", [np.nan, np.nan, np.nan]), dtype=np.float64)
        for _ in range(args.warmup_steps):
            world.step(render=False)
        rows_after_reset = {
            "left_finger_proxy": _prim_physics_row(stage, paths["left_finger"]),
            "right_finger_proxy": _prim_physics_row(stage, paths["right_finger"]),
            "test_object": _prim_physics_row(stage, object_path),
        }
        object_final_box = _bbox_row(stage, object_path)
        object_final_center = np.asarray(object_final_box.get("center", [np.nan, np.nan, np.nan]), dtype=np.float64)
        displacement = float(np.linalg.norm(object_final_center - object_initial_center))
        object_dynamic_after_reset = bool(rows_after_reset["test_object"].get("has_rigid_body_api"))
        proxies_have_rigid_ancestor = bool(
            rows_after_reset["left_finger_proxy"].get("rigid_body_ancestor")
            and rows_after_reset["right_finger_proxy"].get("rigid_body_ancestor")
        )
        no_warmup_ejection = bool(np.isfinite(displacement) and displacement < max(args.object_side_length * 2.0, 1e-4))
        payload.update(
            {
                "status": "PASS" if object_dynamic_after_reset and proxies_have_rigid_ancestor else "FAILED_GATE",
                "object_side_length_stage_units": args.object_side_length,
                "object_initial_center": object_initial_center.tolist(),
                "object_final_center_after_warmup": object_final_center.tolist(),
                "object_displacement_after_warmup": displacement,
                "open_target_values": open_values,
                "object_dynamic_after_reset": object_dynamic_after_reset,
                "finger_proxies_have_rigid_body_ancestor": proxies_have_rigid_ancestor,
                "no_warmup_ejection": no_warmup_ejection,
                "physics_rows_before_reset": rows_before_reset,
                "physics_rows": rows_after_reset,
                "outputs": {"json": _rel(json_path), "markdown": _rel(md_path)},
            }
        )
        _write_json(json_path, payload)
        payload["status"] = (
            "PASS" if object_dynamic_after_reset and proxies_have_rigid_ancestor and no_warmup_ejection else "FAILED_GATE"
        )
        _write_json(json_path, payload)
        md_path.write_text(_render_markdown(_json_safe(payload)))
        print(json.dumps({"status": payload["status"], "json": _rel(json_path), "markdown": _rel(md_path)}, ensure_ascii=False), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except BaseException as exc:
        payload.update(
            {
                "status": "EXCEPTION",
                "exception": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc().splitlines()[-25:],
            }
        )
        _write_json(json_path, payload)
        print(json.dumps({"status": "EXCEPTION", "json": _rel(json_path), "exception": payload["exception"]}, ensure_ascii=False), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    raise SystemExit(main())
