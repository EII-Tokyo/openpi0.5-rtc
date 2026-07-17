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
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _apply_side_base_offsets
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _get_gains
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _get_limits
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _json_safe
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _set_robot_collisions_enabled
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _set_full_state
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _set_full_target


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEFT_USD = REPO_ROOT / "assets/isaac/aloha1_native_physics_wrapper/aloha1_left.usda"
DEFAULT_RIGHT_USD = REPO_ROOT / "assets/isaac/aloha1_native_physics_wrapper/aloha1_right.usda"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase25_native_single_joint_response_20260718"


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
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "side",
        "joint",
        "phase",
        "step",
        "target",
        "qpos",
        "qvel",
        "error",
        "lower",
        "upper",
        "limit_violation",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 25 Native Single-Joint Dynamic Response",
        "",
        f"- status: `{payload['status']}`",
        f"- overall pass: `{payload['overall_pass']}`",
        f"- physics dt: `{payload['inputs']['physics_dt']}`",
        f"- phase steps: `{payload['inputs']['phase_steps']}`",
        f"- final error tolerance: `{payload['inputs']['final_error_tolerance']}`",
        "",
        "## Joint Results",
        "",
        "| side | joint | status | max abs error | max final abs error | limit violations | direction ok |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for item in payload["joint_results"]:
        lines.append(
            f"| {item['side']} | {item['joint']} | `{item['status']}` | "
            f"{item['max_abs_error']:.6f} | {item['max_final_abs_error']:.6f} | "
            f"{item['limit_violations']} | `{item['direction_ok']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This gate isolates drive tracking to one joint at a time. It does not validate multi-joint coordination, collision, gripper dynamics, bottle contact, or task reward.",
            "A failure here means full trajectory replay should not be tuned yet; first inspect the failed joint's target, qpos, qvel, limits, and drive gains in the CSV artifact.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _parse_joint_specs(values: list[str]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"Joint spec must be SIDE:JOINT, got {value!r}")
        side, joint = value.split(":", 1)
        if side not in {"left", "right"}:
            raise ValueError(f"Joint side must be left or right, got {side!r}")
        if not joint:
            raise ValueError(f"Missing joint name in spec {value!r}")
        specs.append((side, joint))
    return specs


def _limit_violation(qpos: float, lower: float, upper: float, tol: float) -> bool:
    return bool(np.isfinite(lower) and np.isfinite(upper) and (qpos < lower - tol or qpos > upper + tol))


def _safe_target(home: float, offset: float, lower: float, upper: float, margin: float) -> tuple[float, bool]:
    target = home + offset
    if not (np.isfinite(lower) and np.isfinite(upper)):
        return target, False
    clipped = min(max(target, lower + margin), upper - margin)
    return float(clipped), bool(abs(clipped - target) > 1e-12)


def _run_joint_response(
    *,
    world: Any,
    art: Any,
    side: str,
    joint: str,
    phase_offsets: list[float],
    phase_steps: int,
    settle_steps: int,
    final_error_tolerance: float,
    limit_margin: float,
    limit_tolerance: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dof_names = list(art.dof_names)
    if joint not in dof_names:
        raise ValueError(f"{side}:{joint} not found in runtime DOFs: {dof_names}")
    idx = dof_names.index(joint)
    limits = _get_limits(art)
    kps, kds = _get_gains(art)
    lower, upper = [float(x) for x in limits[idx]]
    home = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
    home_value = float(home[idx])
    rows: list[dict[str, Any]] = []
    phase_summaries: list[dict[str, Any]] = []
    max_abs_error = 0.0
    limit_violations = 0

    _set_full_state(art, home)
    _set_full_target(art, home)
    for _ in range(settle_steps):
        world.step(render=False)

    for offset in phase_offsets:
        target = home.copy()
        target_value, clipped = _safe_target(home_value, float(offset), lower, upper, limit_margin)
        target[idx] = target_value
        phase_name = f"offset_{offset:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")
        start_qpos = float(np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)[idx])
        phase_qpos: list[float] = []
        phase_qvel: list[float] = []
        phase_errors: list[float] = []
        phase_violations = 0
        for step in range(phase_steps):
            _set_full_target(art, target)
            world.step(render=False)
            qpos = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
            qvel = np.asarray(art.get_joint_velocities(), dtype=np.float64).reshape(-1)
            error = float(qpos[idx] - target_value)
            violation = _limit_violation(float(qpos[idx]), lower, upper, limit_tolerance)
            phase_qpos.append(float(qpos[idx]))
            phase_qvel.append(float(qvel[idx]))
            phase_errors.append(error)
            phase_violations += int(violation)
            rows.append(
                {
                    "side": side,
                    "joint": joint,
                    "phase": phase_name,
                    "step": step,
                    "target": target_value,
                    "qpos": float(qpos[idx]),
                    "qvel": float(qvel[idx]),
                    "error": error,
                    "lower": lower,
                    "upper": upper,
                    "limit_violation": int(violation),
                }
            )
        final_qpos = phase_qpos[-1] if phase_qpos else start_qpos
        final_error = phase_errors[-1] if phase_errors else start_qpos - target_value
        abs_errors = [abs(x) for x in phase_errors]
        max_abs_error = max(max_abs_error, max(abs_errors, default=0.0))
        limit_violations += phase_violations
        moved_direction_ok = True
        if abs(offset) > 1e-12 and not clipped:
            moved_direction_ok = (final_qpos - home_value) * offset > 0.0
        phase_summaries.append(
            {
                "phase": phase_name,
                "requested_offset": float(offset),
                "target": target_value,
                "target_clipped": clipped,
                "start_qpos": start_qpos,
                "final_qpos": final_qpos,
                "final_abs_error": abs(final_error),
                "max_abs_error": max(abs_errors, default=0.0),
                "max_abs_qvel": max((abs(x) for x in phase_qvel), default=0.0),
                "limit_violations": phase_violations,
                "moved_direction_ok": bool(moved_direction_ok),
            }
        )

    max_final_abs_error = max((float(row["final_abs_error"]) for row in phase_summaries), default=0.0)
    direction_ok = all(bool(row["moved_direction_ok"]) for row in phase_summaries)
    no_limit_violations = limit_violations == 0
    converged = max_final_abs_error <= final_error_tolerance
    status = "PASS" if no_limit_violations and direction_ok and converged else "FAIL"
    summary = {
        "side": side,
        "joint": joint,
        "status": status,
        "runtime_index": idx,
        "runtime_lower": lower,
        "runtime_upper": upper,
        "runtime_stiffness": kps[idx],
        "runtime_damping": kds[idx],
        "home_qpos": home_value,
        "max_abs_error": max_abs_error,
        "max_final_abs_error": max_final_abs_error,
        "final_error_tolerance": final_error_tolerance,
        "limit_violations": limit_violations,
        "direction_ok": direction_ok,
        "converged_within_tolerance": converged,
        "phase_summaries": phase_summaries,
    }
    return summary, rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate native ALOHA1 wrapper single-joint dynamic response.")
    parser.add_argument("--left-usd", default=str(DEFAULT_LEFT_USD))
    parser.add_argument("--right-usd", default=str(DEFAULT_RIGHT_USD))
    parser.add_argument("--left-prim-path", default="/World/left/root_joint")
    parser.add_argument("--right-prim-path", default="/World/right/root_joint")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--joint", action="append", default=None)
    parser.add_argument("--phase-offset", action="append", type=float, default=None)
    parser.add_argument("--phase-steps", type=int, default=250)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--physics-dt", type=float, default=1.0 / 50.0)
    parser.add_argument("--gravity", type=float, default=0.0)
    parser.add_argument("--arm-kp", type=float, default=None)
    parser.add_argument("--arm-kd", type=float, default=None)
    parser.add_argument("--disable-robot-collisions", action="store_true")
    parser.add_argument("--base-separation", type=float, default=0.0)
    parser.add_argument("--base-separation-axis", choices=("X", "Y"), default="Y")
    parser.add_argument("--final-error-tolerance", type=float, default=0.075)
    parser.add_argument("--limit-margin", type=float, default=0.02)
    parser.add_argument("--limit-tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    json_path = output_dir / "single_joint_response_metrics.json"
    csv_path = output_dir / "single_joint_response_timeseries.csv"
    md_path = output_dir / "single_joint_response_metrics.md"
    joints = args.joint or ["left:waist", "right:shoulder"]
    phase_offsets = args.phase_offset or [0.0, 0.02, 0.0, -0.02, 0.05, -0.05]
    payload: dict[str, Any] = {
        "status": "STARTED",
        "overall_pass": False,
        "real_robot_touched": False,
        "stage_saved": False,
        "inputs": {
            "left_usd": _rel(args.left_usd),
            "right_usd": _rel(args.right_usd),
            "left_prim_path": args.left_prim_path,
            "right_prim_path": args.right_prim_path,
            "joints": joints,
            "phase_offsets": phase_offsets,
            "phase_steps": args.phase_steps,
            "settle_steps": args.settle_steps,
            "physics_dt": args.physics_dt,
            "gravity": args.gravity,
            "arm_kp": args.arm_kp,
            "arm_kd": args.arm_kd,
            "disable_robot_collisions": bool(args.disable_robot_collisions),
            "base_separation": args.base_separation,
            "base_separation_axis": args.base_separation_axis,
            "final_error_tolerance": args.final_error_tolerance,
            "limit_margin": args.limit_margin,
            "limit_tolerance": args.limit_tolerance,
        },
    }
    _write_json(json_path, payload)

    try:
        joint_specs = _parse_joint_specs(joints)
        from isaacsim import SimulationApp

        app_config = dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG)
        app_config["fast_shutdown"] = False
        _app = SimulationApp(app_config)
        import isaacsim.core.utils.stage as stage_utils
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        import omni.usd

        World.clear_instance()
        stage_utils.create_new_stage()
        world = World(stage_units_in_meters=1.0, backend="numpy", device="cpu")
        world.set_simulation_dt(physics_dt=args.physics_dt, rendering_dt=args.physics_dt)
        stage_utils.add_reference_to_stage(usd_path=str(Path(args.left_usd).resolve()), prim_path="/World/left")
        stage_utils.add_reference_to_stage(usd_path=str(Path(args.right_usd).resolve()), prim_path="/World/right")
        stage = omni.usd.get_context().get_stage()
        base_offsets = _apply_side_base_offsets(stage, args.base_separation_axis, args.base_separation)
        disabled_collision_prims = _set_robot_collisions_enabled(stage, False) if args.disable_robot_collisions else 0
        left = world.scene.add(SingleArticulation(prim_path=args.left_prim_path, name="left_vx300s"))
        right = world.scene.add(SingleArticulation(prim_path=args.right_prim_path, name="right_vx300s"))
        world.reset()
        _apply_gravity(world, args.gravity)
        _apply_arm_gains(left, args.arm_kp, args.arm_kd)
        _apply_arm_gains(right, args.arm_kp, args.arm_kd)
        baseline = {
            "left": np.asarray(left.get_joint_positions(), dtype=np.float64).reshape(-1).copy(),
            "right": np.asarray(right.get_joint_positions(), dtype=np.float64).reshape(-1).copy(),
        }

        joint_results: list[dict[str, Any]] = []
        all_rows: list[dict[str, Any]] = []
        for side, joint in joint_specs:
            # Each joint gate must start from the same baseline. Otherwise a
            # failed earlier probe can contaminate later joints and hide which
            # DOF is actually unstable.
            _set_full_state(left, baseline["left"])
            _set_full_state(right, baseline["right"])
            _set_full_target(left, baseline["left"])
            _set_full_target(right, baseline["right"])
            art = left if side == "left" else right
            result, rows = _run_joint_response(
                world=world,
                art=art,
                side=side,
                joint=joint,
                phase_offsets=phase_offsets,
                phase_steps=args.phase_steps,
                settle_steps=args.settle_steps,
                final_error_tolerance=args.final_error_tolerance,
                limit_margin=args.limit_margin,
                limit_tolerance=args.limit_tolerance,
            )
            joint_results.append(result)
            all_rows.extend(rows)

        overall_pass = all(item["status"] == "PASS" for item in joint_results)
        payload.update(
            {
                "status": "PASS" if overall_pass else "FAILED_GATE",
                "overall_pass": overall_pass,
                "base_offsets": base_offsets,
                "disabled_collision_prims": disabled_collision_prims,
                "left_dof_names": list(left.dof_names),
                "right_dof_names": list(right.dof_names),
                "joint_results": joint_results,
                "csv": _rel(csv_path),
                "markdown": _rel(md_path),
                "next_gate": "multi_joint_dynamic_tracking" if overall_pass else "inspect_failed_single_joint_drive_response",
            }
        )
        _write_csv(csv_path, all_rows)
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
