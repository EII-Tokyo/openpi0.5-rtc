from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np

from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _apply_arm_gains
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _apply_gravity
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _get_limits
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _json_safe
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _set_full_state
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _set_full_target
from aloha_isaac_replay.scripts.validate_aloha1_native_single_joint_response import _safe_target
from aloha_isaac_replay.validation.contact_proxy_profiles import resolve_contact_proxy_paths

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE = REPO_ROOT / "local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_bbox_proxy_runtime.usda"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase42_gripper_proxy_gap_20260718"

FINGER_PROXY_PATHS = resolve_contact_proxy_paths("legacy_puppet")


def _rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n")


def _bbox_row(stage: Any, prim_path: str) -> dict[str, Any]:
    from pxr import Usd
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        return {"path": prim_path, "exists": False, "bbox_valid": False}
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    if box.IsEmpty():
        return {"path": prim_path, "exists": True, "bbox_valid": False}
    min_pt = box.GetMin()
    max_pt = box.GetMax()
    center = [(float(min_pt[i]) + float(max_pt[i])) * 0.5 for i in range(3)]
    size = [float(max_pt[i]) - float(min_pt[i]) for i in range(3)]
    return {
        "path": prim_path,
        "exists": True,
        "bbox_valid": bool(all(item > 0 for item in size)),
        "min": [float(min_pt[i]) for i in range(3)],
        "max": [float(max_pt[i]) for i in range(3)],
        "center": center,
        "size": size,
    }


def _gap_metrics(left_box: dict[str, Any], right_box: dict[str, Any]) -> dict[str, Any]:
    if not left_box.get("bbox_valid") or not right_box.get("bbox_valid"):
        return {"bbox_pair_valid": False}
    left_center = np.asarray(left_box["center"], dtype=np.float64)
    right_center = np.asarray(right_box["center"], dtype=np.float64)
    center_delta = left_center - right_center
    axis_abs = np.abs(center_delta)
    dominant_axis = int(np.argmax(axis_abs))
    axis_names = ["x", "y", "z"]
    return {
        "bbox_pair_valid": True,
        "center_distance": float(np.linalg.norm(center_delta)),
        "center_delta": center_delta.tolist(),
        "dominant_axis": axis_names[dominant_axis],
        "dominant_axis_gap": float(axis_abs[dominant_axis]),
        "y_center_gap": float(axis_abs[1]),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "phase",
        "step",
        "target",
        "qpos",
        "qvel",
        "qpos_error",
        "max_abs_qpos_delta",
        "center_distance",
        "dominant_axis",
        "dominant_axis_gap",
        "y_center_gap",
        "left_center_x",
        "left_center_y",
        "left_center_z",
        "right_center_x",
        "right_center_y",
        "right_center_z",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 42 Gripper Proxy Gap Gate",
        "",
        f"- status: `{payload['status']}`",
        f"- side: `{payload['inputs']['side']}`",
        f"- stage: `{payload['inputs']['stage_usd']}`",
        f"- controlled DOF: `{payload.get('controlled_dof')}`",
        f"- maximum gripper qpos delta: `{payload.get('max_qpos_delta')}`",
        f"- maximum proxy center-distance delta: `{payload.get('max_center_distance_delta')}`",
        f"- minimum required proxy delta: `{payload['inputs']['min_gap_delta']}`",
        "",
        "## Interpretation",
        "",
        "This gate checks whether the gripper-only collision proxies are attached to moving finger links. It does not validate bottle contact or grasp success.",
        "",
        "Passing means the proxy geometry is at least kinematically coupled to gripper motion. Failing means a later bottle contact test would be ambiguous because the collision proxy may not move with the finger.",
        "",
        "## Phase Summary",
        "",
        "| phase | target | final qpos | final center distance | center-distance delta | bbox valid |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in payload.get("phase_summaries", []):
        lines.append(
            f"| `{item['phase']}` | `{item['target']}` | `{item['final_qpos']}` | "
            f"{item['final_center_distance']:.6f} | {item['center_distance_delta_from_home']:.6f} | "
            f"`{item['bbox_pair_valid']}` |"
        )
    path.write_text("\n".join(lines) + "\n")


def _run_gap_gate(
    *,
    world: Any,
    stage: Any,
    art: Any,
    side: str,
    control_mode: str,
    target_origin: str,
    phase_offsets: list[float],
    phase_steps: int,
    settle_steps: int,
    limit_margin: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dof_names = list(art.dof_names)
    limits = _get_limits(art)
    if control_mode == "gripper":
        control_specs = [("gripper", 1.0)]
    elif control_mode == "same_fingers":
        control_specs = [("left_finger", 1.0), ("right_finger", 1.0)]
    elif control_mode == "opposed_fingers":
        control_specs = [("left_finger", 1.0), ("right_finger", -1.0)]
    else:
        raise ValueError(f"Unsupported control mode: {control_mode}")
    missing = [name for name, _sign in control_specs if name not in dof_names]
    if missing:
        raise ValueError(f"{side}:{missing} not found in runtime DOFs: {dof_names}")
    control_indices = [(name, dof_names.index(name), sign) for name, sign in control_specs]
    initial = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
    origin = initial.copy()
    for _name, idx, _sign in control_indices:
        lower, upper = [float(x) for x in limits[idx]]
        if target_origin == "limit_midpoint" and np.isfinite(lower) and np.isfinite(upper):
            origin[idx] = (lower + upper) * 0.5
        elif target_origin == "current":
            origin[idx] = initial[idx]
        else:
            raise ValueError(f"Unsupported target origin: {target_origin}")
    _set_full_state(art, origin)
    _set_full_target(art, origin)
    for _ in range(settle_steps):
        world.step(render=False)

    paths = FINGER_PROXY_PATHS[side]
    home_left_box = _bbox_row(stage, paths["left_finger"])
    home_right_box = _bbox_row(stage, paths["right_finger"])
    home_gap = _gap_metrics(home_left_box, home_right_box)
    rows: list[dict[str, Any]] = []
    phase_summaries: list[dict[str, Any]] = []
    max_qpos_delta = 0.0
    max_center_distance_delta = 0.0

    for offset in phase_offsets:
        target = origin.copy()
        target_values: dict[str, float] = {}
        clipped_flags: dict[str, bool] = {}
        for name, idx, sign in control_indices:
            lower, upper = [float(x) for x in limits[idx]]
            target_value, clipped = _safe_target(float(origin[idx]), float(offset) * sign, lower, upper, limit_margin)
            target[idx] = target_value
            target_values[name] = target_value
            clipped_flags[name] = clipped
        phase_name = f"offset_{offset:+.4f}".replace("+", "p").replace("-", "m").replace(".", "p")
        phase_rows: list[dict[str, Any]] = []
        for step in range(phase_steps):
            _set_full_target(art, target)
            world.step(render=False)
            qpos = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
            qvel = np.asarray(art.get_joint_velocities(), dtype=np.float64).reshape(-1)
            left_box = _bbox_row(stage, paths["left_finger"])
            right_box = _bbox_row(stage, paths["right_finger"])
            gap = _gap_metrics(left_box, right_box)
            qpos_deltas = {name: float(qpos[idx] - origin[idx]) for name, idx, _sign in control_indices}
            target_errors = {name: float(qpos[idx] - target_values[name]) for name, idx, _sign in control_indices}
            row = {
                "phase": phase_name,
                "step": step,
                "target": json.dumps(target_values, sort_keys=True),
                "qpos": json.dumps({name: float(qpos[idx]) for name, idx, _sign in control_indices}, sort_keys=True),
                "qvel": json.dumps({name: float(qvel[idx]) for name, idx, _sign in control_indices}, sort_keys=True),
                "qpos_error": json.dumps(target_errors, sort_keys=True),
                "max_abs_qpos_delta": max(abs(value) for value in qpos_deltas.values()),
                "center_distance": gap.get("center_distance"),
                "dominant_axis": gap.get("dominant_axis"),
                "dominant_axis_gap": gap.get("dominant_axis_gap"),
                "y_center_gap": gap.get("y_center_gap"),
                "left_center_x": left_box.get("center", [None, None, None])[0],
                "left_center_y": left_box.get("center", [None, None, None])[1],
                "left_center_z": left_box.get("center", [None, None, None])[2],
                "right_center_x": right_box.get("center", [None, None, None])[0],
                "right_center_y": right_box.get("center", [None, None, None])[1],
                "right_center_z": right_box.get("center", [None, None, None])[2],
            }
            rows.append(row)
            phase_rows.append(row)
        final = phase_rows[-1]
        final_center_distance = (
            float(final["center_distance"]) if final["center_distance"] is not None else float("nan")
        )
        home_center_distance = float(home_gap.get("center_distance", float("nan")))
        delta = abs(final_center_distance - home_center_distance) if np.isfinite(home_center_distance) else float("nan")
        final_qpos = json.loads(final["qpos"])
        final_qpos_delta = {name: float(final_qpos[name]) - float(origin[idx]) for name, idx, _sign in control_indices}
        max_qpos_delta = max(max_qpos_delta, max(abs(value) for value in final_qpos_delta.values()))
        if np.isfinite(delta):
            max_center_distance_delta = max(max_center_distance_delta, delta)
        phase_summaries.append(
            {
                "phase": phase_name,
                "requested_offset": float(offset),
                "target": target_values,
                "target_clipped": clipped_flags,
                "final_qpos": final_qpos,
                "final_qpos_delta_from_home": final_qpos_delta,
                "final_center_distance": final_center_distance,
                "home_center_distance": home_center_distance,
                "center_distance_delta_from_home": delta,
                "bbox_pair_valid": bool(home_gap.get("bbox_pair_valid") and final["center_distance"] is not None),
            }
        )

    return (
        {
            "control_mode": control_mode,
            "controlled_dofs": [
                {
                    "name": name,
                    "index": idx,
                    "sign": sign,
                    "runtime_lower": float(limits[idx][0]),
                    "runtime_upper": float(limits[idx][1]),
                    "initial_qpos": float(initial[idx]),
                    "origin_qpos": float(origin[idx]),
                }
                for name, idx, sign in control_indices
            ],
            "home_left_box": home_left_box,
            "home_right_box": home_right_box,
            "home_gap": home_gap,
            "max_qpos_delta": max_qpos_delta,
            "max_center_distance_delta": max_center_distance_delta,
            "phase_summaries": phase_summaries,
        },
        rows,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate whether gripper-only bbox proxies move with the gripper DOF."
    )
    parser.add_argument("--stage-usd", default=str(DEFAULT_STAGE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--control-mode", choices=("gripper", "same_fingers", "opposed_fingers"), default="gripper")
    parser.add_argument("--target-origin", choices=("current", "limit_midpoint"), default="limit_midpoint")
    parser.add_argument("--phase-offset", action="append", type=float, default=None)
    parser.add_argument("--phase-steps", type=int, default=160)
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--physics-dt", type=float, default=1.0 / 50.0)
    parser.add_argument("--gravity", type=float, default=0.0)
    parser.add_argument("--arm-kp", type=float, default=None)
    parser.add_argument("--arm-kd", type=float, default=None)
    parser.add_argument("--limit-margin", type=float, default=0.001)
    parser.add_argument("--min-qpos-delta", type=float, default=0.001)
    parser.add_argument("--min-gap-delta", type=float, default=0.0005)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    json_path = output_dir / "gripper_proxy_gap_metrics.json"
    csv_path = output_dir / "gripper_proxy_gap_timeseries.csv"
    md_path = output_dir / "gripper_proxy_gap_metrics.md"
    phase_offsets = args.phase_offset or [0.0, 0.004, 0.0, -0.004, 0.006, -0.006]
    payload: dict[str, Any] = {
        "status": "STARTED",
        "overall_pass": False,
        "real_robot_touched": False,
        "stage_saved": False,
        "inputs": {
            "stage_usd": _rel(args.stage_usd),
            "side": args.side,
            "control_mode": args.control_mode,
            "target_origin": args.target_origin,
            "phase_offsets": phase_offsets,
            "phase_steps": args.phase_steps,
            "settle_steps": args.settle_steps,
            "physics_dt": args.physics_dt,
            "gravity": args.gravity,
            "limit_margin": args.limit_margin,
            "min_qpos_delta": args.min_qpos_delta,
            "min_gap_delta": args.min_gap_delta,
        },
        "outputs": {"json": _rel(json_path), "csv": _rel(csv_path), "markdown": _rel(md_path)},
    }
    _write_json(json_path, payload)

    try:
        from isaacsim import SimulationApp

        app_config = dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG)
        app_config["fast_shutdown"] = False
        _app = SimulationApp(app_config)
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        import isaacsim.core.utils.stage as stage_utils
        import omni.usd

        stage_utils.open_stage(str(Path(args.stage_usd).resolve()))
        World.clear_instance()
        world = World(stage_units_in_meters=0.01, backend="numpy", device="cpu")
        world.set_simulation_dt(physics_dt=args.physics_dt, rendering_dt=args.physics_dt)
        stage = omni.usd.get_context().get_stage()
        art_path = FINGER_PROXY_PATHS[args.side]["articulation"]
        art = world.scene.add(SingleArticulation(prim_path=art_path, name=f"{args.side}_vx300s"))
        world.reset()
        _apply_gravity(world, args.gravity)
        _apply_arm_gains(art, args.arm_kp, args.arm_kd)

        summary, rows = _run_gap_gate(
            world=world,
            stage=stage,
            art=art,
            side=args.side,
            control_mode=args.control_mode,
            target_origin=args.target_origin,
            phase_offsets=phase_offsets,
            phase_steps=args.phase_steps,
            settle_steps=args.settle_steps,
            limit_margin=args.limit_margin,
        )
        qpos_ok = summary["max_qpos_delta"] >= args.min_qpos_delta
        gap_ok = summary["max_center_distance_delta"] >= args.min_gap_delta
        bbox_ok = bool(summary["home_gap"].get("bbox_pair_valid"))
        overall_pass = bool(qpos_ok and gap_ok and bbox_ok)
        payload.update(summary)
        payload.update(
            {
                "status": "PASS" if overall_pass else "FAILED_GATE",
                "overall_pass": overall_pass,
                "qpos_delta_ok": qpos_ok,
                "gap_delta_ok": gap_ok,
                "bbox_pair_ok": bbox_ok,
                "csv": _rel(csv_path),
                "markdown": _rel(md_path),
                "next_gate": "passive_object_contact_smoke"
                if overall_pass
                else "inspect_gripper_proxy_attachment_or_mimic_joint",
            }
        )
        _write_csv(csv_path, rows)
        _write_json(json_path, payload)
        _write_markdown(md_path, _json_safe(payload))
        print(
            json.dumps(
                {"status": payload["status"], "json": _rel(json_path), "markdown": _rel(md_path)}, ensure_ascii=False
            ),
            flush=True,
        )
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
        print(
            json.dumps(
                {"status": payload["status"], "json": _rel(json_path), "exception": payload["exception"]},
                ensure_ascii=False,
            ),
            flush=True,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    raise SystemExit(main())
