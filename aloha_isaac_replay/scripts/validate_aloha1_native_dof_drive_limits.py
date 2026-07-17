from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEFT_USD = REPO_ROOT / "assets/isaac/aloha1_native_physics_wrapper/aloha1_left.usda"
DEFAULT_RIGHT_USD = REPO_ROOT / "assets/isaac/aloha1_native_physics_wrapper/aloha1_right.usda"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase20_dof_drive_limits_20260718"


def _rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        return value.tolist()
    except Exception:
        return repr(value)


def _call_optional(obj: Any, names: tuple[str, ...], *args: Any) -> Any:
    for name in names:
        if hasattr(obj, name):
            try:
                return getattr(obj, name)(*args)
            except Exception:
                continue
    return None


def _as_flat_optional(value: Any, n: int) -> list[float | None]:
    if value is None:
        return [None] * n
    try:
        arr = np.squeeze(np.asarray(value, dtype=np.float64))
    except Exception:
        return [None] * n
    vals = [float(arr)] if arr.ndim == 0 else [float(x) for x in arr.reshape(-1).tolist()]
    return vals[:n] + [None] * max(0, n - len(vals))


def _get_limits(art: Any) -> list[list[float | None]]:
    n = int(art.num_dof)
    value = _call_optional(art._articulation_view, ("get_dof_limits", "get_joint_limits"))
    if value is None and hasattr(art._articulation_view, "_physics_view"):
        value = _call_optional(art._articulation_view._physics_view, ("get_dof_limits", "get_joint_limits"))
    if value is None:
        return [[None, None] for _ in range(n)]
    arr = np.squeeze(np.asarray(value, dtype=np.float64))
    if arr.shape == (n, 2):
        return [[float(lo), float(hi)] for lo, hi in arr.tolist()]
    if arr.shape == (2, n):
        return [[float(lo), float(hi)] for lo, hi in arr.T.tolist()]
    arr = arr.reshape((-1, 2))
    rows = [[float(lo), float(hi)] for lo, hi in arr[:n].tolist()]
    rows.extend([[None, None]] * max(0, n - len(rows)))
    return rows


def _get_gains(art: Any) -> tuple[list[float | None], list[float | None]]:
    n = int(art.num_dof)
    value = _call_optional(art._articulation_view, ("get_gains",))
    if isinstance(value, tuple) and len(value) >= 2:
        return _as_flat_optional(value[0], n), _as_flat_optional(value[1], n)
    return [None] * n, [None] * n


def _get_max_efforts(art: Any) -> list[float | None]:
    n = int(art.num_dof)
    value = _call_optional(art._articulation_view, ("get_max_efforts", "get_dof_max_efforts"))
    if value is None and hasattr(art._articulation_view, "_physics_view"):
        value = _call_optional(art._articulation_view._physics_view, ("get_dof_max_forces", "get_dof_max_efforts"))
    return _as_flat_optional(value, n)


def _get_max_velocities(art: Any) -> list[float | None]:
    n = int(art.num_dof)
    value = _call_optional(art._articulation_view, ("get_max_velocities", "get_dof_max_velocities"))
    if value is None and hasattr(art._articulation_view, "_physics_view"):
        value = _call_optional(art._articulation_view._physics_view, ("get_dof_max_velocities",))
    return _as_flat_optional(value, n)


def _finite_ordered_limit(row: list[float | None]) -> bool:
    lo, hi = row
    return lo is not None and hi is not None and math.isfinite(lo) and math.isfinite(hi) and lo < hi


def _positive_finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0.0


def _drive_values(prim: Any, drive_type: str) -> dict[str, Any]:
    from pxr import UsdPhysics

    drive = UsdPhysics.DriveAPI(prim, drive_type)
    return {
        "type": drive_type,
        "stiffness": drive.GetStiffnessAttr().Get(),
        "damping": drive.GetDampingAttr().Get(),
        "max_force": drive.GetMaxForceAttr().Get(),
        "target_position": drive.GetTargetPositionAttr().Get(),
        "target_velocity": drive.GetTargetVelocityAttr().Get(),
    }


def _find_joint_prim_info(stage: Any, root_prefix: str, dof_name: str) -> dict[str, Any] | None:
    from pxr import UsdPhysics

    suffix = f"/{dof_name}"
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(root_prefix) or not path.endswith(suffix):
            continue
        type_name = prim.GetTypeName()
        if "Joint" not in type_name:
            continue
        info: dict[str, Any] = {
            "path": path,
            "type": type_name,
            "applied_schemas": [str(item) for item in prim.GetAppliedSchemas()],
            "drive": None,
        }
        info["is_mimic"] = any(schema.startswith("PhysxMimicJointAPI") for schema in info["applied_schemas"])
        joint = UsdPhysics.Joint(prim)
        info["body0"] = [str(x) for x in joint.GetBody0Rel().GetTargets()]
        info["body1"] = [str(x) for x in joint.GetBody1Rel().GetTargets()]
        if type_name == "PhysicsRevoluteJoint":
            typed = UsdPhysics.RevoluteJoint(prim)
            info["axis"] = str(typed.GetAxisAttr().Get())
            info["usd_lower"] = typed.GetLowerLimitAttr().Get()
            info["usd_upper"] = typed.GetUpperLimitAttr().Get()
            info["drive"] = _drive_values(prim, "angular")
        elif type_name == "PhysicsPrismaticJoint":
            typed = UsdPhysics.PrismaticJoint(prim)
            info["axis"] = str(typed.GetAxisAttr().Get())
            info["usd_lower"] = typed.GetLowerLimitAttr().Get()
            info["usd_upper"] = typed.GetUpperLimitAttr().Get()
            info["drive"] = _drive_values(prim, "linear")
        return info
    return None


def _side_payload(stage: Any, art: Any, side: str, root_prefix: str) -> dict[str, Any]:
    n = int(art.num_dof)
    limits = _get_limits(art)
    kps, kds = _get_gains(art)
    efforts = _get_max_efforts(art)
    velocities = _get_max_velocities(art)
    qpos = _as_flat_optional(art.get_joint_positions(), n)
    rows = []
    for idx, name in enumerate(list(art.dof_names)):
        static_joint = _find_joint_prim_info(stage, root_prefix, name)
        rows.append(
            {
                "index": idx,
                "name": name,
                "runtime_lower": limits[idx][0],
                "runtime_upper": limits[idx][1],
                "runtime_limit_finite_ordered": _finite_ordered_limit(limits[idx]),
                "runtime_stiffness": kps[idx],
                "runtime_damping": kds[idx],
                "runtime_effort_limit": efforts[idx],
                "runtime_velocity_limit": velocities[idx],
                "runtime_effort_positive": _positive_finite(efforts[idx]),
                "runtime_velocity_positive": _positive_finite(velocities[idx]),
                "current_qpos": qpos[idx],
                "static_joint": static_joint,
                "static_joint_found": static_joint is not None,
                "static_is_mimic": bool(static_joint and static_joint.get("is_mimic")),
            }
        )
    commandable_rows = [row for row in rows if not row["static_is_mimic"]]
    gates = {
        "has_expected_dof_count": n == 9,
        "all_runtime_limits_finite_ordered": all(row["runtime_limit_finite_ordered"] for row in rows),
        "all_commandable_runtime_efforts_positive": all(row["runtime_effort_positive"] for row in commandable_rows),
        "all_runtime_velocities_positive": all(row["runtime_velocity_positive"] for row in rows),
        "all_static_joints_found": all(row["static_joint_found"] for row in rows),
        "mimic_joints_identified": any(row["static_is_mimic"] for row in rows),
    }
    return {
        "side": side,
        "prim_path": art.prim_path,
        "num_dof": n,
        "dof_names": list(art.dof_names),
        "rows": rows,
        "gates": gates,
        "overall_pass": all(gates.values()),
    }


def _find_articulation_roots(stage: Any, prefix: str) -> list[str]:
    roots = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(prefix):
            continue
        schemas = [str(item) for item in prim.GetAppliedSchemas()]
        if "ArticulationRootAPI" in schemas or "PhysicsArticulationRootAPI" in schemas:
            roots.append(path)
    return roots


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n")


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 20 DOF / Drive / Limit Validation",
        "",
        f"- left USD: `{payload['left_usd']}`",
        f"- right USD: `{payload['right_usd']}`",
        f"- overall pass: `{payload['overall_pass']}`",
        "",
        "## Gate Summary",
        "",
        "| Side | DOFs | finite ordered limits | commandable effort | positive velocity | static joints found | mimic found | Gate |",
        "| --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for side in ("left", "right"):
        item = payload["sides"][side]
        gates = item["gates"]
        lines.append(
            f"| {side} | {item['num_dof']} | {gates['all_runtime_limits_finite_ordered']} | "
            f"{gates['all_commandable_runtime_efforts_positive']} | {gates['all_runtime_velocities_positive']} | "
            f"{gates['all_static_joints_found']} | {gates['mimic_joints_identified']} | "
            f"{'PASS' if item['overall_pass'] else 'FAIL'} |"
        )
    lines.extend(["", "## DOF Rows", ""])
    for side in ("left", "right"):
        lines.extend(
            [
                f"### {side}",
                "",
                "| idx | name | mimic | runtime lower | runtime upper | effort | velocity | stiffness | damping | static joint |",
                "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in payload["sides"][side]["rows"]:
            static_path = row["static_joint"]["path"] if row["static_joint"] else ""
            lines.append(
                f"| {row['index']} | `{row['name']}` | {row['static_is_mimic']} | "
                f"{row['runtime_lower']} | {row['runtime_upper']} | "
                f"{row['runtime_effort_limit']} | {row['runtime_velocity_limit']} | "
                f"{row['runtime_stiffness']} | {row['runtime_damping']} | `{static_path}` |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate DOF limits, drives, efforts, and velocities on the ALOHA1 native wrapper candidate.")
    parser.add_argument("--left-usd", default=str(DEFAULT_LEFT_USD))
    parser.add_argument("--right-usd", default=str(DEFAULT_RIGHT_USD))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    json_path = output_dir / "dof_drive_limits.json"
    md_path = output_dir / "dof_drive_limits.md"
    payload: dict[str, Any] = {
        "left_usd": _rel(args.left_usd),
        "right_usd": _rel(args.right_usd),
        "status": "STARTED",
        "real_robot_touched": False,
        "stage_saved": False,
    }
    _write_json(json_path, payload)

    from isaacsim import SimulationApp

    app_config = dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG)
    app_config["fast_shutdown"] = False
    _app = SimulationApp(app_config)
    try:
        import isaacsim.core.utils.stage as stage_utils
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation

        World.clear_instance()
        stage_utils.create_new_stage()
        world = World(stage_units_in_meters=1.0, backend="numpy", device="cpu")
        stage_utils.add_reference_to_stage(usd_path=str(Path(args.left_usd).resolve()), prim_path="/World/left")
        stage_utils.add_reference_to_stage(usd_path=str(Path(args.right_usd).resolve()), prim_path="/World/right")
        stage = world.stage
        roots = {
            "left": _find_articulation_roots(stage, "/World/left"),
            "right": _find_articulation_roots(stage, "/World/right"),
        }
        if len(roots["left"]) != 1 or len(roots["right"]) != 1:
            payload.update({"status": "FAILED", "failure": "expected exactly one articulation root per side", "articulation_roots": roots})
            _write_json(json_path, payload)
            print(json.dumps({"json": _rel(json_path), "status": payload["status"], "failure": payload["failure"]}), flush=True)
            sys.stdout.flush()
            os._exit(2)
        left = world.scene.add(SingleArticulation(prim_path=roots["left"][0], name="left_vx300s"))
        right = world.scene.add(SingleArticulation(prim_path=roots["right"][0], name="right_vx300s"))
        world.reset()
        sides = {
            "left": _side_payload(stage, left, "left", "/World/left"),
            "right": _side_payload(stage, right, "right", "/World/right"),
        }
        payload.update(
            {
                "status": "PASS" if all(side["overall_pass"] for side in sides.values()) else "FAILED_GATE",
                "articulation_roots": roots,
                "sides": sides,
                "overall_pass": all(side["overall_pass"] for side in sides.values()),
            }
        )
        _write_json(json_path, payload)
        _write_markdown(md_path, _json_safe(payload))
        print(json.dumps({"json": _rel(json_path), "markdown": _rel(md_path), "status": payload["status"]}, ensure_ascii=False), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if payload["overall_pass"] else 3)
    except BaseException as exc:
        payload.update(
            {
                "status": "EXCEPTION",
                "exception": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc().splitlines()[-25:],
            }
        )
        _write_json(json_path, payload)
        print(json.dumps({"json": _rel(json_path), "status": payload["status"], "exception": payload["exception"]}, ensure_ascii=False), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    raise SystemExit(main())
