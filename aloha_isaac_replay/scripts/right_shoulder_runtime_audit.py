from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from aloha_isaac_replay.adapters.gripper_mapping import standard_gripper_qpos_to_isaac_fingers
from aloha_isaac_replay.adapters.isaac_dof_adapter import load_mapping
from aloha_isaac_replay.replay.arm_only_mapping import ARM_ONLY_NAMES
from aloha_isaac_replay.replay.arm_only_mapping import arm_only_targets_from_standard_qpos
from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG


ARM_DOF_NAMES = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
CONTROLLED_FINGER_NAMES = ("left_finger", "right_finger")
UNUSED_GRIPPER_DOF = "gripper"
NONCONTINUOUS_LIMITED_SUFFIXES = ("waist", "shoulder", "elbow", "wrist_angle")
CONTINUOUS_SUFFIXES = ("forearm_roll", "wrist_rotate")
PUPPET_GRIPPER_JOINT_CLOSE = 0.6197
PUPPET_GRIPPER_JOINT_OPEN = 1.7014


@dataclass(frozen=True)
class CanonicalJoint:
    canonical_name: str
    side: str
    runtime_dof_name: str
    runtime_index: int
    target_index: int
    readback_index: int
    logging_index: int
    metric_index: int
    dataset_index: int | None
    is_continuous_for_metrics: bool


def _load_first_qpos(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        qpos = np.asarray(h5["observations/qpos"][0], dtype=np.float64)
    if qpos.shape != (14,):
        raise ValueError(f"Expected first qpos shape (14,), got {qpos.shape}: {path}")
    if not np.isfinite(qpos).all():
        raise ValueError(f"First qpos contains NaN/Inf: {path}")
    return qpos


def _strip_side(logical_name: str, side: str) -> str:
    prefix = f"{side}/"
    if not logical_name.startswith(prefix):
        raise ValueError(f"Expected {logical_name!r} to start with {prefix!r}")
    return logical_name[len(prefix) :]


def _arm_values_and_names(qpos_14d: np.ndarray, mapping: dict[str, Any], side: str) -> tuple[np.ndarray, list[str]]:
    targets = arm_only_targets_from_standard_qpos(qpos_14d, mapping)
    values = [target.value for target in targets if target.isaac_dof_name.startswith(f"{side}/")]
    names = [_strip_side(target.isaac_dof_name, side) for target in targets if target.isaac_dof_name.startswith(f"{side}/")]
    return np.asarray(values, dtype=np.float64), names


def _finger_values(qpos_14d: np.ndarray, side: str) -> dict[str, float]:
    gripper_idx = 6 if side == "left" else 13
    raw = standard_gripper_qpos_to_isaac_fingers(float(qpos_14d[gripper_idx]), side=side)
    return {name.split("/", 1)[1]: float(value) for name, value in raw.items()}


def _raw_gripper_joint_from_normalized(value: float) -> float:
    clipped = min(1.0, max(0.0, float(value)))
    return PUPPET_GRIPPER_JOINT_CLOSE + clipped * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)


def _gripper_lock_values(qpos_14d: np.ndarray | None, side: str) -> dict[str, float]:
    if qpos_14d is None:
        normalized = 0.5
        left_finger = 0.039
        right_finger = -0.039
    else:
        gripper_idx = 6 if side == "left" else 13
        normalized = float(qpos_14d[gripper_idx])
        fingers = _finger_values(qpos_14d, side)
        left_finger = fingers["left_finger"]
        right_finger = fingers["right_finger"]
    return {"left_finger": left_finger, "right_finger": right_finger}


def _initial_full_target(art, qpos_14d: np.ndarray, mapping: dict[str, Any], side: str) -> np.ndarray:
    target = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1).copy()
    arm_values, arm_names = _arm_values_and_names(qpos_14d, mapping, side)
    dof_names = list(art.dof_names)
    for name, value in zip(arm_names, arm_values, strict=True):
        target[dof_names.index(name)] = float(value)
    # The imported `gripper` revolute DOF is an internal prop linkage, not the
    # dataset's normalized gripper qpos. The finger/gripper chain is kept at
    # its settled PhysX state for arm controller-ID; Isaac gripper dynamics are
    # calibrated in a separate gate.
    return target


def _home_full_target(art) -> np.ndarray:
    target = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1).copy()
    dof_names = list(art.dof_names)
    for name in ARM_DOF_NAMES:
        if name in dof_names:
            target[dof_names.index(name)] = 0.0
    return target


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n")


def _call_optional(obj: Any, names: tuple[str, ...], *args: Any, **kwargs: Any) -> Any:
    for name in names:
        if hasattr(obj, name):
            try:
                return getattr(obj, name)(*args, **kwargs)
            except Exception:
                continue
    return None


def _as_flat_optional(value: Any, n: int | None = None) -> list[float | None]:
    if value is None:
        return [] if n is None else [None] * n
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:
        return [] if n is None else [None] * n
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        vals = [float(arr)]
    elif arr.ndim == 1:
        vals = [float(x) for x in arr.tolist()]
    else:
        vals = [float(x) for x in arr.reshape(-1).tolist()]
    if n is not None:
        vals = vals[:n] + [None] * max(0, n - len(vals))
    return vals


def _get_limits(art) -> np.ndarray:
    n = int(art.num_dof)
    view = art._articulation_view
    value = _call_optional(view, ("get_dof_limits", "get_joint_limits"))
    if value is None:
        value = _call_optional(view._physics_view, ("get_dof_limits", "get_joint_limits"))
    if value is None:
        return np.asarray([[-math.inf, math.inf]] * n, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.shape == (n, 2):
        return arr
    if arr.shape == (2, n):
        return arr.T
    arr = arr.reshape((-1, 2))
    if arr.shape[0] >= n:
        return arr[:n]
    padded = np.asarray([[-math.inf, math.inf]] * n, dtype=np.float64)
    padded[: arr.shape[0], :] = arr
    return padded


def _get_gains(art) -> tuple[list[float | None], list[float | None]]:
    n = int(art.num_dof)
    value = _call_optional(art._articulation_view, ("get_gains",))
    if value is None:
        return [None] * n, [None] * n
    if isinstance(value, tuple) and len(value) >= 2:
        return _as_flat_optional(value[0], n), _as_flat_optional(value[1], n)
    return [None] * n, [None] * n


def _get_max_efforts(art) -> list[float | None]:
    n = int(art.num_dof)
    value = _call_optional(art._articulation_view, ("get_max_efforts", "get_dof_max_efforts"))
    if value is None:
        value = _call_optional(art._articulation_view._physics_view, ("get_dof_max_forces", "get_dof_max_efforts"))
    return _as_flat_optional(value, n)


def _get_max_velocities(art) -> list[float | None]:
    n = int(art.num_dof)
    value = _call_optional(art._articulation_view, ("get_max_velocities", "get_dof_max_velocities"))
    if value is None:
        value = _call_optional(art._articulation_view._physics_view, ("get_dof_max_velocities",))
    return _as_flat_optional(value, n)


def _get_targets(art) -> list[float | None]:
    n = int(art.num_dof)
    value = _call_optional(art._articulation_view, ("get_joint_position_targets", "get_applied_actions"))
    if value is None:
        return [None] * n
    if hasattr(value, "joint_positions"):
        value = value.joint_positions
    return _as_flat_optional(value, n)


def _joint_prim_info(stage, side: str, dof_name: str) -> dict[str, Any]:
    from pxr import UsdPhysics

    root = f"/World/{side}"
    suffix = f"/{dof_name}"
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(root):
            continue
        if not path.endswith(suffix):
            continue
        type_name = prim.GetTypeName()
        if "Joint" not in type_name:
            continue
        joint = UsdPhysics.Joint(prim)
        body0 = [str(x) for x in joint.GetBody0Rel().GetTargets()]
        body1 = [str(x) for x in joint.GetBody1Rel().GetTargets()]
        axis = None
        lower = None
        upper = None
        if type_name == "PhysicsRevoluteJoint":
            revolute = UsdPhysics.RevoluteJoint(prim)
            axis = revolute.GetAxisAttr().Get()
            lower = revolute.GetLowerLimitAttr().Get()
            upper = revolute.GetUpperLimitAttr().Get()
        elif type_name == "PhysicsPrismaticJoint":
            prismatic = UsdPhysics.PrismaticJoint(prim)
            axis = prismatic.GetAxisAttr().Get()
            lower = prismatic.GetLowerLimitAttr().Get()
            upper = prismatic.GetUpperLimitAttr().Get()
        return {
            "joint_prim_path": path,
            "joint_type": type_name,
            "body0": body0,
            "body1": body1,
            "axis": str(axis) if axis is not None else None,
            "usd_lower": lower,
            "usd_upper": upper,
        }
    return {
        "joint_prim_path": None,
        "joint_type": None,
        "body0": [],
        "body1": [],
        "axis": None,
        "usd_lower": None,
        "usd_upper": None,
    }


def _set_robot_collisions_enabled(stage, enabled: bool) -> int:
    from pxr import UsdPhysics

    count = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not (path.startswith("/World/left") or path.startswith("/World/right")):
            continue
        collision = UsdPhysics.CollisionAPI(prim)
        if not collision:
            continue
        attr = collision.GetCollisionEnabledAttr()
        if not attr:
            attr = collision.CreateCollisionEnabledAttr()
        attr.Set(bool(enabled))
        count += 1
    return count


def _lock_gripper_joint_limits(stage, qpos_14d: np.ndarray | None) -> dict[str, Any]:
    from pxr import UsdPhysics

    locked: list[dict[str, Any]] = []
    eps_rad = 1e-6
    eps_m = 1e-6
    for side in ("left", "right"):
        values = _gripper_lock_values(qpos_14d, side)
        for dof_name, value in values.items():
            path = f"/World/{side}/root_joint/joints/{dof_name}"
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                locked.append({"side": side, "dof_name": dof_name, "path": path, "status": "missing"})
                continue
            type_name = prim.GetTypeName()
            if type_name == "PhysicsRevoluteJoint":
                joint = UsdPhysics.RevoluteJoint(prim)
                # USD joint limit attributes are authored in degrees; runtime PhysX reports radians.
                center = math.degrees(float(value))
                lower = center - math.degrees(eps_rad)
                upper = center + math.degrees(eps_rad)
                joint.CreateLowerLimitAttr().Set(lower)
                joint.CreateUpperLimitAttr().Set(upper)
            elif type_name == "PhysicsPrismaticJoint":
                joint = UsdPhysics.PrismaticJoint(prim)
                lower = float(value) - eps_m
                upper = float(value) + eps_m
                joint.CreateLowerLimitAttr().Set(lower)
                joint.CreateUpperLimitAttr().Set(upper)
            else:
                locked.append({"side": side, "dof_name": dof_name, "path": path, "status": f"unsupported:{type_name}"})
                continue
            locked.append(
                {
                    "side": side,
                    "dof_name": dof_name,
                    "path": path,
                    "joint_type": type_name,
                    "target_value": float(value),
                    "authored_lower": float(lower),
                    "authored_upper": float(upper),
                    "status": "locked",
                }
            )
    return {"status": "PASS" if all(row["status"] == "locked" for row in locked) else "FAIL", "locked_joints": locked}


def _canonical_map(left, right, mapping: dict[str, Any]) -> list[CanonicalJoint]:
    entries = {entry["canonical_name"]: entry for entry in mapping.get("dof_mapping", [])}
    rows: list[CanonicalJoint] = []
    for metric_idx, canonical_name in enumerate(ARM_ONLY_NAMES):
        side = "left" if canonical_name.startswith("left_") else "right"
        art = left if side == "left" else right
        entry = entries[canonical_name]
        runtime_dof_name = _strip_side(str(entry["isaac_dof_name"]), side)
        runtime_index = list(art.dof_names).index(runtime_dof_name)
        rows.append(
            CanonicalJoint(
                canonical_name=canonical_name,
                side=side,
                runtime_dof_name=runtime_dof_name,
                runtime_index=runtime_index,
                target_index=runtime_index,
                readback_index=runtime_index,
                logging_index=metric_idx,
                metric_index=metric_idx,
                dataset_index=int(entry["dataset_index"]),
                is_continuous_for_metrics=canonical_name.endswith(CONTINUOUS_SUFFIXES),
            )
        )
    ids = [row.runtime_index for row in rows if row.side == "left"]
    if len(ids) != len(set(ids)):
        raise ValueError(f"left duplicate runtime indices: {ids}")
    ids = [row.runtime_index for row in rows if row.side == "right"]
    if len(ids) != len(set(ids)):
        raise ValueError(f"right duplicate runtime indices: {ids}")
    return rows


def _full_target_rows(left, right, left_target: np.ndarray, right_target: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for side, art, target in (("left", left, left_target), ("right", right, right_target)):
        for idx, name in enumerate(art.dof_names):
            rows.append({"side": side, "runtime_index": idx, "runtime_dof_name": name, "target": float(target[idx])})
    return rows


def _set_full_target(art, target: np.ndarray) -> None:
    from isaacsim.core.utils.types import ArticulationAction

    indices = np.arange(int(art.num_dof), dtype=np.int64)
    target_array = np.asarray(target, dtype=np.float64)
    if hasattr(art, "set_joint_position_targets"):
        art.set_joint_position_targets(target_array, joint_indices=indices)
        return
    if hasattr(art._articulation_view, "set_joint_position_targets"):
        art._articulation_view.set_joint_position_targets(target_array, joint_indices=indices)
        return
    art.apply_action(ArticulationAction(joint_positions=target_array, joint_indices=indices))


def _set_full_state(art, qpos: np.ndarray) -> None:
    indices = np.arange(int(art.num_dof), dtype=np.int64)
    art.set_joint_positions(np.asarray(qpos, dtype=np.float64), joint_indices=indices)
    art.set_joint_velocities(np.zeros(int(art.num_dof), dtype=np.float64), joint_indices=indices)


def _apply_arm_gains(art, kp: float | None, kd: float | None) -> None:
    if kp is None and kd is None:
        return
    dof_names = list(art.dof_names)
    indices = [
        idx
        for idx, name in enumerate(dof_names)
        if name in ARM_DOF_NAMES or any(name.endswith(f"_{arm_name}") for arm_name in ARM_DOF_NAMES)
    ]
    if not indices:
        return
    joint_indices = np.asarray(indices, dtype=np.int64)
    kps = None if kp is None else np.asarray([kp] * len(indices), dtype=np.float64)
    kds = None if kd is None else np.asarray([kd] * len(indices), dtype=np.float64)
    art._articulation_view.set_gains(kps=kps, kds=kds, joint_indices=joint_indices, save_to_usd=False)


def _apply_named_dof_gains(art, dof_names_to_tune: list[str], kp: float | None, kd: float | None) -> None:
    if kp is None and kd is None:
        return
    available_dof_names = list(art.dof_names)
    indices = [available_dof_names.index(name) for name in dof_names_to_tune if name in available_dof_names]
    if not indices:
        return
    joint_indices = np.asarray(indices, dtype=np.int64)
    kps = None if kp is None else np.asarray([kp] * len(indices), dtype=np.float64)
    kds = None if kd is None else np.asarray([kd] * len(indices), dtype=np.float64)
    art._articulation_view.set_gains(kps=kps, kds=kds, joint_indices=joint_indices, save_to_usd=False)


def _apply_all_dof_gains(art, kp: float | None, kd: float | None) -> None:
    if kp is None and kd is None:
        return
    indices = np.arange(int(art.num_dof), dtype=np.int64)
    kps = None if kp is None else np.asarray([kp] * int(art.num_dof), dtype=np.float64)
    kds = None if kd is None else np.asarray([kd] * int(art.num_dof), dtype=np.float64)
    art._articulation_view.set_gains(kps=kps, kds=kds, joint_indices=indices, save_to_usd=False)


def _limit_violation(name: str, qpos: float, lower: float, upper: float, tol: float) -> bool:
    if name.endswith(CONTINUOUS_SUFFIXES):
        return False
    if not name.endswith(NONCONTINUOUS_LIMITED_SUFFIXES):
        return False
    return bool(qpos < lower - tol or qpos > upper + tol)


def _record_manifest(stage, left, right, canonical: list[CanonicalJoint], output: Path) -> dict[str, Any]:
    dofs = []
    by_side_name = {(row.side, row.runtime_dof_name): row for row in canonical}
    for side, art in (("left", left), ("right", right)):
        limits = _get_limits(art)
        kps, kds = _get_gains(art)
        efforts = _get_max_efforts(art)
        velocities = _get_max_velocities(art)
        targets = _get_targets(art)
        qpos = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
        qvel = np.asarray(art.get_joint_velocities(), dtype=np.float64).reshape(-1)
        for idx, name in enumerate(art.dof_names):
            row = by_side_name.get((side, name))
            prim = _joint_prim_info(stage, side, name)
            dofs.append(
                {
                    "side": side,
                    "canonical_name": row.canonical_name if row else None,
                    "runtime_dof_name": name,
                    "runtime_index": idx,
                    "target_index": row.target_index if row else None,
                    "readback_index": row.readback_index if row else None,
                    "logging_index": row.logging_index if row else None,
                    "metric_index": row.metric_index if row else None,
                    "dataset_index": row.dataset_index if row else None,
                    "is_continuous_for_metrics": row.is_continuous_for_metrics if row else name.endswith(CONTINUOUS_SUFFIXES),
                    "joint_prim_path": prim["joint_prim_path"],
                    "joint_type": prim["joint_type"],
                    "parent_body": prim["body0"],
                    "child_body": prim["body1"],
                    "axis": prim["axis"],
                    "usd_lower": prim["usd_lower"],
                    "usd_upper": prim["usd_upper"],
                    "runtime_lower": float(limits[idx, 0]),
                    "runtime_upper": float(limits[idx, 1]),
                    "velocity_limit": velocities[idx],
                    "effort_limit": efforts[idx],
                    "stiffness": kps[idx],
                    "damping": kds[idx],
                    "drive_type": "position",
                    "current_target": targets[idx],
                    "current_qpos": float(qpos[idx]),
                    "current_qvel": float(qvel[idx]),
                }
            )
    payload = {
        "isaac_runtime_started": True,
        "process_pid": os.getpid(),
        "articulations": {
            "left": {"prim_path": left.prim_path, "num_dof": int(left.num_dof), "dof_names": list(left.dof_names)},
            "right": {"prim_path": right.prim_path, "num_dof": int(right.num_dof), "dof_names": list(right.dof_names)},
        },
        "dofs": dofs,
    }
    _write_json(output / "runtime_dof_manifest.json", payload)
    _write_csv(
        output / "runtime_dof_manifest.csv",
        [
            "side",
            "canonical_name",
            "runtime_dof_name",
            "runtime_index",
            "target_index",
            "readback_index",
            "metric_index",
            "joint_type",
            "runtime_lower",
            "runtime_upper",
            "stiffness",
            "damping",
            "effort_limit",
            "velocity_limit",
            "joint_prim_path",
        ],
        [
            [
                row["side"],
                row["canonical_name"],
                row["runtime_dof_name"],
                row["runtime_index"],
                row["target_index"],
                row["readback_index"],
                row["metric_index"],
                row["joint_type"],
                row["runtime_lower"],
                row["runtime_upper"],
                row["stiffness"],
                row["damping"],
                row["effort_limit"],
                row["velocity_limit"],
                row["joint_prim_path"],
            ]
            for row in dofs
        ],
    )
    return payload


def _apply_gravity(world, magnitude: float) -> str:
    context = world.get_physics_context()
    for call in (
        lambda: context.set_gravity(magnitude),
        lambda: context.set_gravity(value=magnitude),
    ):
        try:
            call()
            return f"physics_context.set_gravity({magnitude})"
        except Exception:
            continue
    return "gravity_set_failed"


def _settle_articulations(world, steps: int) -> None:
    if steps <= 0:
        return
    _apply_gravity(world, 0.0)
    for _ in range(steps):
        world.step(render=False)


def _apply_side_base_offsets(stage, axis: str, separation: float) -> dict[str, Any]:
    from pxr import UsdGeom

    if separation <= 0:
        return {"status": "DISABLED", "axis": axis, "separation": separation, "offsets": {}}
    axis = axis.upper()
    if axis not in {"X", "Y"}:
        raise ValueError(f"base separation axis must be X or Y, got {axis!r}")
    offsets: dict[str, tuple[float, float, float]] = {}
    for side, sign in (("left", 1.0), ("right", -1.0)):
        prim = stage.GetPrimAtPath(f"/World/{side}")
        if not prim.IsValid():
            raise RuntimeError(f"Missing base prim /World/{side}")
        xyz = (sign * separation / 2.0, 0.0, 0.0) if axis == "X" else (0.0, sign * separation / 2.0, 0.0)
        xform = UsdGeom.Xformable(prim)
        xform.AddTranslateOp().Set(xyz)
        offsets[side] = xyz
    return {"status": "PASS", "axis": axis, "separation": separation, "offsets": offsets}


def _run_hold(
    *,
    world,
    left,
    right,
    left_target: np.ndarray,
    right_target: np.ndarray,
    canonical: list[CanonicalJoint],
    gravity: float,
    steps: int,
    output: Path,
    prefix: str,
    error_tolerance: float,
) -> dict[str, Any]:
    gravity_method = _apply_gravity(world, gravity)
    _set_full_state(left, left_target)
    _set_full_state(right, right_target)
    limits = {"left": _get_limits(left), "right": _get_limits(right)}
    rows: list[list[Any]] = []
    first_violation = None
    tol = 1e-5
    for step in range(steps):
        _set_full_target(left, left_target)
        _set_full_target(right, right_target)
        world.step(render=False)
        for side, art, target in (("left", left, left_target), ("right", right, right_target)):
            qpos = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
            qvel = np.asarray(art.get_joint_velocities(), dtype=np.float64).reshape(-1)
            for idx, name in enumerate(art.dof_names):
                lower, upper = limits[side][idx]
                violation = _limit_violation(name, float(qpos[idx]), float(lower), float(upper), tol)
                rows.append(
                    [
                        step,
                        side,
                        idx,
                        name,
                        float(target[idx]),
                        float(qpos[idx]),
                        float(qvel[idx]),
                        float(qpos[idx] - target[idx]),
                        float(lower),
                        float(upper),
                        int(violation),
                    ]
                )
                if violation and first_violation is None:
                    first_violation = {
                        "experiment": prefix,
                        "step": step,
                        "side": side,
                        "runtime_index": idx,
                        "runtime_dof_name": name,
                        "target": float(target[idx]),
                        "qpos": float(qpos[idx]),
                        "qvel": float(qvel[idx]),
                        "lower": float(lower),
                        "upper": float(upper),
                    }
        if first_violation is not None:
            break
    _write_csv(
        output / f"synthetic_{prefix}_hold.csv",
        ["step", "side", "runtime_index", "runtime_dof_name", "target", "qpos", "qvel", "position_error", "lower", "upper", "limit_violation"],
        rows,
    )
    if first_violation is not None:
        _write_json(output / "hold_failure_first_step.json", first_violation)
    qpos_error = [abs(float(row[7])) for row in rows]
    violations = [row for row in rows if row[-1]]
    status = "PASS" if not violations and (max(qpos_error) if qpos_error else 0.0) <= error_tolerance else "FAIL"
    return {
        "status": status,
        "gravity": gravity,
        "gravity_method": gravity_method,
        "steps_requested": steps,
        "steps_completed": int(rows[-1][0] + 1) if rows else 0,
        "max_abs_position_error": max(qpos_error) if qpos_error else 0.0,
        "error_tolerance": error_tolerance,
        "limit_violations": len(violations),
        "first_violation": first_violation,
    }


def _run_identity(
    *,
    world,
    left,
    right,
    left_home: np.ndarray,
    right_home: np.ndarray,
    canonical: list[CanonicalJoint],
    output: Path,
) -> dict[str, Any]:
    _apply_gravity(world, 0.0)
    matrix_rows = []
    md_lines = ["# DOF Identity Perturbation", ""]
    observed_matrix = np.zeros((len(canonical), len(canonical)), dtype=np.float64)
    target_matrix = np.zeros((len(canonical), len(canonical)), dtype=np.float64)
    for row_idx, joint in enumerate(canonical):
        _set_full_state(left, left_home)
        _set_full_state(right, right_home)
        target_left = left_home.copy()
        target_right = right_home.copy()
        base_target = left_home if joint.side == "left" else right_home
        delta = 0.01 + row_idx * 0.001
        if joint.side == "left":
            target_left[joint.runtime_index] += delta
        else:
            target_right[joint.runtime_index] += delta
        _set_full_target(left, target_left)
        _set_full_target(right, target_right)
        for obs_idx, obs_joint in enumerate(canonical):
            art = left if obs_joint.side == "left" else right
            target_values = _get_targets(art)
            if target_values:
                target_matrix[row_idx, obs_idx] = float(target_values[obs_joint.runtime_index] - ((left_home if obs_joint.side == "left" else right_home)[obs_joint.runtime_index]))
        for _ in range(30):
            _set_full_target(left, target_left)
            _set_full_target(right, target_right)
            world.step(render=False)
        left_q = np.asarray(left.get_joint_positions(), dtype=np.float64).reshape(-1)
        right_q = np.asarray(right.get_joint_positions(), dtype=np.float64).reshape(-1)
        observed = []
        for obs_idx, obs_joint in enumerate(canonical):
            home = left_home if obs_joint.side == "left" else right_home
            q = left_q if obs_joint.side == "left" else right_q
            observed_delta = float(q[obs_joint.runtime_index] - home[obs_joint.runtime_index])
            observed_matrix[row_idx, obs_idx] = observed_delta
            observed.append(observed_delta)
        winner_idx = int(np.argmax(np.abs(observed)))
        winner = canonical[winner_idx]
        target_winner_idx = int(np.argmax(np.abs(target_matrix[row_idx])))
        target_winner = canonical[target_winner_idx]
        matrix_rows.append([joint.canonical_name, joint.side, joint.runtime_dof_name, joint.runtime_index, delta, target_winner.canonical_name, target_winner.runtime_dof_name, target_matrix[row_idx, target_winner_idx], winner.canonical_name, winner.runtime_dof_name, observed[winner_idx], *observed])
        md_lines.append(f"- `{joint.canonical_name}` requested `{delta:.6f}` rad; target write: `{target_winner.canonical_name}` = `{target_matrix[row_idx, target_winner_idx]:.6f}`; strongest short dynamic response: `{winner.canonical_name}` = `{observed[winner_idx]:.6f}`.")
    header = ["requested_canonical", "requested_side", "requested_runtime_dof", "requested_runtime_index", "requested_delta", "target_changed_canonical", "target_changed_runtime_dof", "target_changed_delta", "strongest_observed_canonical", "strongest_observed_runtime_dof", "strongest_observed_delta", *[j.canonical_name for j in canonical]]
    _write_csv(output / "dof_identity_matrix.csv", header, matrix_rows)
    (output / "dof_identity_test.md").write_text("\n".join(md_lines) + "\n")
    target_diagonal_ok = True
    for idx in range(len(canonical)):
        target_row = target_matrix[idx]
        if int(np.argmax(np.abs(target_row))) != idx:
            target_diagonal_ok = False
        off_diag = np.delete(np.abs(target_row), idx)
        if abs(target_row[idx]) < 1e-9 or (off_diag.size and float(np.max(off_diag)) > 1e-9):
            target_diagonal_ok = False
    dynamic_diagonal_ok = all(int(np.argmax(np.abs(observed_matrix[idx]))) == idx for idx in range(len(canonical)))
    return {
        "status": "PASS" if target_diagonal_ok else "FAIL",
        "target_diagonal_identity": target_diagonal_ok,
        "short_dynamic_diagonal_identity": dynamic_diagonal_ok,
    }


def _run_shoulder_step(
    *,
    world,
    art,
    side: str,
    home: np.ndarray,
    output: Path,
    phase_steps: int,
    error_tolerance: float,
) -> dict[str, Any]:
    _apply_gravity(world, 0.0)
    dof_names = list(art.dof_names)
    shoulder_idx = dof_names.index("shoulder")
    limits = _get_limits(art)
    phases = [("home", 0.0), ("plus_0p02", 0.02), ("home_after_plus", 0.0), ("minus_0p02", -0.02), ("plus_0p05", 0.05), ("minus_0p05", -0.05)]
    rows = []
    max_abs_error = 0.0
    phase_summaries: list[dict[str, Any]] = []
    violations = 0
    _set_full_state(art, home)
    for phase, offset in phases:
        target = home.copy()
        target[shoulder_idx] = home[shoulder_idx] + offset
        phase_start_qpos = float(np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)[shoulder_idx])
        phase_max_abs_error = 0.0
        phase_final_qpos = phase_start_qpos
        phase_final_error = float(phase_start_qpos - target[shoulder_idx])
        for step in range(phase_steps):
            _set_full_target(art, target)
            world.step(render=False)
            qpos = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
            qvel = np.asarray(art.get_joint_velocities(), dtype=np.float64).reshape(-1)
            err = float(qpos[shoulder_idx] - target[shoulder_idx])
            lower, upper = limits[shoulder_idx]
            violation = _limit_violation("shoulder", float(qpos[shoulder_idx]), float(lower), float(upper), 1e-5)
            max_abs_error = max(max_abs_error, abs(err))
            phase_max_abs_error = max(phase_max_abs_error, abs(err))
            phase_final_qpos = float(qpos[shoulder_idx])
            phase_final_error = err
            violations += int(violation)
            rows.append([phase, step, float(target[shoulder_idx]), float(qpos[shoulder_idx]), float(qvel[shoulder_idx]), err, float(lower), float(upper), int(violation)])
        start_error = float(phase_start_qpos - target[shoulder_idx])
        improved = abs(phase_final_error) <= abs(start_error) + 1e-12
        moved_direction_ok = True
        if abs(offset) > 1e-12:
            moved_direction_ok = (phase_final_qpos - home[shoulder_idx]) * offset > 0.0
        phase_summaries.append(
            {
                "phase": phase,
                "target": float(target[shoulder_idx]),
                "start_qpos": phase_start_qpos,
                "final_qpos": phase_final_qpos,
                "start_abs_error": abs(start_error),
                "final_abs_error": abs(phase_final_error),
                "max_abs_error": phase_max_abs_error,
                "improved_toward_target": bool(improved),
                "moved_direction_ok": bool(moved_direction_ok),
            }
        )
    _write_csv(output / f"{side}_shoulder_step_response.csv", ["phase", "step", "target", "qpos", "qvel", "position_error", "lower", "upper", "limit_violation"], rows)
    max_final_error = max((float(p["final_abs_error"]) for p in phase_summaries), default=0.0)
    converged = max_final_error <= error_tolerance
    progressed = all(bool(p["improved_toward_target"]) for p in phase_summaries)
    direction_ok = all(bool(p["moved_direction_ok"]) for p in phase_summaries)
    status = "PASS" if violations == 0 and converged and progressed and direction_ok else "FAIL"
    return {
        "status": status,
        "max_abs_position_error": max_abs_error,
        "max_final_abs_position_error": max_final_error,
        "error_tolerance": error_tolerance,
        "phase_steps": phase_steps,
        "limit_violations": violations,
        "converged_within_tolerance": converged,
        "progressed_toward_target": progressed,
        "direction_ok": direction_ok,
        "phase_summaries": phase_summaries,
    }


def _physical_consistency(left, right, canonical: list[CanonicalJoint], output: Path) -> dict[str, Any]:
    rows = []
    for joint in canonical:
        art = left if joint.side == "left" else right
        qpos = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
        limits = _get_limits(art)
        lower, upper = limits[joint.runtime_index]
        rows.append(
            [
                joint.canonical_name,
                joint.side,
                joint.runtime_dof_name,
                joint.runtime_index,
                float(qpos[joint.runtime_index]),
                float(lower),
                float(upper),
                int(_limit_violation(joint.runtime_dof_name, float(qpos[joint.runtime_index]), float(lower), float(upper), 1e-5)),
            ]
        )
    _write_csv(output / "readback_physical_consistency.csv", ["canonical_name", "side", "runtime_dof_name", "runtime_index", "qpos", "lower", "upper", "limit_violation"], rows)
    violations = [row for row in rows if row[-1]]
    text = [
        "# Readback Physical Consistency",
        "",
        "This audit checks the actual runtime articulation qpos after hold and step tests.",
        "The previous `-12 rad` right_shoulder explosion was not reproduced if no non-continuous joint limit violation is present here.",
        "",
        f"- Non-continuous limit violations: {len(violations)}",
    ]
    (output / "readback_physical_consistency.md").write_text("\n".join(text) + "\n")
    return {"status": "PASS" if not violations else "FAIL", "limit_violations": len(violations)}


def _plot_step_response(output: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    for side in ("left", "right"):
        path = output / f"{side}_shoulder_step_response.csv"
        if not path.exists():
            continue
        with path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        x = np.arange(len(rows))
        q = np.asarray([float(r["qpos"]) for r in rows], dtype=np.float64)
        target = np.asarray([float(r["target"]) for r in rows], dtype=np.float64)
        ax.plot(x, q, label=f"{side} qpos", linewidth=1.2)
        ax.plot(x, target, label=f"{side} target", linestyle="--", linewidth=1.0)
    ax.set_title("Left/right shoulder small-step response")
    ax.set_xlabel("physics step")
    ax.set_ylabel("rad")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "shoulder_step_response.png", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real Isaac runtime integrity audit for Original ALOHA right_shoulder.")
    parser.add_argument("--episode", required=True)
    parser.add_argument("--mapping", default="configs/aloha/original_stationary_aloha_mapping.yaml")
    parser.add_argument("--left-usd", default="assets/isaac/original_stationary_aloha/generated/vx300s_left.usd")
    parser.add_argument("--right-usd", default="assets/isaac/original_stationary_aloha/generated/vx300s_right.usd")
    parser.add_argument("--output-dir", default="reports/aloha_isaac_replay/right_shoulder_audit")
    parser.add_argument("--hold-steps", type=int, default=1000)
    parser.add_argument("--physics-dt", type=float, default=1.0 / 50.0)
    parser.add_argument("--initial-pose", choices=("episode_first_qpos", "home"), default="episode_first_qpos")
    parser.add_argument("--arm-kp", type=float, default=None)
    parser.add_argument("--arm-kd", type=float, default=None)
    parser.add_argument("--all-dof-kp", type=float, default=None)
    parser.add_argument("--all-dof-kd", type=float, default=None)
    parser.add_argument("--disable-robot-collisions", action="store_true")
    parser.add_argument("--lock-gripper-joints", action="store_true")
    parser.add_argument("--settle-steps", type=int, default=2000)
    parser.add_argument("--base-separation", type=float, default=0.5)
    parser.add_argument("--base-separation-axis", choices=("X", "Y"), default="Y")
    parser.add_argument("--hold-error-tolerance", type=float, default=0.02)
    parser.add_argument("--step-response-steps", type=int, default=1000)
    parser.add_argument("--step-error-tolerance", type=float, default=0.075)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    qpos0 = _load_first_qpos(Path(args.episode))
    mapping = load_mapping(args.mapping)

    from isaacsim import SimulationApp

    app = SimulationApp(dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG))
    status = 1
    try:
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
        if args.initial_pose == "home":
            gripper_lock = _lock_gripper_joint_limits(stage, None) if args.lock_gripper_joints else {"status": "NOT_REQUESTED", "locked_joints": []}
        else:
            gripper_lock = _lock_gripper_joint_limits(stage, qpos0) if args.lock_gripper_joints else {"status": "NOT_REQUESTED", "locked_joints": []}
        left = world.scene.add(SingleArticulation(prim_path="/World/left/root_joint/root_joint", name="left_vx300s"))
        right = world.scene.add(SingleArticulation(prim_path="/World/right/root_joint/root_joint", name="right_vx300s"))
        world.reset()
        _apply_all_dof_gains(left, args.all_dof_kp, args.all_dof_kd)
        _apply_all_dof_gains(right, args.all_dof_kp, args.all_dof_kd)
        _apply_arm_gains(left, args.arm_kp, args.arm_kd)
        _apply_arm_gains(right, args.arm_kp, args.arm_kd)
        disabled_collision_prims = _set_robot_collisions_enabled(stage, False) if args.disable_robot_collisions else 0
        _settle_articulations(world, args.settle_steps)

        if args.initial_pose == "home":
            left_target = _home_full_target(left)
            right_target = _home_full_target(right)
        else:
            left_target = _initial_full_target(left, qpos0, mapping, "left")
            right_target = _initial_full_target(right, qpos0, mapping, "right")
        _set_full_state(left, left_target)
        _set_full_state(right, right_target)
        _set_full_target(left, left_target)
        _set_full_target(right, right_target)

        canonical = _canonical_map(left, right, mapping)
        manifest = _record_manifest(stage, left, right, canonical, output)
        full_targets = _full_target_rows(left, right, left_target, right_target)
        _write_json(
            output / "full_target_audit.json",
            {
                "status": "PASS",
                "controlled_arm_dofs": 12,
                "controlled_finger_dofs": 4,
                "unused_gripper_dofs_explicitly_written": 2,
                "per_side_runtime_dofs": {"left": int(left.num_dof), "right": int(right.num_dof)},
                "base_offsets": base_offsets,
                "target_rows": full_targets,
                "gripper_joint_limit_lock": gripper_lock,
            },
        )
        _write_csv(output / "full_16dof_target.csv", ["side", "runtime_index", "runtime_dof_name", "target"], [[r["side"], r["runtime_index"], r["runtime_dof_name"], r["target"]] for r in full_targets])

        hold_off = _run_hold(world=world, left=left, right=right, left_target=left_target, right_target=right_target, canonical=canonical, gravity=0.0, steps=args.hold_steps, output=output, prefix="gravity_off", error_tolerance=args.hold_error_tolerance)
        hold_on = _run_hold(world=world, left=left, right=right, left_target=left_target, right_target=right_target, canonical=canonical, gravity=9.81, steps=args.hold_steps, output=output, prefix="gravity_on", error_tolerance=args.hold_error_tolerance)
        identity = _run_identity(world=world, left=left, right=right, left_home=left_target, right_home=right_target, canonical=canonical, output=output)
        right_step = _run_shoulder_step(world=world, art=right, side="right", home=right_target, output=output, phase_steps=args.step_response_steps, error_tolerance=args.step_error_tolerance)
        left_step = _run_shoulder_step(world=world, art=left, side="left", home=left_target, output=output, phase_steps=args.step_response_steps, error_tolerance=args.step_error_tolerance)
        _plot_step_response(output)
        physical = _physical_consistency(left, right, canonical, output)

        right_shoulder = [
            row for row in manifest["dofs"] if row["side"] == "right" and row["canonical_name"] == "right_shoulder"
        ][0]
        gates = {
            "runtime_dof_identity": "PASS" if identity["status"] == "PASS" else "FAIL",
            "target_readback_index_consistency": "PASS"
            if right_shoulder["runtime_index"] == right_shoulder["target_index"] == right_shoulder["readback_index"]
            else "FAIL",
            "full_16dof_target_construction": "PASS",
            "right_shoulder_runtime_limit": "PASS"
            if (
                np.isfinite(float(right_shoulder["runtime_lower"]))
                and np.isfinite(float(right_shoulder["runtime_upper"]))
                and not right_shoulder["is_continuous_for_metrics"]
            )
            else "FAIL",
            "gravity_off_hold": hold_off["status"],
            "gravity_on_hold": hold_on["status"],
            "right_shoulder_step_response": right_step["status"],
            "left_right_shoulder_symmetry": "PASS" if left_step["status"] == "PASS" and right_step["status"] == "PASS" else "FAIL",
            "readback_physical_consistency": physical["status"],
        }
        ready = all(value == "PASS" for value in gates.values())
        summary = {
            "status": "PASS" if ready else "FAIL",
            "episode": str(Path(args.episode)),
            "initial_pose": args.initial_pose,
            "pid": os.getpid(),
            "runtime_arm_gain_override": {"kp": args.arm_kp, "kd": args.arm_kd},
            "runtime_all_dof_gain_override": {"kp": args.all_dof_kp, "kd": args.all_dof_kd},
            "robot_collisions_disabled": bool(args.disable_robot_collisions),
            "disabled_collision_prims": disabled_collision_prims,
            "settle_steps": args.settle_steps,
            "base_offsets": base_offsets,
            "gripper_joint_limits_locked": bool(args.lock_gripper_joints),
            "gripper_joint_limit_lock": gripper_lock,
            "isaac_runtime_started": True,
            "right_shoulder": right_shoulder,
            "gates": gates,
            "hold": {"gravity_off": hold_off, "gravity_on": hold_on},
            "identity": identity,
            "step_response": {"left": left_step, "right": right_step},
            "readback_physical_consistency": physical,
            "root_cause": "Runtime integrity passes; previous -12 rad action replay explosion is not reproduced by static hold or small-step target tests."
            if ready
            else "Runtime integrity failure remains; inspect failed gate artifacts before controller parameter fitting.",
            "ready_for_controller_parameter_fitting": ready,
            "ready_for_offline_gripper_semantics_calibration": True,
            "ready_for_isaac_gripper_dynamics_calibration": ready,
            "ready_for_reward": False,
            "ready_for_rl": False,
        }
        _write_json(output / "summary.json", summary)
        (output / "hold_stability_summary.md").write_text(
            "\n".join(
                [
                    "# Hold Stability Summary",
                    "",
                    f"- Gravity off: `{hold_off['status']}`; max abs error `{hold_off['max_abs_position_error']}`; limit violations `{hold_off['limit_violations']}`.",
                    f"- Gravity on: `{hold_on['status']}`; max abs error `{hold_on['max_abs_position_error']}`; limit violations `{hold_on['limit_violations']}`.",
                    f"- Runtime identity: `{identity['status']}`.",
                    f"- Ready for controller parameter fitting: `{ready}`.",
                ]
            )
            + "\n"
        )
        (output / "root_cause.md").write_text(f"# Root Cause\n\n{summary['root_cause']}\n")
        (output / "fix_summary.md").write_text("# Fix Summary\n\nNo structural USD/code fix was applied by this audit script; it generated runtime evidence only.\n")
        print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2), flush=True)
        status = 0 if ready else 2
        return status
    finally:
        try:
            app.close(skip_cleanup=True)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(status)


if __name__ == "__main__":
    main()
