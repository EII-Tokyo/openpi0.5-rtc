from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from aloha_isaac_replay.adapters.isaac_dof_adapter import load_mapping
from aloha_isaac_replay.controller_system_id.continuous_joints import nearest_equivalent_targets
from aloha_isaac_replay.replay.arm_only_mapping import ARM_ONLY_NAMES
from aloha_isaac_replay.replay.arm_only_mapping import arm_only_targets_from_standard_qpos
from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG


def _load_episode(path: Path, max_steps: int | None) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        qpos = h5["observations/qpos"][:]
        action = h5["action"][:]
    if qpos.ndim != 2 or qpos.shape[1] != 14:
        raise ValueError(f"Expected qpos shape (T, 14), got {qpos.shape}")
    if action.ndim != 2 or action.shape[1] != 14:
        raise ValueError(f"Expected action shape (T, 14), got {action.shape}")
    steps = min(len(action), len(qpos) - 1)
    if max_steps is not None:
        steps = min(steps, max_steps)
    qpos = qpos[: steps + 1]
    action = action[:steps]
    if not np.isfinite(qpos).all() or not np.isfinite(action).all():
        raise ValueError(f"Episode contains NaN/Inf: {path}")
    return qpos, action


def _side_name(logical_name: str, side: str) -> str:
    prefix = f"{side}/"
    if not logical_name.startswith(prefix):
        raise ValueError(f"Expected {logical_name!r} to start with {prefix!r}")
    return logical_name[len(prefix) :]


def _resolve_indices(actual_dof_names: list[str], mapped_names: list[str], side: str) -> list[int]:
    stripped = [_side_name(name, side) for name in mapped_names]
    missing = [name for name in stripped if name not in actual_dof_names]
    if missing:
        raise ValueError(f"{side} missing DOFs {missing}; actual={actual_dof_names}")
    indices = [actual_dof_names.index(name) for name in stripped]
    if len(indices) != len(set(indices)):
        raise ValueError(f"{side} duplicate DOF indices: {indices}")
    return indices


def _arm_values(frame_14d: np.ndarray, mapping: dict, side: str) -> tuple[np.ndarray, list[str]]:
    targets = arm_only_targets_from_standard_qpos(frame_14d, mapping)
    values = [target.value for target in targets if target.isaac_dof_name.startswith(f"{side}/")]
    names = [target.isaac_dof_name for target in targets if target.isaac_dof_name.startswith(f"{side}/")]
    return np.asarray(values, dtype=np.float64), names


def _write_csv(path: Path, header: list[str], rows: list[list[float | int]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _plot_errors(path: Path, errors: np.ndarray, title: str, ylabel: str = "sim qpos - real qpos") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(errors.shape[0])
    for idx, name in enumerate(ARM_ONLY_NAMES):
        ax.plot(x, errors[:, idx], linewidth=1.0, label=name)
    ax.set_xlabel("action step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _per_joint_metrics(errors: np.ndarray) -> dict[str, dict[str, float]]:
    payload = {}
    for idx, name in enumerate(ARM_ONLY_NAMES):
        e = errors[:, idx]
        payload[name] = {
            "rmse": float(np.sqrt(np.mean(np.square(e)))),
            "mae": float(np.mean(np.abs(e))),
            "max_abs": float(np.max(np.abs(e))),
            "bias": float(np.mean(e)),
        }
    return payload


def _set_arm_gains(art, indices: list[int], kp: float | None, kd: float | None) -> None:
    if kp is None and kd is None:
        return
    joint_indices = np.asarray(indices, dtype=np.int64)
    kps = None if kp is None else np.asarray([kp] * len(indices), dtype=np.float64)
    kds = None if kd is None else np.asarray([kd] * len(indices), dtype=np.float64)
    art._articulation_view.set_gains(kps=kps, kds=kds, joint_indices=joint_indices, save_to_usd=False)


def _apply_base_offsets(stage, usd_geom, axis: str, separation: float) -> dict[str, object]:
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
        usd_geom.Xformable(prim).AddTranslateOp().Set(xyz)
        offsets[side] = xyz
    return {"status": "PASS", "axis": axis, "separation": separation, "offsets": offsets}


def _get_limits(art, indices: list[int]) -> np.ndarray:
    raw = art._articulation_view.get_dof_limits()
    limits = np.asarray(raw[0] if getattr(raw, "ndim", 0) == 3 else raw, dtype=np.float64)
    return limits[np.asarray(indices, dtype=np.int64)]


def _limit_violations(values: np.ndarray, limits: np.ndarray, names: list[str], tol: float = 1e-5) -> list[dict[str, object]]:
    violations = []
    for idx, name in enumerate(names):
        lower, upper = limits[idx]
        value = float(values[idx])
        if np.isfinite(lower) and value < float(lower) - tol:
            violations.append({"joint": name, "value": value, "lower": float(lower), "upper": float(upper), "kind": "lower"})
        if np.isfinite(upper) and value > float(upper) + tol:
            violations.append({"joint": name, "value": value, "lower": float(lower), "upper": float(upper), "kind": "upper"})
    return violations


def _lag_metrics(predicted: np.ndarray, real_qpos: np.ndarray, max_lag: int) -> dict[str, object]:
    rows = []
    for lag in range(max_lag + 1):
        samples = min(len(predicted), len(real_qpos) - lag)
        if samples <= 0:
            continue
        pred = predicted[:samples]
        ref = real_qpos[lag : lag + samples]
        if len(pred) == 0:
            continue
        err = pred - ref
        rows.append(
            {
                "lag_steps": lag,
                "rmse": float(np.sqrt(np.mean(np.square(err)))),
                "mae": float(np.mean(np.abs(err))),
                "samples": int(len(pred)),
            }
        )
    best = min(rows, key=lambda row: row["rmse"]) if rows else None
    return {"rows": rows, "best_by_rmse": best}


def _load_known_joint_limits() -> dict[str, dict[str, float]]:
    path = Path("reports/aloha_isaac_replay/dof_properties_probe.json")
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    limits = {}
    for side in ("left", "right"):
        for item in payload:
            name = f"{side}_{item['name']}"
            if name in ARM_ONLY_NAMES:
                limits[name] = {
                    "lower": float(item["lower"]),
                    "upper": float(item["upper"]),
                    "max_velocity": float(item["maxVelocity"]),
                    "max_effort": float(item["maxEffort"]),
                }
    return limits


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay HDF5 action targets through Isaac articulation controllers.")
    parser.add_argument("--episode", required=True)
    parser.add_argument("--mapping", default="configs/aloha/original_stationary_aloha_mapping.yaml")
    parser.add_argument("--left-usd", default="assets/isaac/original_stationary_aloha/generated/vx300s_left.usd")
    parser.add_argument("--right-usd", default="assets/isaac/original_stationary_aloha/generated/vx300s_right.usd")
    parser.add_argument("--left-prim-path", default="/World/left/root_joint/root_joint")
    parser.add_argument("--right-prim-path", default="/World/right/root_joint/root_joint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--physics-dt", type=float, default=1.0 / 50.0)
    parser.add_argument("--steps-per-action", type=int, default=1)
    parser.add_argument("--max-lag-steps", type=int, default=15)
    parser.add_argument("--arm-kp", type=float, default=None)
    parser.add_argument("--arm-kd", type=float, default=None)
    parser.add_argument("--base-separation", type=float, default=0.5)
    parser.add_argument("--base-separation-axis", choices=("X", "Y"), default="Y")
    args = parser.parse_args()

    from isaacsim import SimulationApp

    app = SimulationApp(dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG))
    try:
        import isaacsim.core.utils.stage as stage_utils
        import omni.usd
        from pxr import UsdGeom
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction

        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        qpos, action = _load_episode(Path(args.episode), args.max_steps)
        mapping = load_mapping(args.mapping)

        World.clear_instance()
        stage_utils.create_new_stage()
        world = World(stage_units_in_meters=1.0, backend="numpy", device="cpu")
        world.set_simulation_dt(physics_dt=args.physics_dt, rendering_dt=args.physics_dt)
        stage_utils.add_reference_to_stage(usd_path=str(Path(args.left_usd).resolve()), prim_path="/World/left")
        stage_utils.add_reference_to_stage(usd_path=str(Path(args.right_usd).resolve()), prim_path="/World/right")
        base_offsets = _apply_base_offsets(omni.usd.get_context().get_stage(), UsdGeom, args.base_separation_axis, args.base_separation)
        left = world.scene.add(SingleArticulation(prim_path=args.left_prim_path, name="left_vx300s"))
        right = world.scene.add(SingleArticulation(prim_path=args.right_prim_path, name="right_vx300s"))
        world.reset()

        left_init, left_names = _arm_values(qpos[0], mapping, "left")
        right_init, right_names = _arm_values(qpos[0], mapping, "right")
        left_indices = _resolve_indices(list(left.dof_names), left_names, "left")
        right_indices = _resolve_indices(list(right.dof_names), right_names, "right")
        left_idx_array = np.asarray(left_indices, dtype=np.int64)
        right_idx_array = np.asarray(right_indices, dtype=np.int64)
        _set_arm_gains(left, left_indices, args.arm_kp, args.arm_kd)
        _set_arm_gains(right, right_indices, args.arm_kp, args.arm_kd)
        left_limits = _get_limits(left, left_indices)
        right_limits = _get_limits(right, right_indices)

        left.set_joint_positions(left_init, joint_indices=left_idx_array)
        right.set_joint_positions(right_init, joint_indices=right_idx_array)
        left.set_joint_velocities(np.zeros_like(left_init), joint_indices=left_idx_array)
        right.set_joint_velocities(np.zeros_like(right_init), joint_indices=right_idx_array)

        raw_action_rows: list[list[float | int]] = []
        canonical_rows: list[list[float | int]] = []
        nearest_rows: list[list[float | int]] = []
        drive_rows: list[list[float | int]] = []
        real_pre_rows: list[list[float | int]] = []
        sim_rows: list[list[float | int]] = []
        target_rows: list[list[float | int]] = []
        best_error_rows: list[list[float | int]] = []
        predicted = []
        raw_targets = []
        nearest_targets = []
        real_qpos_sequence = arm_values = []
        wrap_events: list[dict[str, object]] = []
        limit_violations: list[dict[str, object]] = []

        for step_idx, action_frame in enumerate(action):
            left_target, _ = _arm_values(action_frame, mapping, "left")
            right_target, _ = _arm_values(action_frame, mapping, "right")
            raw_target_arm = np.concatenate([left_target, right_target]).astype(np.float64)
            real_pre_arm = np.concatenate([qpos[step_idx, :6], qpos[step_idx, 7:13]]).astype(np.float64)
            left_pre = left.get_joint_positions(joint_indices=left_idx_array)
            right_pre = right.get_joint_positions(joint_indices=right_idx_array)
            sim_pre_arm = np.concatenate([left_pre, right_pre]).astype(np.float64)
            nearest_target_arm, events = nearest_equivalent_targets(raw_target_arm, sim_pre_arm, ARM_ONLY_NAMES)
            if events:
                wrap_events.append({"step": int(step_idx), "joints": events})
            left_drive = nearest_target_arm[:6]
            right_drive = nearest_target_arm[6:]
            left.apply_action(ArticulationAction(joint_positions=left_drive, joint_indices=left_idx_array))
            right.apply_action(ArticulationAction(joint_positions=right_drive, joint_indices=right_idx_array))
            for _ in range(args.steps_per_action):
                world.step(render=False)
            left_pred = left.get_joint_positions(joint_indices=left_idx_array)
            right_pred = right.get_joint_positions(joint_indices=right_idx_array)
            sim_arm = np.concatenate([left_pred, right_pred]).astype(np.float64)
            for item in _limit_violations(left_pred, left_limits, ARM_ONLY_NAMES[:6]):
                limit_violations.append({"step": int(step_idx), **item})
            for item in _limit_violations(right_pred, right_limits, ARM_ONLY_NAMES[6:]):
                limit_violations.append({"step": int(step_idx), **item})
            target_arm = raw_target_arm
            predicted.append(sim_arm)
            raw_targets.append(raw_target_arm)
            nearest_targets.append(nearest_target_arm)
            raw_action_rows.append([step_idx, *raw_target_arm.tolist()])
            canonical_rows.append([step_idx, *target_arm.tolist()])
            nearest_rows.append([step_idx, *nearest_target_arm.tolist()])
            drive_rows.append([step_idx, *nearest_target_arm.tolist()])
            real_pre_rows.append([step_idx, *real_pre_arm.tolist()])
            sim_rows.append([step_idx, *sim_arm.tolist()])
            target_rows.append([step_idx, *target_arm.tolist()])

        predicted_arr = np.asarray(predicted, dtype=np.float64)
        raw_targets_arr = np.asarray(raw_targets, dtype=np.float64)
        nearest_targets_arr = np.asarray(nearest_targets, dtype=np.float64)
        real_qpos_arr = np.concatenate([qpos[:, :6], qpos[:, 7:13]], axis=1).astype(np.float64)
        lag_scan = _lag_metrics(predicted_arr, real_qpos_arr, args.max_lag_steps)
        best_lag = int(lag_scan["best_by_rmse"]["lag_steps"]) if lag_scan["best_by_rmse"] else 0
        samples = min(len(predicted_arr), len(real_qpos_arr) - best_lag)
        errors = predicted_arr[:samples] - real_qpos_arr[best_lag : best_lag + samples]
        for step_idx, err in enumerate(errors):
            best_error_rows.append([step_idx, *err.tolist()])
        header = ["step", *ARM_ONLY_NAMES]
        _write_csv(output / "raw_action_arm.csv", header, raw_action_rows)
        _write_csv(output / "canonical_absolute_target_arm.csv", header, canonical_rows)
        _write_csv(output / "nearest_equivalent_target_arm.csv", header, nearest_rows)
        _write_csv(output / "isaac_drive_target_arm.csv", header, drive_rows)
        _write_csv(output / "real_pre_action_qpos_arm.csv", header, real_pre_rows)
        _write_csv(output / "sim_qpos_arm.csv", header, sim_rows)
        _write_csv(output / "action_target_arm.csv", header, target_rows)
        _write_csv(output / "action_replay_error_arm.csv", header, best_error_rows)
        _plot_errors(
            output / "action_replay_error_arm.png",
            errors,
            f"Corrected arm action replay error at lag {best_lag}",
            ylabel=f"sim qpos[t] - real qpos[t+{best_lag}]",
        )

        metrics = {
            "status": "ANALYZED",
            "episode": args.episode,
            "frames_qpos": int(qpos.shape[0]),
            "steps_replayed": int(action.shape[0]),
            "mode": "corrected_arm_absolute_position_targets_only",
            "hdf5_action_space": "standard_aloha_like_raw_runtime_command",
            "additional_openpi_transform_applied": False,
            "action_type": "absolute_follower_joint_target",
            "continuous_nearest_equivalent": "forearm_roll and wrist_rotate only, relative to current Isaac qpos",
            "uses_controller": True,
            "uses_action": True,
            "uses_gripper_action": False,
            "compare_to": f"HDF5 observations/qpos[t+{best_lag}] arm dimensions after delay scan 0..{args.max_lag_steps}",
            "physics_dt": args.physics_dt,
            "steps_per_action": args.steps_per_action,
            "base_offsets": base_offsets,
            "arm_kp": args.arm_kp,
            "arm_kd": args.arm_kd,
            "arm_rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "arm_mae": float(np.mean(np.abs(errors))),
            "arm_max_abs": float(np.max(np.abs(errors))),
            "per_joint": _per_joint_metrics(errors),
            "lag_scan": lag_scan,
            "best_lag_steps": best_lag,
            "raw_target_range": {
                name: {"min": float(np.min(raw_targets_arr[:, idx])), "max": float(np.max(raw_targets_arr[:, idx]))}
                for idx, name in enumerate(ARM_ONLY_NAMES)
            },
            "nearest_target_range": {
                name: {"min": float(np.min(nearest_targets_arr[:, idx])), "max": float(np.max(nearest_targets_arr[:, idx]))}
                for idx, name in enumerate(ARM_ONLY_NAMES)
            },
            "known_joint_limits": _load_known_joint_limits(),
            "wrap_events": wrap_events,
            "joint_limit_violations": len(limit_violations),
            "first_joint_limit_violation": limit_violations[0] if limit_violations else None,
            "action_replay_no_explosion": len(limit_violations) == 0,
            "left_dof_names": list(left.dof_names),
            "right_dof_names": list(right.dof_names),
            "left_indices": left_indices,
            "right_indices": right_indices,
            "gripper_action_note": "action[6]/action[13] are intentionally excluded because qpos/action gripper spaces differ; see gripper_semantics.md",
        }
        (output / "action_replay_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return 0
    finally:
        app.close(skip_cleanup=True)


if __name__ == "__main__":
    raise SystemExit(main())
