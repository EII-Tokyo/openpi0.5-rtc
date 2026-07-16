from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from aloha_isaac_replay.controller_system_id.action_semantics import canonical_absolute_targets
from aloha_isaac_replay.controller_system_id.action_semantics import arm_qpos_from_raw_hdf5_qpos
from aloha_isaac_replay.controller_system_id.action_semantics import canonical_arm_names
from aloha_isaac_replay.controller_system_id.continuous_joints import CONTINUOUS_JOINT_SUFFIXES
from aloha_isaac_replay.controller_system_id.delay_scan import scan_action_qpos_delays
from aloha_isaac_replay.controller_system_id.offline_models import evaluate_offline_models
from aloha_isaac_replay.controller_system_id.offline_models import split_episode_ids
from aloha_isaac_replay.controller_system_id.right_arm_hold import analyze_right_arm_hold
from aloha_isaac_replay.controller_system_id.right_arm_hold import summarize_right_arm_hold


def _episode_id(path: str | Path) -> str:
    p = Path(path)
    if p.parent.name.startswith("key_region_"):
        return p.parent.name
    return p.stem


def _load_episode(path: str | Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        qpos_14d = np.asarray(h5["observations/qpos"][:], dtype=np.float64)
        action_14d = np.asarray(h5["action"][:], dtype=np.float64)
        fps = float(h5.attrs.get("fps", 50.0))
    qpos = arm_qpos_from_raw_hdf5_qpos(qpos_14d)
    action = canonical_absolute_targets(action_14d)
    return {
        "id": _episode_id(path),
        "path": str(path),
        "fps": fps,
        "qpos": qpos,
        "action": action,
        "qpos_14d_shape": list(qpos_14d.shape),
        "action_14d_shape": list(action_14d.shape),
    }


def _load_selected(path: Path, limit: int) -> list[str]:
    payload = json.loads(path.read_text())
    selected = [item["path"] for item in payload["selected"]]
    return selected[:limit]


def _weighted_delay_scan(episodes: list[dict[str, Any]], max_delay: int, joint_names: tuple[str, ...]) -> dict[str, Any]:
    per_episode = []
    per_delay_errors: dict[int, list[np.ndarray]] = {delay: [] for delay in range(max_delay + 1)}
    per_joint_errors: dict[str, dict[int, list[np.ndarray]]] = {
        name: {delay: [] for delay in range(max_delay + 1)} for name in joint_names
    }
    per_joint_raw_errors: dict[str, dict[int, list[np.ndarray]]] = {
        name: {delay: [] for delay in range(max_delay + 1)} for name in joint_names
    }
    for episode in episodes:
        scan = scan_action_qpos_delays(episode["action"], episode["qpos"], max_delay=max_delay, joint_names=joint_names)
        per_episode.append({"episode_id": episode["id"], "path": episode["path"], "scan": scan})
        for delay in range(max_delay + 1):
            samples = min(episode["action"].shape[0], episode["qpos"].shape[0] - delay)
            if samples <= 0:
                continue
            pred = episode["action"][:samples]
            ref = episode["qpos"][delay : delay + samples]
            err = pred - ref
            per_delay_errors[delay].append(err)
            for idx, name in enumerate(joint_names):
                per_joint_errors[name][delay].append(err[:, [idx]])
                per_joint_raw_errors[name][delay].append(err[:, [idx]])

    rows = []
    for delay, parts in per_delay_errors.items():
        if not parts:
            continue
        err = np.concatenate(parts, axis=0)
        rows.append(
            {
                "delay": delay,
                "samples": int(err.shape[0]),
                "rmse": float(np.sqrt(np.mean(np.square(err)))),
                "mae": float(np.mean(np.abs(err))),
                "max_abs": float(np.max(np.abs(err))),
            }
        )
    best = min(rows, key=lambda row: row["rmse"])
    per_joint = {}
    for name in joint_names:
        joint_rows = []
        for delay, parts in per_joint_errors[name].items():
            if not parts:
                continue
            err = np.concatenate(parts, axis=0).reshape(-1)
            joint_rows.append(
                {
                    "delay": delay,
                    "samples": int(err.shape[0]),
                    "rmse": float(np.sqrt(np.mean(np.square(err)))),
                    "mae": float(np.mean(np.abs(err))),
                    "max_abs": float(np.max(np.abs(err))),
                }
            )
        best_joint = min(joint_rows, key=lambda row: row["rmse"])
        per_joint[name] = {"best_delay": int(best_joint["delay"]), "best": best_joint, "rows": joint_rows}
    return {
        "range": [0, max_delay],
        "aggregate": {"best_delay": int(best["delay"]), "best": best, "rows": rows},
        "per_joint": per_joint,
        "per_episode": per_episode,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _flatten_hold_row(row: dict[str, Any]) -> dict[str, Any]:
    reference_stats = row.get("reference_stats") or {}
    return {
        "episode_id": row["episode_id"],
        "frames": row["frames"],
        "phase": row["phase"],
        "reward": row["reward"],
        "right_arm_hold_detected": row["right_arm_hold_detected"],
        "right_arm_static_command_detected": row["right_arm_static_command_detected"],
        "right_arm_hold_or_static_detected": row["right_arm_hold_or_static_detected"],
        "usable_for_right_arm_controller_id": row["usable_for_right_arm_controller_id"],
        "right_arm_hold_max_abs_after_transition": row["right_arm_hold_max_abs_after_transition"],
        "right_arm_action_joint_std_mean_after_transition": row["right_arm_action_joint_std_mean_after_transition"],
        "right_arm_action_reference_max_abs_diff": row["right_arm_action_reference_max_abs_diff"],
        "right_shoulder_action_min": row["right_shoulder_action_min"],
        "right_shoulder_action_max": row["right_shoulder_action_max"],
        "right_shoulder_action_std": row["right_shoulder_action_std"],
        "right_shoulder_qpos_min": row["right_shoulder_qpos_min"],
        "right_shoulder_qpos_max": row["right_shoulder_qpos_max"],
        "right_shoulder_reference_min": reference_stats.get("right_shoulder_reference_min"),
        "right_shoulder_reference_max": reference_stats.get("right_shoulder_reference_max"),
        "path": row["path"],
    }


def _write_markdown_reports(
    out: Path,
    summary: dict[str, Any],
    delay_scan: dict[str, Any],
    offline: dict[str, Any],
    right_arm_hold_rows: list[dict[str, Any]],
    right_arm_hold_summary: dict[str, Any],
) -> None:
    best = delay_scan["aggregate"]["best"]
    lines = [
        "# Corrected Adapter",
        "",
        "Raw HDF5 arm action is used directly as a standard ALOHA absolute follower joint target.",
        "",
        "Forbidden transforms in this stage:",
        "",
        "- no OpenPI `adapt_to_pi` sign flip;",
        "- no delta integration;",
        "- no gripper action;",
        "- no default `qpos[t+1]` comparison;",
        "- no shoulder/elbow wrap.",
        "",
        "The comparison scans `action[t]` against `qpos[t+d]`, because `qpos[t]` is observed before `action[t]` is emitted.",
    ]
    (out / "corrected_adapter.md").write_text("\n".join(lines) + "\n")

    delay_lines = [
        "# Delay Scan",
        "",
        f"Delay range: `{delay_scan['range'][0]}..{delay_scan['range'][1]}` frames.",
        f"Aggregate best delay: `{delay_scan['aggregate']['best_delay']}` frames.",
        f"Aggregate best RMSE: `{best['rmse']:.6f}` rad.",
        "",
        "Per-joint best delays:",
        "",
        "| joint | best delay | RMSE | max abs |",
        "|---|---:|---:|---:|",
    ]
    for name, payload in delay_scan["per_joint"].items():
        row = payload["best"]
        delay_lines.append(f"| {name} | {row['delay']} | {row['rmse']:.6f} | {row['max_abs']:.6f} |")
    (out / "delay_scan.md").write_text("\n".join(delay_lines) + "\n")

    (out / "continuous_joint_handling.md").write_text(
        "# Continuous Joint Handling\n\n"
        f"Only joints ending in `{CONTINUOUS_JOINT_SUFFIXES}` are eligible for nearest-equivalent wrapping.\n\n"
        "Shoulder and elbow joints are never wrapped; a large shoulder/elbow error remains a blocker rather than being hidden by angle wrapping.\n"
    )

    baseline_lines = [
        "# Corrected Baseline",
        "",
        "Corrected baseline uses raw absolute HDF5 arm action and compares to future qpos after delay scan.",
        "",
        f"- Aggregate best delay: `{delay_scan['aggregate']['best_delay']}` frames",
        f"- Corrected baseline RMSE: `{best['rmse']:.6f}` rad",
        f"- Corrected baseline MAE: `{best['mae']:.6f}` rad",
        f"- Corrected baseline max abs: `{best['max_abs']:.6f}` rad",
        "",
        "Important qualification:",
        "",
        "- Selected actor key-region episodes hold the right arm after takeover.",
        "- The low offline right-arm error is therefore partly a hold-stability statistic, not evidence that these episodes excite the right-arm controller dynamics.",
        f"- Right-arm exact hold detected: `{right_arm_hold_summary['right_arm_hold_detected_count']}/{right_arm_hold_summary['episode_count']}` episodes.",
        f"- Right-arm hold/static command detected: `{right_arm_hold_summary['right_arm_hold_or_static_detected_count']}/{right_arm_hold_summary['episode_count']}` episodes.",
        f"- Usable right-arm controller-ID episodes: `{right_arm_hold_summary['right_arm_controller_id_usable_count']}`.",
    ]
    (out / "corrected_baseline.md").write_text("\n".join(baseline_lines) + "\n")

    hold_lines = [
        "# RLT Right Arm Hold Audit",
        "",
        "The RLT key-region runtime can freeze right arm dimensions to the robot state latched at actor takeover. This is not the same as zeroing the action; right_shoulder often appears near zero because the latched shoulder state is near zero. Some episodes are not exactly equal to the entry qpos, but still carry almost no right-arm command variation after takeover.",
        "",
        f"- Episodes checked: `{right_arm_hold_summary['episode_count']}`",
        f"- Exact hold detected: `{right_arm_hold_summary['right_arm_hold_detected_count']}`",
        f"- Hold/static command detected: `{right_arm_hold_summary['right_arm_hold_or_static_detected_count']}`",
        f"- Usable for right-arm controller ID: `{right_arm_hold_summary['right_arm_controller_id_usable_count']}`",
        f"- Max hold error after transition: `{right_arm_hold_summary.get('max_hold_error_after_transition', 0.0):.9f}`",
        f"- Status: `{right_arm_hold_summary['status']}`",
        "",
        "| episode | exact hold | static cmd | usable for right-arm ID | max hold error | right_shoulder action range | right_shoulder qpos range |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in right_arm_hold_rows:
        hold_lines.append(
            f"| {row['episode_id']} | {row['right_arm_hold_detected']} | "
            f"{row['right_arm_static_command_detected']} | "
            f"{row['usable_for_right_arm_controller_id']} | "
            f"{row['right_arm_hold_max_abs_after_transition']:.9f} | "
            f"[{row['right_shoulder_action_min']:.6f}, {row['right_shoulder_action_max']:.6f}] | "
            f"[{row['right_shoulder_qpos_min']:.6f}, {row['right_shoulder_qpos_max']:.6f}] |"
        )
    (out / "rlt_right_arm_hold_audit.md").write_text("\n".join(hold_lines) + "\n")

    offline_lines = [
        "# Offline Model Fit",
        "",
        f"Common delay used: `{offline['common_delay']}` frames.",
        "",
        "| split | M0 RMSE | M1 RMSE | M2 RMSE | M3 RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("identification", "validation", "heldout"):
        row = offline[split]
        offline_lines.append(
            f"| {split} | {row['M0']['rmse']:.6f} | {row['M1']['rmse']:.6f} | "
            f"{row['M2']['rmse']:.6f} | {row['M3']['rmse']:.6f} |"
        )
    (out / "offline_model_fit.md").write_text("\n".join(offline_lines) + "\n")

    (out / "single_joint_response.md").write_text(
        "# Single Joint Response\n\n"
        "Not run in this pass. Required next Isaac stage: hold all other joints fixed and test ±0.05 rad, ±0.15 rad, and low-frequency sine per joint.\n"
    )
    isaac_lines = [
        "# Isaac Controller Fit",
        "",
        "No controller parameter tuning was performed in this pass.",
        "",
    ]
    isaac_first = summary.get("isaac_corrected_first_episode")
    if isaac_first:
        isaac_lines += [
            "Corrected adapter was run once in Isaac for the first selected episode.",
            "",
            f"- RMSE: `{isaac_first['arm_rmse']:.6f}` rad",
            f"- right_shoulder max error: `{isaac_first['right_shoulder_max_error']:.6f}` rad",
            f"- right_shoulder target range: `{isaac_first['right_shoulder_target_range']}`",
            f"- right_shoulder sim range: `{isaac_first['right_shoulder_sim_range']}`",
            "",
            "This remains blocked before stiffness/damping tuning: the target is near zero, but the Isaac readback runs far outside the URDF shoulder limit.",
        ]
    else:
        isaac_lines.append(
            "Isaac corrected action replay was not run. Per the gate, stiffness/damping/effort/friction remain untouched until corrected adapter and delay scan are accepted."
        )
    (out / "isaac_controller_fit.md").write_text("\n".join(isaac_lines) + "\n")
    (out / "heldout_validation.md").write_text(
        "# Held-out Validation\n\n"
        f"Held-out offline M3 RMSE: `{offline['heldout']['M3']['rmse']:.6f}` rad.\n"
    )
    blocker_lines = [
        "# Blockers",
        "",
        "- Ready for contact/reward: `NO`.",
        "- Ready for RL: `NO`.",
        "- Gripper command calibration is separate and not fit in this pass.",
    ]
    if summary.get("isaac_runtime_gate") == "BLOCKED_ISAAC_RIGHT_SHOULDER":
        blocker_lines += [
            "- Isaac corrected action replay still has a right_shoulder limit-scale failure.",
            "- Do not tune stiffness/damping/friction yet; first run single-joint step response and inspect drive/limit enforcement.",
        ]
    if summary.get("right_arm_data_gate") == "BLOCKED_RLT_RIGHT_ARM_HOLD_OR_STATIC_COMMAND":
        blocker_lines += [
            "- Selected actor key-region data holds or nearly freezes the right-arm command after takeover.",
            "- Do not use these episodes as right-arm excitation data for controller parameter fitting.",
        ]
    if (
        summary.get("isaac_runtime_gate") != "BLOCKED_ISAAC_RIGHT_SHOULDER"
        and summary.get("right_arm_data_gate") != "BLOCKED_RLT_RIGHT_ARM_HOLD_OR_STATIC_COMMAND"
    ):
        blocker_lines.append("- Next pass should start with 50 Hz target hold and delay queue before controller tuning.")
    (out / "blockers.md").write_text("\n".join(blocker_lines) + "\n")


def _load_isaac_corrected_first_episode(out: Path) -> dict[str, Any] | None:
    metrics_path = out / "isaac_corrected_first_episode" / "action_replay_metrics.json"
    if not metrics_path.exists():
        return None
    metrics = json.loads(metrics_path.read_text())
    raw_range = metrics["raw_target_range"]["right_shoulder"]
    sim_csv = out / "isaac_corrected_first_episode" / "sim_qpos_arm.csv"
    sim_values = []
    if sim_csv.exists():
        with sim_csv.open() as f:
            reader = csv.DictReader(f)
            sim_values = [float(row["right_shoulder"]) for row in reader]
    sim_range = {
        "min": float(np.min(sim_values)) if sim_values else None,
        "max": float(np.max(sim_values)) if sim_values else None,
    }
    return {
        "arm_rmse": float(metrics["arm_rmse"]),
        "arm_mae": float(metrics["arm_mae"]),
        "arm_max_abs": float(metrics["arm_max_abs"]),
        "best_lag_steps": int(metrics["best_lag_steps"]),
        "right_shoulder_max_error": float(metrics["per_joint"]["right_shoulder"]["max_abs"]),
        "right_shoulder_rmse": float(metrics["per_joint"]["right_shoulder"]["rmse"]),
        "right_shoulder_target_range": raw_range,
        "right_shoulder_sim_range": sim_range,
        "status": metrics["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Original ALOHA controller system-id analysis.")
    parser.add_argument("--selected", default="reports/aloha_isaac_replay/selected_success_hdf5.json")
    parser.add_argument("--output-dir", default="reports/aloha_isaac_replay/controller_system_id")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-delay", type=int, default=15)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    episode_paths = _load_selected(Path(args.selected), args.limit)
    episodes = [_load_episode(path) for path in episode_paths]
    right_arm_hold_rows = [analyze_right_arm_hold(path) for path in episode_paths]
    right_arm_hold_summary = summarize_right_arm_hold(right_arm_hold_rows)
    joint_names = canonical_arm_names()

    delay_scan = _weighted_delay_scan(episodes, args.max_delay, joint_names)
    common_delay = int(delay_scan["aggregate"]["best_delay"])
    episodes_by_id = {episode["id"]: {"qpos": episode["qpos"], "action": episode["action"]} for episode in episodes}
    splits = split_episode_ids([episode["id"] for episode in episodes])
    offline = evaluate_offline_models(episodes_by_id, splits, common_delay=common_delay)

    right_shoulder = delay_scan["per_joint"]["right_shoulder"]["best"]
    isaac_corrected = _load_isaac_corrected_first_episode(out)
    right_arm_data_gate = right_arm_hold_summary["status"]
    isaac_runtime_gate = "NOT_RUN"
    gate = "PASS_CORRECTED_OFFLINE_BASELINE"
    if right_arm_data_gate == "BLOCKED_RLT_RIGHT_ARM_HOLD_OR_STATIC_COMMAND":
        gate = "BLOCKED_RLT_RIGHT_ARM_HOLD_OR_STATIC_DATA"
    if float(right_shoulder["max_abs"]) >= 3.5:
        gate = "BLOCKED_RIGHT_SHOULDER"
    if isaac_corrected and float(isaac_corrected["right_shoulder_max_error"]) >= 3.5:
        isaac_runtime_gate = "BLOCKED_ISAAC_RIGHT_SHOULDER"
        gate = (
            "BLOCKED_RLT_RIGHT_ARM_HOLD_OR_STATIC_DATA_AND_ISAAC_RIGHT_SHOULDER_RUNTIME"
            if right_arm_data_gate == "BLOCKED_RLT_RIGHT_ARM_HOLD_OR_STATIC_COMMAND"
            else "BLOCKED_ISAAC_RIGHT_SHOULDER"
        )
    elif isaac_corrected:
        isaac_runtime_gate = "PASS_CORRECTED_ISAAC_ACTION_REPLAY"

    summary = {
        "hdf5_arm_action": "standard ALOHA-like raw HDF5 action arm dimensions",
        "additional_openpi_transform_applied": False,
        "action_type": "absolute_follower_joint_target",
        "qpos_action_pairing": "qpos[t] observed before action[t]; compare action[t] to qpos[t+d]",
        "delay_scan_range": [0, int(args.max_delay)],
        "per_joint_delays": {name: payload["best_delay"] for name, payload in delay_scan["per_joint"].items()},
        "effective_common_delay": common_delay,
        "continuous_joint_handling": "nearest-equivalent only for forearm_roll and wrist_rotate",
        "right_shoulder_max_error": float(right_shoulder["max_abs"]),
        "joint_limit_violations": "not evaluated against Isaac runtime limits in pure NumPy pass",
        "corrected_baseline_rmse": float(delay_scan["aggregate"]["best"]["rmse"]),
        "offline_m0_rmse": float(offline["heldout"]["M0"]["rmse"]),
        "offline_m1_rmse": float(offline["heldout"]["M1"]["rmse"]),
        "offline_m2_rmse": float(offline["heldout"]["M2"]["rmse"]),
        "offline_m3_rmse": float(offline["heldout"]["M3"]["rmse"]),
        "isaac_fitted_rmse": None,
        "isaac_corrected_first_episode": isaac_corrected,
        "right_arm_hold_summary": right_arm_hold_summary,
        "right_arm_data_gate": right_arm_data_gate,
        "isaac_runtime_gate": isaac_runtime_gate,
        "validation_rmse": float(offline["validation"]["M3"]["rmse"]),
        "heldout_rmse": float(offline["heldout"]["M3"]["rmse"]),
        "uses_gripper_action": False,
        "ready_for_gripper_command_calibration": True,
        "ready_for_contact_reward": False,
        "ready_for_rl": False,
        "gate": gate,
        "episode_count": len(episodes),
        "episodes": [{"id": episode["id"], "path": episode["path"], "fps": episode["fps"]} for episode in episodes],
    }

    _write_json(out / "delay_scan.json", delay_scan)
    _write_json(out / "offline_model_fit.json", offline)
    _write_json(out / "rlt_right_arm_hold_audit.json", {
        "summary": right_arm_hold_summary,
        "episodes": right_arm_hold_rows,
    })
    _write_json(out / "summary.json", summary)
    _write_csv(out / "rlt_right_arm_hold_audit.csv", [_flatten_hold_row(row) for row in right_arm_hold_rows])
    _write_csv(out / "per_joint_delay_summary.csv", [
        {
            "joint": name,
            "best_delay": payload["best_delay"],
            "rmse": payload["best"]["rmse"],
            "mae": payload["best"]["mae"],
            "max_abs": payload["best"]["max_abs"],
        }
        for name, payload in delay_scan["per_joint"].items()
    ])
    _write_markdown_reports(out, summary, delay_scan, offline, right_arm_hold_rows, right_arm_hold_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
