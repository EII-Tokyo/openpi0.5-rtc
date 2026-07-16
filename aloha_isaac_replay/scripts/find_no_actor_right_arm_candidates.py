from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


RIGHT_ARM_JOINT_INDICES = tuple(range(7, 13))
RIGHT_SHOULDER_INDEX = 8
LEFT_ARM_JOINT_INDICES = tuple(range(0, 6))
LEFT_SHOULDER_INDEX = 1


def _episode_id(path: Path) -> str:
    if path.parent.name.startswith("key_region_"):
        return path.parent.name
    return path.stem


def _safe_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _safe_attr(value.item())
        return [_safe_attr(item) for item in value.tolist()]
    return value


def _source_bucket(path: Path) -> str:
    parts = path.parts
    for marker in ("warmup", "rl", "no_actor", "human", "expert"):
        if marker in parts:
            return marker
    return "unknown"


def _rms(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values))))


def _coverage(values: np.ndarray, *, eps: float = 1e-5) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    return float(np.mean(values > eps)), float(np.mean(values < -eps))


def _arm_features(action: np.ndarray, qpos: np.ndarray, indices: tuple[int, ...]) -> dict[str, Any]:
    arm_action = action[:, indices]
    arm_qpos = qpos[:, indices]
    action_delta = np.diff(arm_action, axis=0)
    action_accel = np.diff(action_delta, axis=0)
    qpos_delta = np.diff(arm_qpos, axis=0)
    positive, negative = _coverage(action_delta)
    action_std = np.std(arm_action, axis=0)
    action_range = np.ptp(arm_action, axis=0)
    qpos_range = np.ptp(arm_qpos, axis=0)
    meaningful_changes = np.sum(np.abs(action_delta) > 1e-4, axis=0) if action_delta.size else np.zeros(len(indices))
    saturation = np.mean(np.isclose(np.abs(arm_action), np.pi, atol=1e-3))
    static_steps = np.mean(np.max(np.abs(action_delta), axis=1) <= 1e-5) if action_delta.size else 1.0
    return {
        "action_std_mean": float(np.mean(action_std)),
        "action_std_max": float(np.max(action_std)),
        "action_range_mean": float(np.mean(action_range)),
        "action_range_max": float(np.max(action_range)),
        "action_velocity_rms": _rms(action_delta),
        "action_acceleration_rms": _rms(action_accel),
        "meaningful_command_changes_mean": float(np.mean(meaningful_changes)),
        "meaningful_command_changes_min": float(np.min(meaningful_changes)),
        "qpos_range_mean": float(np.mean(qpos_range)),
        "qpos_range_max": float(np.max(qpos_range)),
        "qpos_velocity_rms": _rms(qpos_delta),
        "positive_direction_coverage": positive,
        "negative_direction_coverage": negative,
        "saturation_ratio": float(saturation),
        "static_hold_ratio": float(static_steps),
    }


def analyze_episode(path: Path, *, static_tolerance: float = 1e-3) -> dict[str, Any] | None:
    try:
        with h5py.File(path, "r") as h5:
            if "action" not in h5 or "observations/qpos" not in h5:
                return None
            action = np.asarray(h5["action"][:], dtype=np.float64)
            qpos = np.asarray(h5["observations/qpos"][:], dtype=np.float64)
            if action.ndim != 2 or qpos.ndim != 2 or action.shape[1] < 14 or qpos.shape[1] < 14:
                return None
            reference = np.asarray(h5["reference_action"][:], dtype=np.float64) if "reference_action" in h5 else None
            attrs = {key: _safe_attr(value) for key, value in h5.attrs.items()}
            rlt_attrs = {key: _safe_attr(value) for key, value in h5["rlt"].attrs.items()} if "rlt" in h5 else {}
    except OSError:
        return None

    n = min(action.shape[0], qpos.shape[0])
    action = action[:n]
    qpos = qpos[:n]
    if reference is not None:
        reference = reference[: min(n, reference.shape[0])]

    left_features = _arm_features(action, qpos, LEFT_ARM_JOINT_INDICES)
    right_features = _arm_features(action, qpos, RIGHT_ARM_JOINT_INDICES)
    right_shoulder_action = action[:, RIGHT_SHOULDER_INDEX]
    right_shoulder_qpos = qpos[:, RIGHT_SHOULDER_INDEX]
    left_shoulder_action = action[:, LEFT_SHOULDER_INDEX]

    reference_max_abs_diff = None
    action_equals_reference = None
    if reference is not None and reference.shape == action.shape:
        reference_max_abs_diff = float(np.max(np.abs(action - reference)))
        action_equals_reference = bool(reference_max_abs_diff <= 1e-5)

    phase = attrs.get("phase")
    action_source = attrs.get("action_source") or rlt_attrs.get("action_source")
    behavior_policy = attrs.get("behavior_policy") or rlt_attrs.get("behavior_policy")
    source_bucket = _source_bucket(path)
    has_rlt_actor_adjustment = bool(
        (action_source == "rlt_actor_adjusted_action")
        or (behavior_policy == "rlt_actor")
        or (reference_max_abs_diff is not None and reference_max_abs_diff > 1e-5 and source_bucket == "rl")
    )
    no_actor_likely = bool(
        source_bucket == "warmup"
        or action_source == "vla_reference_action"
        or behavior_policy == "vla_reference"
        or action_equals_reference is True
    ) and not has_rlt_actor_adjustment
    static_right_arm = bool(right_features["action_std_max"] <= float(static_tolerance))
    static_left_arm = bool(left_features["action_std_max"] <= float(static_tolerance))
    right_arm_excitation_score = float(right_features["action_std_mean"])
    right_arm_range_score = float(right_features["action_range_mean"])
    left_arm_excitation_score = float(left_features["action_std_mean"])
    left_arm_range_score = float(left_features["action_range_mean"])
    right_shoulder_range = float(np.ptp(right_shoulder_action))
    left_shoulder_range = float(np.ptp(left_shoulder_action))

    return {
        "episode_id": _episode_id(path),
        "path": str(path),
        "frames": int(n),
        "source_bucket": source_bucket,
        "phase": phase,
        "reward": attrs.get("reward"),
        "is_key_region": attrs.get("is_key_region"),
        "has_reference_action": reference is not None,
        "action_source": action_source,
        "behavior_policy": behavior_policy,
        "action_equals_reference": action_equals_reference,
        "action_reference_max_abs_diff": reference_max_abs_diff,
        "has_rlt_actor_adjustment": has_rlt_actor_adjustment,
        "no_actor_likely": no_actor_likely,
        "static_left_arm": static_left_arm,
        "static_right_arm": static_right_arm,
        "right_arm_action_std_mean": right_arm_excitation_score,
        "right_arm_action_std_max": right_features["action_std_max"],
        "right_arm_action_range_mean": right_arm_range_score,
        "right_arm_action_range_max": right_features["action_range_max"],
        "right_arm_action_velocity_rms": right_features["action_velocity_rms"],
        "right_arm_action_acceleration_rms": right_features["action_acceleration_rms"],
        "right_arm_meaningful_command_changes_mean": right_features["meaningful_command_changes_mean"],
        "right_arm_meaningful_command_changes_min": right_features["meaningful_command_changes_min"],
        "right_arm_qpos_range_mean": right_features["qpos_range_mean"],
        "right_arm_qpos_velocity_rms": right_features["qpos_velocity_rms"],
        "right_arm_positive_direction_coverage": right_features["positive_direction_coverage"],
        "right_arm_negative_direction_coverage": right_features["negative_direction_coverage"],
        "right_arm_saturation_ratio": right_features["saturation_ratio"],
        "right_arm_static_hold_ratio": right_features["static_hold_ratio"],
        "left_arm_action_std_mean": left_arm_excitation_score,
        "left_arm_action_range_mean": left_arm_range_score,
        "left_arm_action_velocity_rms": left_features["action_velocity_rms"],
        "left_arm_action_acceleration_rms": left_features["action_acceleration_rms"],
        "left_arm_meaningful_command_changes_mean": left_features["meaningful_command_changes_mean"],
        "left_arm_qpos_range_mean": left_features["qpos_range_mean"],
        "left_arm_qpos_velocity_rms": left_features["qpos_velocity_rms"],
        "left_arm_positive_direction_coverage": left_features["positive_direction_coverage"],
        "left_arm_negative_direction_coverage": left_features["negative_direction_coverage"],
        "left_arm_saturation_ratio": left_features["saturation_ratio"],
        "left_arm_static_hold_ratio": left_features["static_hold_ratio"],
        "left_shoulder_action_range": left_shoulder_range,
        "right_shoulder_action_min": float(np.min(right_shoulder_action)),
        "right_shoulder_action_max": float(np.max(right_shoulder_action)),
        "right_shoulder_action_range": right_shoulder_range,
        "right_shoulder_action_std": float(np.std(right_shoulder_action)),
        "right_shoulder_qpos_min": float(np.min(right_shoulder_qpos)),
        "right_shoulder_qpos_max": float(np.max(right_shoulder_qpos)),
        "right_shoulder_qpos_range": float(np.ptp(right_shoulder_qpos)),
    }


def _quantiles(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows if row.get(key) is not None], dtype=np.float64)
        if values.size == 0:
            continue
        out[key] = {
            "min": float(np.min(values)),
            "p10": float(np.quantile(values, 0.10)),
            "p25": float(np.quantile(values, 0.25)),
            "p50": float(np.quantile(values, 0.50)),
            "p75": float(np.quantile(values, 0.75)),
            "p90": float(np.quantile(values, 0.90)),
            "max": float(np.max(values)),
        }
    return out


def _classify_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    no_actor = [row for row in rows if row["no_actor_likely"]]
    base = no_actor if no_actor else rows
    distribution = _quantiles(
        rows,
        [
            "left_arm_action_std_mean",
            "left_arm_action_range_mean",
            "left_shoulder_action_range",
            "right_arm_action_std_mean",
            "right_arm_action_range_mean",
            "right_shoulder_action_range",
            "right_arm_action_velocity_rms",
            "right_arm_action_acceleration_rms",
            "right_arm_qpos_range_mean",
        ],
    )
    no_actor_distribution = _quantiles(
        base,
        [
            "left_arm_action_std_mean",
            "left_arm_action_range_mean",
            "left_shoulder_action_range",
            "right_arm_action_std_mean",
            "right_arm_action_range_mean",
            "right_shoulder_action_range",
            "right_arm_action_velocity_rms",
            "right_arm_action_acceleration_rms",
            "right_arm_qpos_range_mean",
        ],
    )
    thresholds = {
        "left_arm_action_range_mean": no_actor_distribution["left_arm_action_range_mean"]["p75"],
        "left_shoulder_action_range": no_actor_distribution["left_shoulder_action_range"]["p50"],
        "right_arm_action_range_mean": no_actor_distribution["right_arm_action_range_mean"]["p75"],
        "right_shoulder_action_range": no_actor_distribution["right_shoulder_action_range"]["p50"],
        "min_frames": 100,
        "max_static_hold_ratio": 0.98,
        "max_saturation_ratio": 0.05,
    }
    for row in rows:
        enough_length = int(row["frames"]) >= thresholds["min_frames"]
        no_actor_likely = bool(row["no_actor_likely"])
        left_excited = bool(
            row["left_arm_action_range_mean"] >= thresholds["left_arm_action_range_mean"]
            and row["left_shoulder_action_range"] >= thresholds["left_shoulder_action_range"]
            and row["left_arm_static_hold_ratio"] <= thresholds["max_static_hold_ratio"]
        )
        right_excited = bool(
            row["right_arm_action_range_mean"] >= thresholds["right_arm_action_range_mean"]
            and row["right_shoulder_action_range"] >= thresholds["right_shoulder_action_range"]
            and row["right_arm_static_hold_ratio"] <= thresholds["max_static_hold_ratio"]
        )
        safe_quality = bool(
            row["right_arm_saturation_ratio"] <= thresholds["max_saturation_ratio"]
            and row["left_arm_saturation_ratio"] <= thresholds["max_saturation_ratio"]
        )
        row["usable_left_arm_id"] = bool(no_actor_likely and enough_length and left_excited and safe_quality)
        row["usable_right_arm_id"] = bool(no_actor_likely and enough_length and right_excited and safe_quality)
        row["usable_bimanual_id"] = bool(row["usable_left_arm_id"] and row["usable_right_arm_id"])
        row["hold_stability_only"] = bool(row["static_left_arm"] or row["static_right_arm"])
        row["insufficient_excitation"] = bool(no_actor_likely and not (row["usable_left_arm_id"] or row["usable_right_arm_id"]))
        row["contact_dominated"] = bool(row.get("phase") == "rl" and row.get("is_key_region") is True)
        row["jitter_candidate"] = bool(
            row["right_arm_action_acceleration_rms"] >= no_actor_distribution["right_arm_action_acceleration_rms"]["p90"]
        )
        row["usable_for_right_arm_controller_id"] = row["usable_right_arm_id"]
    return {
        "distribution": distribution,
        "no_actor_distribution": no_actor_distribution,
        "thresholds": thresholds,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_distribution_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Dataset Excitation Distribution",
        "",
        f"- Episodes scanned: `{payload['episode_count']}`",
        f"- No-actor likely: `{payload['no_actor_likely_count']}`",
        f"- Usable left-arm ID: `{payload['usable_left_arm_id_count']}`",
        f"- Usable right-arm ID: `{payload['usable_right_arm_id_count']}`",
        f"- Usable bimanual ID: `{payload['usable_bimanual_id_count']}`",
        f"- Hold stability only: `{payload['hold_stability_only_count']}`",
        "",
        "Thresholds are derived from the no-actor subset distribution, not picked before scanning.",
        "",
        "## Data-Derived Thresholds",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key, value in payload["thresholds"].items():
        lines.append(f"| {key} | {value} |")
    lines += ["", "## No-Actor Distribution Quantiles", ""]
    for metric, stats in payload["no_actor_distribution"].items():
        lines += [
            f"### `{metric}`",
            "",
            "| min | p10 | p25 | p50 | p75 | p90 | max |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| {stats['min']:.6f} | {stats['p10']:.6f} | {stats['p25']:.6f} | "
                f"{stats['p50']:.6f} | {stats['p75']:.6f} | {stats['p90']:.6f} | {stats['max']:.6f} |"
            ),
            "",
        ]
    path.write_text("\n".join(lines) + "\n")


def _write_markdown(path: Path, rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> None:
    lines = [
        "# No-Actor Right-Arm Candidate Scan",
        "",
        f"- Episodes scanned: `{len(rows)}`",
        f"- No-actor likely: `{sum(1 for row in rows if row['no_actor_likely'])}`",
        f"- Usable for right-arm controller ID: `{sum(1 for row in rows if row['usable_right_arm_id'])}`",
        "",
        "Selection rule: no actor likely, enough length, data-derived right-arm range threshold, visible right_shoulder excitation, not static, and low saturation.",
        "",
        "## Selected Candidates",
        "",
        "| rank | episode | bucket | frames | reward | right arm std mean | right arm range mean | right shoulder range | path |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(selected, start=1):
        lines.append(
            f"| {rank} | {row['episode_id']} | {row['source_bucket']} | {row['frames']} | "
            f"{row['reward']} | {row['right_arm_action_std_mean']:.6f} | "
            f"{row['right_arm_action_range_mean']:.6f} | {row['right_shoulder_action_range']:.6f} | "
            f"`{row['path']}` |"
        )
    lines += [
        "",
        "## Top No-Actor Rows By Right-Arm Variation",
        "",
        "| rank | episode | usable | bucket | action==reference | right arm std mean | right arm range mean | right shoulder range |",
        "|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    top_no_actor = [row for row in rows if row["no_actor_likely"]][:30]
    for rank, row in enumerate(top_no_actor, start=1):
        lines.append(
            f"| {rank} | {row['episode_id']} | {row['usable_for_right_arm_controller_id']} | "
            f"{row['source_bucket']} | {row['action_equals_reference']} | "
            f"{row['right_arm_action_std_mean']:.6f} | {row['right_arm_action_range_mean']:.6f} | "
            f"{row['right_shoulder_action_range']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Find no-actor episodes with right-arm excitation.")
    parser.add_argument("--root", default="/home/eii/data/openpi0.5-rtc-reward-learning/from_103")
    parser.add_argument("--output-dir", default="reports/aloha_isaac_replay/controller_system_id")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(root.rglob("episode.hdf5"))
    rows = [row for path in paths if (row := analyze_episode(path)) is not None]
    distribution_payload = _classify_rows(rows)
    rows.sort(
        key=lambda row: (
            not row["no_actor_likely"],
            not row["usable_right_arm_id"],
            -row["right_arm_action_std_mean"],
            row["path"],
        )
    )
    selected = [row for row in rows if row["usable_right_arm_id"]][: args.limit]
    left_candidates = [row for row in rows if row["usable_left_arm_id"]]
    right_candidates = [row for row in rows if row["usable_right_arm_id"]]
    bimanual_candidates = [row for row in rows if row["usable_bimanual_id"]]
    hold_candidates = [row for row in rows if row["hold_stability_only"]]

    payload = {
        "root": str(root),
        "episode_count": len(rows),
        "no_actor_likely_count": sum(1 for row in rows if row["no_actor_likely"]),
        "usable_for_right_arm_controller_id_count": len(right_candidates),
        "usable_left_arm_id_count": len(left_candidates),
        "usable_right_arm_id_count": len(right_candidates),
        "usable_bimanual_id_count": len(bimanual_candidates),
        "hold_stability_only_count": len(hold_candidates),
        "insufficient_excitation_count": sum(1 for row in rows if row["insufficient_excitation"]),
        "contact_dominated_count": sum(1 for row in rows if row["contact_dominated"]),
        "jitter_candidate_count": sum(1 for row in rows if row["jitter_candidate"]),
        "distribution": distribution_payload["distribution"],
        "no_actor_distribution": distribution_payload["no_actor_distribution"],
        "thresholds": distribution_payload["thresholds"],
        "selected": selected,
    }
    (out / "no_actor_right_arm_candidates.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    (out / "dataset_excitation_distribution.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    _write_csv(out / "no_actor_right_arm_candidates.csv", rows)
    _write_csv(out / "right_arm_id_candidates.csv", right_candidates)
    _write_csv(out / "left_arm_id_candidates.csv", left_candidates)
    _write_csv(out / "bimanual_id_candidates.csv", bimanual_candidates)
    _write_csv(out / "hold_stability_candidates.csv", hold_candidates)
    _write_markdown(out / "no_actor_right_arm_candidates.md", rows, selected)
    _write_distribution_markdown(out / "dataset_excitation_distribution.md", payload)
    selected_payload = {
        "selected": [{"path": row["path"], "episode_id": row["episode_id"], "fps": 50.0} for row in selected]
    }
    (out / "selected_no_actor_right_arm_hdf5.json").write_text(
        json.dumps(selected_payload, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "episode_count",
                    "no_actor_likely_count",
                    "usable_left_arm_id_count",
                    "usable_right_arm_id_count",
                    "usable_bimanual_id_count",
                    "hold_stability_only_count",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
