from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aloha_isaac_replay.scripts.find_no_actor_right_arm_candidates import _arm_features
from aloha_isaac_replay.scripts.find_no_actor_right_arm_candidates import _quantiles
from aloha_isaac_replay.scripts.find_no_actor_right_arm_candidates import LEFT_ARM_JOINT_INDICES
from aloha_isaac_replay.scripts.find_no_actor_right_arm_candidates import LEFT_SHOULDER_INDEX
from aloha_isaac_replay.scripts.find_no_actor_right_arm_candidates import RIGHT_ARM_JOINT_INDICES
from aloha_isaac_replay.scripts.find_no_actor_right_arm_candidates import RIGHT_SHOULDER_INDEX


def _as_matrix(series: pd.Series) -> np.ndarray:
    return np.asarray(series.tolist(), dtype=np.float64)


def _dataset_name(path: Path) -> str:
    # .../<dataset>/data/chunk-000/file-000.parquet
    for parent in path.parents:
        if (parent / "meta" / "info.json").exists():
            return parent.name
    return path.parent.name


def _scan_file(path: Path) -> list[dict[str, Any]]:
    df = pd.read_parquet(path, columns=["observation.state", "action", "timestamp", "episode_index", "frame_index"])
    dataset = _dataset_name(path)
    rows: list[dict[str, Any]] = []
    for episode_index, group in df.groupby("episode_index", sort=True):
        group = group.sort_values("frame_index")
        qpos = _as_matrix(group["observation.state"])
        action = _as_matrix(group["action"])
        if qpos.ndim != 2 or action.ndim != 2 or qpos.shape[1] < 14 or action.shape[1] < 14:
            continue
        n = min(len(qpos), len(action))
        qpos = qpos[:n]
        action = action[:n]
        left = _arm_features(action, qpos, LEFT_ARM_JOINT_INDICES)
        right = _arm_features(action, qpos, RIGHT_ARM_JOINT_INDICES)
        rows.append(
            {
                "dataset": dataset,
                "episode_id": f"{dataset}/episode_{int(episode_index):06d}",
                "episode_index": int(episode_index),
                "path": str(path),
                "frames": int(n),
                "duration_s": float(group["timestamp"].iloc[-1] - group["timestamp"].iloc[0]) if n > 1 else 0.0,
                "source_type": "lerobot_human",
                "right_arm_action_std_mean": right["action_std_mean"],
                "right_arm_action_range_mean": right["action_range_mean"],
                "right_arm_action_velocity_rms": right["action_velocity_rms"],
                "right_arm_action_acceleration_rms": right["action_acceleration_rms"],
                "right_arm_qpos_range_mean": right["qpos_range_mean"],
                "right_arm_static_hold_ratio": right["static_hold_ratio"],
                "right_arm_saturation_ratio": right["saturation_ratio"],
                "right_shoulder_action_range": float(np.ptp(action[:, RIGHT_SHOULDER_INDEX])),
                "right_shoulder_qpos_range": float(np.ptp(qpos[:, RIGHT_SHOULDER_INDEX])),
                "left_arm_action_std_mean": left["action_std_mean"],
                "left_arm_action_range_mean": left["action_range_mean"],
                "left_arm_action_velocity_rms": left["action_velocity_rms"],
                "left_arm_action_acceleration_rms": left["action_acceleration_rms"],
                "left_arm_qpos_range_mean": left["qpos_range_mean"],
                "left_arm_static_hold_ratio": left["static_hold_ratio"],
                "left_arm_saturation_ratio": left["saturation_ratio"],
                "left_shoulder_action_range": float(np.ptp(action[:, LEFT_SHOULDER_INDEX])),
                "action_qpos_mean_abs_delta": float(np.mean(np.abs(action[:, :14] - qpos[:, :14]))),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, payload: dict[str, Any], right_candidates: list[dict[str, Any]]) -> None:
    lines = [
        "# LeRobot Human Controller-ID Candidate Scan",
        "",
        f"- Datasets scanned: `{payload['dataset_count']}`",
        f"- Episodes scanned: `{payload['episode_count']}`",
        f"- Right-arm candidates: `{payload['right_arm_candidate_count']}`",
        f"- Left-arm candidates: `{payload['left_arm_candidate_count']}`",
        f"- Bimanual candidates: `{payload['bimanual_candidate_count']}`",
        "",
        "These are raw LeRobot human-control episodes with `observation.state`, `action`, `timestamp`, and `episode_index`. They are suitable for controller-ID qualification after confirming the same 14D ALOHA convention; they are different from converted RLT Q replay shards.",
        "",
        "## Right-Arm Candidates",
        "",
        "| rank | episode | frames | right arm range mean | right shoulder range | action-qpos mean abs delta |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(right_candidates[:40], start=1):
        lines.append(
            f"| {rank} | {row['episode_id']} | {row['frames']} | "
            f"{row['right_arm_action_range_mean']:.6f} | {row['right_shoulder_action_range']:.6f} | "
            f"{row['action_qpos_mean_abs_delta']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan raw LeRobot human episodes for controller-ID excitation.")
    parser.add_argument("--root", default="/home/eii/.cache/huggingface/lerobot/lyl472324464")
    parser.add_argument("--output-dir", default="reports/aloha_isaac_replay/controller_system_id/lerobot_human_scan")
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(root.glob("*/data/chunk-*/file-*.parquet"))
    rows: list[dict[str, Any]] = []
    for path in files:
        rows.extend(_scan_file(path))
    distribution = _quantiles(
        rows,
        [
            "left_arm_action_range_mean",
            "left_shoulder_action_range",
            "right_arm_action_range_mean",
            "right_shoulder_action_range",
            "right_arm_action_velocity_rms",
            "right_arm_qpos_range_mean",
            "action_qpos_mean_abs_delta",
        ],
    )
    thresholds = {
        "min_frames": 100,
        "right_arm_action_range_mean": distribution["right_arm_action_range_mean"]["p75"],
        "right_shoulder_action_range": distribution["right_shoulder_action_range"]["p50"],
        "left_arm_action_range_mean": distribution["left_arm_action_range_mean"]["p75"],
        "left_shoulder_action_range": distribution["left_shoulder_action_range"]["p50"],
        "max_static_hold_ratio": 0.98,
        "max_saturation_ratio": 0.05,
    }
    for row in rows:
        row["usable_right_arm_id"] = bool(
            row["frames"] >= thresholds["min_frames"]
            and row["right_arm_action_range_mean"] >= thresholds["right_arm_action_range_mean"]
            and row["right_shoulder_action_range"] >= thresholds["right_shoulder_action_range"]
            and row["right_arm_static_hold_ratio"] <= thresholds["max_static_hold_ratio"]
            and row["right_arm_saturation_ratio"] <= thresholds["max_saturation_ratio"]
        )
        row["usable_left_arm_id"] = bool(
            row["frames"] >= thresholds["min_frames"]
            and row["left_arm_action_range_mean"] >= thresholds["left_arm_action_range_mean"]
            and row["left_shoulder_action_range"] >= thresholds["left_shoulder_action_range"]
            and row["left_arm_static_hold_ratio"] <= thresholds["max_static_hold_ratio"]
            and row["left_arm_saturation_ratio"] <= thresholds["max_saturation_ratio"]
        )
        row["usable_bimanual_id"] = bool(row["usable_left_arm_id"] and row["usable_right_arm_id"])
    rows.sort(key=lambda row: (-row["right_arm_action_range_mean"], row["episode_id"]))
    right_candidates = [row for row in rows if row["usable_right_arm_id"]]
    left_candidates = [row for row in rows if row["usable_left_arm_id"]]
    bimanual_candidates = [row for row in rows if row["usable_bimanual_id"]]
    payload = {
        "root": str(root),
        "dataset_count": len({row["dataset"] for row in rows}),
        "episode_count": len(rows),
        "right_arm_candidate_count": len(right_candidates),
        "left_arm_candidate_count": len(left_candidates),
        "bimanual_candidate_count": len(bimanual_candidates),
        "thresholds": thresholds,
        "distribution": distribution,
    }
    (out / "lerobot_human_controller_id_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    _write_csv(out / "lerobot_human_all_episodes.csv", rows)
    _write_csv(out / "lerobot_human_right_arm_candidates.csv", right_candidates)
    _write_csv(out / "lerobot_human_left_arm_candidates.csv", left_candidates)
    _write_csv(out / "lerobot_human_bimanual_candidates.csv", bimanual_candidates)
    _write_markdown(out / "lerobot_human_controller_id_summary.md", payload, right_candidates)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
