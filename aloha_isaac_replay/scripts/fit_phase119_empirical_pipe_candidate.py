from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE117_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase117_diagnostic_held_bottle_replay_20260719"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase119_empirical_pipe_candidate_20260719"
DEFAULT_CONFIG = REPO_ROOT / "examples/aloha_isaac/config/workcell_user_measured.yaml"


def _load_mouth_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    positions: list[list[float]] = []
    axes: list[list[float]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("object_mouth_frame_exists") not in {"True", "true", "1"}:
                continue
            positions.append([float(row["object_mouth_x"]), float(row["object_mouth_y"]), float(row["object_mouth_z"])])
            axis = np.asarray(
                [float(row["object_mouth_axis_x"]), float(row["object_mouth_axis_y"]), float(row["object_mouth_axis_z"])],
                dtype=np.float64,
            )
            norm = float(np.linalg.norm(axis))
            axes.append((axis / norm if norm > 0 else axis).tolist())
    if not positions:
        raise ValueError(f"{path} has no valid object_mouth rows")
    return np.asarray(positions, dtype=np.float64), np.asarray(axes, dtype=np.float64)


def _measured_pipe(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    measurement = config["pipe_placeholder"]["measurement"]
    table = config["table"]
    size = table["size"]
    translation = table["pose"]["translation"]
    table_edge = measurement["table_edge"]
    left_edge_x = float(translation[0]) - float(size[0]) / 2.0
    edge_y = float(translation[1]) + (float(size[1]) / 2.0 if table_edge == "w1" else -float(size[1]) / 2.0)
    base = np.asarray(
        [
            left_edge_x + float(measurement["a_distance_from_left_edge_m"]),
            edge_y + (1.0 if table_edge == "w1" else -1.0) * float(measurement["base_offset_outside_table_m"]),
            float(measurement["mount_height_m"]),
        ],
        dtype=np.float64,
    )
    length = float(measurement["pipe_length_m"])
    tilt = math.radians(float(measurement["side_tilt_deg"]))
    horizontal = length * math.cos(tilt)
    vertical = length * math.sin(tilt)
    plan_direction = measurement.get("plan_direction", "toward_table")
    if plan_direction == "parallel_to_table_edge_toward_left_arm":
        entry = base + np.asarray([-horizontal, 0.0, vertical], dtype=np.float64)
    elif plan_direction == "toward_table":
        toward_table_sign = -1.0 if table_edge == "w1" else 1.0
        entry = base + np.asarray([0.0, toward_table_sign * horizontal, vertical], dtype=np.float64)
    else:
        raise ValueError(f"unsupported plan_direction {plan_direction!r}")
    return base, entry, length


def _fit_candidate(positions: np.ndarray, axes: np.ndarray, tail_fraction: float, pipe_length: float) -> dict[str, Any]:
    tail_count = max(3, int(round(len(positions) * tail_fraction)))
    tail_positions = positions[-tail_count:]
    tail_axes = axes[-tail_count:]
    entry = np.mean(tail_positions, axis=0)
    axis = np.mean(tail_axes, axis=0)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    base = entry - axis * pipe_length
    spread = np.linalg.norm(tail_positions - entry[None, :], axis=1)
    return {
        "tail_count": int(tail_count),
        "tail_fraction": float(tail_fraction),
        "entry": entry.tolist(),
        "axis_unit_base_to_entry": axis.tolist(),
        "base": base.tolist(),
        "tail_position_rms_spread_m": float(np.sqrt(np.mean(spread * spread))),
        "tail_position_max_spread_m": float(np.max(spread)),
    }


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)
    return math.degrees(math.acos(float(np.clip(a @ b, -1.0, 1.0))))


def _plot(positions: np.ndarray, measured_base: np.ndarray, measured_entry: np.ndarray, candidate: dict[str, Any], output_png: Path) -> None:
    candidate_base = np.asarray(candidate["base"], dtype=np.float64)
    candidate_entry = np.asarray(candidate["entry"], dtype=np.float64)
    fig = plt.figure(figsize=(14, 5), constrained_layout=True)
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    ax_top = fig.add_subplot(1, 3, 2)
    ax_side = fig.add_subplot(1, 3, 3)

    ax3d.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=np.arange(len(positions)), cmap="viridis", s=10)
    ax3d.plot(
        [measured_base[0], measured_entry[0]],
        [measured_base[1], measured_entry[1]],
        [measured_base[2], measured_entry[2]],
        color="red",
        linewidth=3,
        label="measured pipe",
    )
    ax3d.plot(
        [candidate_base[0], candidate_entry[0]],
        [candidate_base[1], candidate_entry[1]],
        [candidate_base[2], candidate_entry[2]],
        color="green",
        linewidth=3,
        label="empirical candidate",
    )
    ax3d.set_title("3D")
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.legend(fontsize=8)

    ax_top.scatter(positions[:, 0], positions[:, 1], c=np.arange(len(positions)), cmap="viridis", s=10)
    ax_top.plot([measured_base[0], measured_entry[0]], [measured_base[1], measured_entry[1]], color="red", linewidth=3, label="measured")
    ax_top.plot([candidate_base[0], candidate_entry[0]], [candidate_base[1], candidate_entry[1]], color="green", linewidth=3, label="empirical")
    ax_top.set_title("Top view: x-y")
    ax_top.set_xlabel("x (m)")
    ax_top.set_ylabel("y (m)")
    ax_top.axis("equal")
    ax_top.grid(True, alpha=0.25)
    ax_top.legend(fontsize=8)

    ax_side.scatter(positions[:, 0], positions[:, 2], c=np.arange(len(positions)), cmap="viridis", s=10)
    ax_side.plot([measured_base[0], measured_entry[0]], [measured_base[2], measured_entry[2]], color="red", linewidth=3, label="measured")
    ax_side.plot([candidate_base[0], candidate_entry[0]], [candidate_base[2], candidate_entry[2]], color="green", linewidth=3, label="empirical")
    ax_side.set_title("Side view: x-z")
    ax_side.set_xlabel("x (m)")
    ax_side.set_ylabel("z (m)")
    ax_side.axis("equal")
    ax_side.grid(True, alpha=0.25)
    ax_side.legend(fontsize=8)

    fig.suptitle("Phase119 empirical pipe candidate from final bottle-mouth cluster")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit a diagnostic empirical pipe candidate from the final Phase117 bottle-mouth cluster.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_PHASE117_DIR / "gripper_passive_contact_timeseries.csv")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tail-fraction", type=float, default=0.2)
    args = parser.parse_args()

    positions, axes = _load_mouth_csv(args.csv)
    config = yaml.safe_load(args.config.read_text())
    measured_base, measured_entry, pipe_length = _measured_pipe(config)
    candidate = _fit_candidate(positions, axes, args.tail_fraction, pipe_length)
    candidate_base = np.asarray(candidate["base"], dtype=np.float64)
    candidate_entry = np.asarray(candidate["entry"], dtype=np.float64)
    measured_axis = measured_entry - measured_base
    candidate_axis = np.asarray(candidate["axis_unit_base_to_entry"], dtype=np.float64)
    summary = {
        "status": "PASS",
        "csv": str(args.csv),
        "config": str(args.config),
        "measured_pipe": {
            "base": measured_base.tolist(),
            "entry": measured_entry.tolist(),
            "axis_unit_base_to_entry": (measured_axis / max(float(np.linalg.norm(measured_axis)), 1e-12)).tolist(),
        },
        "empirical_candidate": candidate,
        "candidate_minus_measured": {
            "base_delta_m": (candidate_base - measured_base).tolist(),
            "entry_delta_m": (candidate_entry - measured_entry).tolist(),
            "axis_angle_deg": _angle_deg(candidate_axis, measured_axis),
        },
        "warning": (
            "This is a diagnostic candidate from a held-object replay, not a calibrated physical pipe transform. "
            "Use it to decide what to measure next, not as an automatic config replacement."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "empirical_pipe_candidate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot(positions, measured_base, measured_entry, candidate, args.output_dir / "empirical_pipe_candidate.png")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
