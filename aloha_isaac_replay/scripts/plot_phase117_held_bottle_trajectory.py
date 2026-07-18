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
DEFAULT_CONFIG = REPO_ROOT / "examples/aloha_isaac/config/workcell_user_measured.yaml"


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    if value in {None, ""}:
        raise ValueError(f"missing required CSV field {key!r}")
    return float(value)


def _load_mouth_rows(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{csv_path} has no rows")

    positions: list[list[float]] = []
    axes: list[list[float]] = []
    phases: list[str] = []
    for row in rows:
        exists = row.get("object_mouth_frame_exists")
        if exists not in {"True", "true", "1"}:
            continue
        positions.append([_float(row, "object_mouth_x"), _float(row, "object_mouth_y"), _float(row, "object_mouth_z")])
        axis = np.array(
            [_float(row, "object_mouth_axis_x"), _float(row, "object_mouth_axis_y"), _float(row, "object_mouth_axis_z")],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(axis))
        axes.append((axis / norm if norm > 0 else axis).tolist())
        phases.append(row.get("phase", "unknown"))

    if not positions:
        raise ValueError(f"{csv_path} has no object_mouth_* rows")
    return np.asarray(positions, dtype=np.float64), np.asarray(axes, dtype=np.float64), phases


def _resolve_pipe_axis(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    measurement = config["pipe_placeholder"]["measurement"]
    table_size = config["table"]["size"]
    table_translation = config["table"]["pose"]["translation"]
    table_edge = measurement["table_edge"]
    if table_edge not in {"w0", "w1"}:
        raise ValueError(f"unsupported table_edge {table_edge!r}")

    left_edge_x = float(table_translation[0]) - float(table_size[0]) / 2.0
    edge_y = float(table_translation[1]) + (float(table_size[1]) / 2.0 if table_edge == "w1" else -float(table_size[1]) / 2.0)
    a_point = np.array([left_edge_x + float(measurement["a_distance_from_left_edge_m"]), edge_y, 0.0], dtype=np.float64)
    outside_sign = 1.0 if table_edge == "w1" else -1.0
    start = np.array(
        [
            a_point[0],
            a_point[1] + outside_sign * float(measurement["base_offset_outside_table_m"]),
            float(measurement["mount_height_m"]),
        ],
        dtype=np.float64,
    )

    length = float(measurement["pipe_length_m"])
    tilt_rad = math.radians(float(measurement["side_tilt_deg"]))
    horizontal = length * math.cos(tilt_rad)
    vertical = length * math.sin(tilt_rad)
    plan_direction = measurement.get("plan_direction", "toward_table")
    if plan_direction == "parallel_to_table_edge_toward_left_arm":
        end = np.array([start[0] - horizontal, start[1], start[2] + vertical], dtype=np.float64)
    elif plan_direction == "toward_table":
        toward_table_sign = -1.0 if table_edge == "w1" else 1.0
        end = np.array([start[0], start[1] + toward_table_sign * horizontal, start[2] + vertical], dtype=np.float64)
    else:
        raise ValueError(f"unsupported plan_direction {plan_direction!r}")
    return start, end


def _point_segment_distances(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 0:
        raise ValueError("pipe axis start and end are identical")
    t = np.clip(((points - start) @ segment) / denom, 0.0, 1.0)
    closest = start + t[:, None] * segment
    return np.linalg.norm(points - closest, axis=1)


def _set_equal_3d(ax: Any, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 0.05)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius), center[2] + radius)


def _plot(csv_path: Path, config_path: Path, output_png: Path, output_json: Path) -> None:
    positions, axes, phases = _load_mouth_rows(csv_path)
    config = yaml.safe_load(config_path.read_text())
    pipe_start, pipe_end = _resolve_pipe_axis(config)

    all_points = np.vstack([positions, pipe_start[None, :], pipe_end[None, :]])
    distances_to_axis = _point_segment_distances(positions, pipe_start, pipe_end)
    distances_to_entry = np.linalg.norm(positions - pipe_end[None, :], axis=1)

    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    ax_top = fig.add_subplot(1, 3, 2)
    ax_side = fig.add_subplot(1, 3, 3)

    idx = np.arange(len(positions))
    colors = np.where(np.array(phases) == "settle", "#6f6f6f", "#1f77b4")

    ax3d.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=idx, cmap="viridis", s=12, label="bottle mouth")
    ax3d.plot([pipe_start[0], pipe_end[0]], [pipe_start[1], pipe_end[1]], [pipe_start[2], pipe_end[2]], color="red", linewidth=3, label="pipe axis")
    ax3d.scatter([pipe_end[0]], [pipe_end[1]], [pipe_end[2]], color="red", s=45, label="pipe entry")
    stride = max(1, len(positions) // 12)
    ax3d.quiver(
        positions[::stride, 0],
        positions[::stride, 1],
        positions[::stride, 2],
        axes[::stride, 0],
        axes[::stride, 1],
        axes[::stride, 2],
        length=0.025,
        color="#cc5500",
        normalize=True,
    )
    ax3d.set_title("3D mouth trajectory and direction")
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.legend(loc="upper left", fontsize=8)
    _set_equal_3d(ax3d, all_points)

    ax_top.scatter(positions[:, 0], positions[:, 1], c=colors, s=14)
    ax_top.plot([pipe_start[0], pipe_end[0]], [pipe_start[1], pipe_end[1]], color="red", linewidth=3)
    ax_top.scatter([pipe_end[0]], [pipe_end[1]], color="red", s=45)
    ax_top.set_title("Top view: x-y")
    ax_top.set_xlabel("x (m)")
    ax_top.set_ylabel("y (m)")
    ax_top.axis("equal")
    ax_top.grid(True, alpha=0.25)

    ax_side.scatter(positions[:, 0], positions[:, 2], c=colors, s=14)
    ax_side.plot([pipe_start[0], pipe_end[0]], [pipe_start[2], pipe_end[2]], color="red", linewidth=3)
    ax_side.scatter([pipe_end[0]], [pipe_end[2]], color="red", s=45)
    ax_side.set_title("Side view: x-z")
    ax_side.set_xlabel("x (m)")
    ax_side.set_ylabel("z (m)")
    ax_side.axis("equal")
    ax_side.grid(True, alpha=0.25)

    fig.suptitle("Phase117 diagnostic held-bottle replay: bottle mouth vs measured pipe axis")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)

    summary = {
        "status": "PASS",
        "csv_path": str(csv_path),
        "config_path": str(config_path),
        "output_png": str(output_png),
        "mouth_samples": int(len(positions)),
        "pipe_start": pipe_start.tolist(),
        "pipe_end_entry": pipe_end.tolist(),
        "mouth_start": positions[0].tolist(),
        "mouth_end": positions[-1].tolist(),
        "mouth_total_displacement_m": float(np.linalg.norm(positions[-1] - positions[0])),
        "min_mouth_distance_to_pipe_axis_m": float(np.min(distances_to_axis)),
        "final_mouth_distance_to_pipe_axis_m": float(distances_to_axis[-1]),
        "min_mouth_distance_to_pipe_entry_m": float(np.min(distances_to_entry)),
        "final_mouth_distance_to_pipe_entry_m": float(distances_to_entry[-1]),
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Phase117 diagnostic bottle-mouth trajectory against the measured pipe axis.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_PHASE117_DIR / "gripper_passive_contact_timeseries.csv")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_PHASE117_DIR / "held_bottle_mouth_trajectory.png")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_PHASE117_DIR / "held_bottle_mouth_trajectory_summary.json")
    args = parser.parse_args()
    _plot(args.csv, args.config, args.output_png, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
