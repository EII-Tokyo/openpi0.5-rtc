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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase118_pipe_axis_hypothesis_probe_20260719"
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


def _pipe_base_and_deltas(config: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    measurement = config["pipe_placeholder"]["measurement"]
    table = config["table"]
    size = table["size"]
    translation = table["pose"]["translation"]
    table_edge = measurement["table_edge"]
    if table_edge not in {"w0", "w1"}:
        raise ValueError(f"unsupported table_edge {table_edge!r}")

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
    tilt_rad = math.radians(float(measurement["side_tilt_deg"]))
    horizontal = length * math.cos(tilt_rad)
    vertical = length * math.sin(tilt_rad)
    return base, {
        "x_negative_current": np.asarray([-horizontal, 0.0, vertical], dtype=np.float64),
        "x_positive_opposite": np.asarray([horizontal, 0.0, vertical], dtype=np.float64),
        "y_negative_toward_table": np.asarray([0.0, -horizontal, vertical], dtype=np.float64),
        "y_positive_outward": np.asarray([0.0, horizontal, vertical], dtype=np.float64),
    }


def _point_segment_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    denom = float(segment @ segment)
    if denom <= 0:
        raise ValueError("pipe axis segment has zero length")
    t = np.clip(((points - start) @ segment) / denom, 0.0, 1.0)
    closest = start + t[:, None] * segment
    return np.linalg.norm(points - closest, axis=1)


def _axis_unsigned_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)
    direct = math.degrees(math.acos(float(np.clip(a @ b, -1.0, 1.0))))
    flipped = math.degrees(math.acos(float(np.clip((-a) @ b, -1.0, 1.0))))
    return min(direct, flipped)


def _evaluate(positions: np.ndarray, axes: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    base, deltas = _pipe_base_and_deltas(config)
    final_axis = np.mean(axes[-10:], axis=0)
    final_axis = final_axis / max(float(np.linalg.norm(final_axis)), 1e-12)
    final_position = np.mean(positions[-10:], axis=0)
    hypotheses = []
    for name, delta in deltas.items():
        end = base + delta
        distances = _point_segment_distance(positions, base, end)
        unit = delta / max(float(np.linalg.norm(delta)), 1e-12)
        hypotheses.append(
            {
                "name": name,
                "pipe_start": base.tolist(),
                "pipe_end": end.tolist(),
                "min_mouth_to_axis_distance_m": float(np.min(distances)),
                "final_mouth_to_axis_distance_m": float(distances[-1]),
                "final_axis_unsigned_angle_deg": _axis_unsigned_angle_deg(final_axis, unit),
            }
        )
    return {
        "status": "PASS",
        "mouth_samples": int(len(positions)),
        "final_mouth_mean": final_position.tolist(),
        "final_mouth_axis_mean": final_axis.tolist(),
        "hypotheses": sorted(
            hypotheses,
            key=lambda row: (float(row["final_axis_unsigned_angle_deg"]), float(row["final_mouth_to_axis_distance_m"])),
        ),
    }


def _plot(positions: np.ndarray, summary: dict[str, Any], output_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.scatter(positions[:, 0], positions[:, 1], s=12, c=np.arange(len(positions)), cmap="viridis", label="bottle mouth path")
    colors = {
        "x_negative_current": "#d62728",
        "x_positive_opposite": "#2ca02c",
        "y_negative_toward_table": "#1f77b4",
        "y_positive_outward": "#9467bd",
    }
    for row in summary["hypotheses"]:
        start = np.asarray(row["pipe_start"], dtype=np.float64)
        end = np.asarray(row["pipe_end"], dtype=np.float64)
        ax.plot([start[0], end[0]], [start[1], end[1]], linewidth=2.5, color=colors[row["name"]], label=row["name"])
        ax.scatter([end[0]], [end[1]], color=colors[row["name"]], s=35)
    ax.set_title("Phase118 pipe axis hypotheses, top view")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe measured pipe direction hypotheses against Phase117 mouth trajectory.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_PHASE117_DIR / "gripper_passive_contact_timeseries.csv")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    positions, axes = _load_mouth_csv(args.csv)
    summary = _evaluate(positions, axes, yaml.safe_load(args.config.read_text()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pipe_axis_hypothesis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot(positions, summary, args.output_dir / "pipe_axis_hypotheses_top_view.png")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
