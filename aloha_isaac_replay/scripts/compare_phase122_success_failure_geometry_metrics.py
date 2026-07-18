from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUCCESS_ROOT = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase120_success_hdf5_empirical_pipe_cluster_20260719"
DEFAULT_FAILURE_ROOT = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase121_failure_hdf5_empirical_pipe_cluster_20260719"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase122_success_failure_geometry_metrics_20260719"


def _load_positions(csv_path: Path) -> np.ndarray:
    positions: list[list[float]] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("object_mouth_frame_exists") in {"True", "true", "1"}:
                positions.append([float(row["object_mouth_x"]), float(row["object_mouth_y"]), float(row["object_mouth_z"])])
    if not positions:
        raise ValueError(f"{csv_path} has no valid object_mouth rows")
    return np.asarray(positions, dtype=np.float64)


def _metrics(positions: np.ndarray, reference_entry: np.ndarray, reference_axis: np.ndarray) -> dict[str, float | int]:
    projection = (positions - reference_entry) @ reference_axis
    lateral = np.linalg.norm((positions - reference_entry) - projection[:, None] * reference_axis, axis=1)
    tail_count = max(3, int(round(len(positions) * 0.2)))
    diffs = np.diff(positions, axis=0)
    return {
        "sample_count": int(len(positions)),
        "start_projection_m": float(projection[0]),
        "tail_projection_mean_m": float(np.mean(projection[-tail_count:])),
        "tail_progress_m": float(np.mean(projection[-tail_count:]) - projection[0]),
        "tail_lateral_mean_m": float(np.mean(lateral[-tail_count:])),
        "tail_lateral_std_m": float(np.std(lateral[-tail_count:])),
        "tail_lateral_max_m": float(np.max(lateral[-tail_count:])),
        "path_length_m": float(np.sum(np.linalg.norm(diffs, axis=1))),
        "net_displacement_m": float(np.linalg.norm(positions[-1] - positions[0])),
    }


def _load_cluster(root: Path, label: str, reference_entry: np.ndarray, reference_axis: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for item_dir in sorted(root.glob("2026-07-08_*")):
        csv_path = item_dir / "replay/gripper_passive_contact_timeseries.csv"
        if not csv_path.exists():
            continue
        metrics = _metrics(_load_positions(csv_path), reference_entry, reference_axis)
        rows.append({"label": label, "item": item_dir.name, "csv": str(csv_path), **metrics})
    return rows


def _group_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "tail_projection_mean_m",
        "tail_progress_m",
        "tail_lateral_mean_m",
        "tail_lateral_max_m",
        "path_length_m",
        "net_displacement_m",
    ]
    grouped: dict[str, Any] = {}
    for label in sorted({row["label"] for row in rows}):
        subset = [row for row in rows if row["label"] == label]
        grouped[label] = {"count": len(subset)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in subset], dtype=np.float64)
            grouped[label][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values.tolist(),
            }
    return grouped


def _auc_from_scores(success_scores: np.ndarray, failure_scores: np.ndarray) -> float:
    wins = 0.0
    total = 0
    for success_score in success_scores:
        wins += float(np.sum(success_score > failure_scores))
        wins += 0.5 * float(np.sum(success_score == failure_scores))
        total += int(failure_scores.size)
    if total == 0:
        return float("nan")
    return wins / float(total)


def _separation_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_directions = {
        "path_length_m": "higher",
        "net_displacement_m": "higher",
        "tail_progress_m": "higher",
        "tail_lateral_mean_m": "lower",
        "tail_lateral_max_m": "lower",
    }
    success_rows = [row for row in rows if row["label"] == "success"]
    failure_rows = [row for row in rows if row["label"] == "failure"]
    stats: dict[str, Any] = {}
    for metric, direction in metric_directions.items():
        success_values = np.asarray([float(row[metric]) for row in success_rows], dtype=np.float64)
        failure_values = np.asarray([float(row[metric]) for row in failure_rows], dtype=np.float64)
        if direction == "higher":
            success_scores = success_values
            failure_scores = failure_values
        else:
            success_scores = -success_values
            failure_scores = -failure_values
        success_mean = float(np.mean(success_values)) if success_values.size else float("nan")
        failure_mean = float(np.mean(failure_values)) if failure_values.size else float("nan")
        stats[metric] = {
            "success_direction": direction,
            "success_mean": success_mean,
            "failure_mean": failure_mean,
            "mean_gap_success_minus_failure_m": success_mean - failure_mean,
            "auc": float(_auc_from_scores(success_scores, failure_scores)),
        }
    return stats


def _write_separation_csv(separation_stats: dict[str, Any], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "success_direction",
                "success_mean",
                "failure_mean",
                "mean_gap_success_minus_failure_m",
                "auc",
            ],
        )
        writer.writeheader()
        for metric, stats in separation_stats.items():
            writer.writerow({"metric": metric, **stats})


def _plot(rows: list[dict[str, Any]], output_png: Path) -> None:
    metrics = [
        ("path_length_m", "Path length (m)"),
        ("net_displacement_m", "Net displacement (m)"),
        ("tail_lateral_mean_m", "Tail lateral mean (m)"),
        ("tail_progress_m", "Tail progress (m)"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4), constrained_layout=True)
    labels = ["success", "failure"]
    colors = {"success": "#2ca02c", "failure": "#d62728"}
    for ax, (metric, title) in zip(axes, metrics):
        data = [[float(row[metric]) for row in rows if row["label"] == label] for label in labels]
        ax.boxplot(data, tick_labels=labels, patch_artist=True)
        for i, values in enumerate(data, start=1):
            ax.scatter(np.full(len(values), i), values, color=colors[labels[i - 1]], s=25, zorder=3)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare richer geometric metrics on Phase120 success and Phase121 failure replay CSVs.")
    parser.add_argument("--success-root", type=Path, default=DEFAULT_SUCCESS_ROOT)
    parser.add_argument("--failure-root", type=Path, default=DEFAULT_FAILURE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    success_summary = json.loads((args.success_root / "phase120_cluster_summary.json").read_text())["aggregate"]
    reference_entry = np.asarray(success_summary["entry_mean"], dtype=np.float64)
    reference_axis = np.asarray(success_summary["axis_mean_unit"], dtype=np.float64)
    reference_axis = reference_axis / max(float(np.linalg.norm(reference_axis)), 1e-12)

    rows = [
        *_load_cluster(args.success_root, "success", reference_entry, reference_axis),
        *_load_cluster(args.failure_root, "failure", reference_entry, reference_axis),
    ]
    output = {
        "status": "PASS",
        "reference": {
            "source": str(args.success_root / "phase120_cluster_summary.json"),
            "entry": reference_entry.tolist(),
            "axis_unit": reference_axis.tolist(),
        },
        "group_stats": _group_stats(rows),
        "separation_stats": _separation_stats(rows),
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "success_failure_geometry_metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    _write_separation_csv(output["separation_stats"], args.output_dir / "success_failure_geometry_metric_separation.csv")
    _plot(rows, args.output_dir / "success_failure_geometry_metrics.png")
    print(json.dumps({"group_stats": output["group_stats"], "separation_stats": output["separation_stats"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
