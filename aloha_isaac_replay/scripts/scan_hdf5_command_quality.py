from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from aloha_isaac_replay.scripts.analyze_hdf5_command_spike_feasibility import (
    _cluster_steps,
    _fmt,
    _load_mapping,
    _load_qpos,
    _mapped_values,
    _percentile_summary,
)


CLASSIFICATION_SEVERITY = {
    "COMMAND_SMOOTHNESS_PASS": 0,
    "SINGLE_SPIKE_RESIDUAL": 1,
    "REPEATED_SPIKE_CLUSTER": 2,
    "GLOBAL_HIGH_RATE_COMMAND_MISMATCH": 3,
}


def _classify_joint(*, spike_count: int, total_steps: int, clusters: list[dict[str, Any]]) -> str:
    if spike_count == 0:
        return "COMMAND_SMOOTHNESS_PASS"
    if spike_count / max(1, total_steps) > 0.1:
        return "GLOBAL_HIGH_RATE_COMMAND_MISMATCH"
    if spike_count > 1 or any(int(row.get("cluster_count") or 0) > 1 for row in clusters):
        return "REPEATED_SPIKE_CLUSTER"
    return "SINGLE_SPIKE_RESIDUAL"


def _worst_classification(classifications: list[str]) -> str:
    if not classifications:
        return "COMMAND_SMOOTHNESS_PASS"
    return max(classifications, key=lambda item: CLASSIFICATION_SEVERITY.get(item, -1))


def _recommendation(classification: str) -> str:
    if classification == "COMMAND_SMOOTHNESS_PASS":
        return "ALLOW_ISAAC_REPLAY_GATE"
    if classification == "SINGLE_SPIKE_RESIDUAL":
        return "REVIEW_SPIKE_BEFORE_REPLAY_TUNING"
    return "BLOCK_CCD_FIX_COMMAND_CONTINUITY_FIRST"


def scan_hdf5_command_windows(
    *,
    hdf5_path: Path,
    mapping_path: Path,
    output_dir: Path,
    start_frame: int = 0,
    end_frame: int | None = None,
    hdf5_rate_hz: float = 50.0,
    window_size_frames: int = 570,
    window_stride_frames: int = 50,
    spike_threshold_rad_s: float = 2.0,
    strong_velocity_threshold_rad_s: float = 3.0,
    accel_warning_threshold_rad_s2: float = 100.0,
) -> dict[str, Any]:
    qpos = _load_qpos(hdf5_path)
    mapping = _load_mapping(mapping_path)
    resolved_start = int(start_frame)
    resolved_end = int(qpos.shape[0] if end_frame is None else end_frame)
    if window_size_frames <= 1 or window_stride_frames <= 0:
        raise ValueError("window_size_frames must be > 1 and window_stride_frames must be > 0")
    if resolved_start < 0 or resolved_end > qpos.shape[0] or resolved_end <= resolved_start + 1:
        raise ValueError(
            f"invalid frame window start={resolved_start}, end={resolved_end}, qpos_frames={qpos.shape[0]}"
        )
    if window_size_frames > resolved_end - resolved_start:
        raise ValueError(
            f"window_size_frames={window_size_frames} exceeds scan range {resolved_end - resolved_start}"
        )

    dt = 1.0 / float(hdf5_rate_hz)
    rows: list[dict[str, Any]] = []
    for window_start in range(resolved_start, resolved_end - int(window_size_frames) + 1, int(window_stride_frames)):
        window_end = window_start + int(window_size_frames)
        target_frames = np.arange(window_start + 1, window_end, dtype=np.int64)
        target_steps = np.arange(len(target_frames), dtype=np.int64)
        joint_classes: list[str] = []
        total_spikes = 0
        total_strong_spikes = 0
        total_accel_warnings = 0
        max_velocity = 0.0
        max_velocity_joint = None
        max_velocity_frame = None
        max_accel = 0.0
        for entry in mapping:
            name = str(entry["canonical_name"])
            values = _mapped_values(qpos, entry)
            current = values[target_frames]
            previous = values[target_frames - 1]
            prev_prev = values[np.maximum(target_frames - 2, 0)]
            delta = current - previous
            prev_delta = previous - prev_prev
            abs_velocity = np.abs(delta / dt)
            abs_accel = np.abs((delta - prev_delta) / (dt * dt))
            spike_count = int(np.sum(abs_velocity > float(spike_threshold_rad_s)))
            strong_count = int(np.sum(abs_velocity > float(strong_velocity_threshold_rad_s)))
            accel_count = int(np.sum(abs_accel > float(accel_warning_threshold_rad_s2)))
            spike_steps = [int(step) for step in target_steps[abs_velocity > float(spike_threshold_rad_s)]]
            clusters = [
                {"cluster_count": int(row["length_steps"])}
                for row in _cluster_steps(spike_steps, max_gap=3)
            ]
            joint_classes.append(
                _classify_joint(spike_count=spike_count, total_steps=len(target_steps), clusters=clusters)
            )
            total_spikes += spike_count
            total_strong_spikes += strong_count
            total_accel_warnings += accel_count
            joint_max_i = int(np.argmax(abs_velocity)) if abs_velocity.size else 0
            joint_max = float(abs_velocity[joint_max_i]) if abs_velocity.size else 0.0
            if joint_max > max_velocity:
                max_velocity = joint_max
                max_velocity_joint = name
                max_velocity_frame = int(target_frames[joint_max_i])
            max_accel = max(max_accel, float(np.max(abs_accel)) if abs_accel.size else 0.0)
        classification = _worst_classification(joint_classes)
        rows.append(
            {
                "window_start_frame": window_start,
                "window_end_frame": window_end,
                "duration_s": float((window_size_frames - 1) * dt),
                "classification": classification,
                "recommendation": _recommendation(classification),
                "total_spikes": total_spikes,
                "total_strong_spikes": total_strong_spikes,
                "total_accel_warnings": total_accel_warnings,
                "max_abs_target_velocity": max_velocity,
                "max_abs_target_velocity_joint": max_velocity_joint,
                "max_abs_target_velocity_hdf5_frame": max_velocity_frame,
                "max_abs_target_acceleration": max_accel,
            }
        )

    ranked = sorted(
        rows,
        key=lambda row: (
            CLASSIFICATION_SEVERITY.get(str(row["classification"]), 99),
            int(row["total_spikes"]),
            float(row["max_abs_target_velocity"]),
        ),
    )
    report = {
        "script_name": Path(__file__).name,
        "read_only": True,
        "episode_path": str(hdf5_path),
        "mapping_path": str(mapping_path),
        "scan_range": {"start": resolved_start, "end": resolved_end},
        "hdf5_rate_hz": float(hdf5_rate_hz),
        "window_size_frames": int(window_size_frames),
        "window_stride_frames": int(window_stride_frames),
        "window_count": len(rows),
        "spike_threshold_rad_s": float(spike_threshold_rad_s),
        "strong_velocity_threshold_rad_s": float(strong_velocity_threshold_rad_s),
        "accel_warning_threshold_rad_s2": float(accel_warning_threshold_rad_s2),
        "best_windows": ranked[:20],
        "worst_windows": list(reversed(ranked[-20:])),
        "windows": rows,
    }
    json_path = output_dir / "hdf5_command_quality_windows.json"
    csv_path = output_dir / "hdf5_command_quality_windows.csv"
    md_path = output_dir / "hdf5_command_quality_windows.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_window_csv(csv_path, rows)
    _write_window_markdown(md_path, report)
    report["json"] = str(json_path)
    report["csv"] = str(csv_path)
    report["markdown"] = str(md_path)
    return report


def scan_hdf5_command_quality(
    *,
    hdf5_path: Path,
    mapping_path: Path,
    output_dir: Path,
    start_frame: int = 0,
    end_frame: int | None = None,
    hdf5_rate_hz: float = 50.0,
    spike_threshold_rad_s: float = 2.0,
    strong_velocity_threshold_rad_s: float = 3.0,
    accel_warning_threshold_rad_s2: float = 100.0,
    cluster_gap_steps: int = 3,
) -> dict[str, Any]:
    qpos = _load_qpos(hdf5_path)
    mapping = _load_mapping(mapping_path)
    resolved_end = int(qpos.shape[0] if end_frame is None else end_frame)
    resolved_start = int(start_frame)
    if resolved_start < 0 or resolved_end > qpos.shape[0] or resolved_end <= resolved_start + 1:
        raise ValueError(
            f"invalid frame window start={resolved_start}, end={resolved_end}, qpos_frames={qpos.shape[0]}"
        )

    dt = 1.0 / float(hdf5_rate_hz)
    target_frames = np.arange(resolved_start + 1, resolved_end, dtype=np.int64)
    target_steps = np.arange(len(target_frames), dtype=np.int64)
    per_joint: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    classifications: list[str] = []

    for entry in mapping:
        name = str(entry["canonical_name"])
        values = _mapped_values(qpos, entry)
        current = values[target_frames]
        previous = values[target_frames - 1]
        prev_prev = values[np.maximum(target_frames - 2, 0)]
        delta = current - previous
        previous_delta = previous - prev_prev
        velocity = delta / dt
        acceleration = (delta - previous_delta) / (dt * dt)
        abs_velocity = np.abs(velocity)
        abs_acceleration = np.abs(acceleration)
        spike_mask = abs_velocity > float(spike_threshold_rad_s)
        strong_spike_mask = abs_velocity > float(strong_velocity_threshold_rad_s)
        accel_warning_mask = abs_acceleration > float(accel_warning_threshold_rad_s2)
        spike_steps = [int(step) for step in target_steps[spike_mask]]
        clusters = []
        for cluster in _cluster_steps(spike_steps, max_gap=int(cluster_gap_steps)):
            mask = (
                (target_steps >= int(cluster["start_step"]))
                & (target_steps <= int(cluster["end_step"]))
                & spike_mask
            )
            cluster_steps = target_steps[mask]
            cluster_vel = abs_velocity[mask]
            cluster_accel = abs_acceleration[mask]
            peak_i = int(np.argmax(cluster_vel)) if cluster_vel.size else 0
            clusters.append(
                {
                    "joint_name": name,
                    "cluster_start_step": int(cluster["start_step"]),
                    "cluster_end_step": int(cluster["end_step"]),
                    "cluster_count": int(cluster_steps.size),
                    "cluster_span_steps": int(cluster["end_step"] - cluster["start_step"] + 1),
                    "cluster_span_seconds": float((cluster["end_step"] - cluster["start_step"] + 1) * dt),
                    "peak_step": int(cluster_steps[peak_i]) if cluster_steps.size else None,
                    "peak_hdf5_frame": int(resolved_start + 1 + int(cluster_steps[peak_i]))
                    if cluster_steps.size
                    else None,
                    "peak_abs_target_velocity": float(cluster_vel[peak_i]) if cluster_vel.size else None,
                    "peak_abs_target_acceleration": float(cluster_accel[peak_i]) if cluster_accel.size else None,
                }
            )
        classification = _classify_joint(
            spike_count=int(np.sum(spike_mask)),
            total_steps=len(target_steps),
            clusters=clusters,
        )
        classifications.append(classification)
        max_velocity_idx = int(np.argmax(abs_velocity)) if abs_velocity.size else 0
        max_accel_idx = int(np.argmax(abs_acceleration)) if abs_acceleration.size else 0
        row = {
            "joint": name,
            "dataset_index": int(entry["dataset_index"]),
            "unit": str(entry.get("unit", "")),
            "classification": classification,
            "recommendation": _recommendation(classification),
            "spike_count": int(np.sum(spike_mask)),
            "strong_spike_count": int(np.sum(strong_spike_mask)),
            "accel_warning_count": int(np.sum(accel_warning_mask)),
            "spike_fraction": float(np.sum(spike_mask) / max(1, len(target_steps))),
            "cluster_count": len(clusters),
            "max_abs_target_velocity": float(abs_velocity[max_velocity_idx]) if abs_velocity.size else None,
            "max_abs_target_velocity_step": int(target_steps[max_velocity_idx]) if target_steps.size else None,
            "max_abs_target_velocity_hdf5_frame": int(target_frames[max_velocity_idx]) if target_frames.size else None,
            "max_abs_target_acceleration": float(abs_acceleration[max_accel_idx]) if abs_acceleration.size else None,
            "max_abs_target_acceleration_step": int(target_steps[max_accel_idx]) if target_steps.size else None,
            "max_abs_target_acceleration_hdf5_frame": int(target_frames[max_accel_idx])
            if target_frames.size
            else None,
        }
        rows.append(row)
        per_joint[name] = {
            **row,
            "target_delta": _percentile_summary(np.abs(delta)),
            "target_velocity": _percentile_summary(abs_velocity),
            "target_acceleration": _percentile_summary(abs_acceleration),
            "spike_clusters": clusters,
        }

    overall_classification = _worst_classification(classifications)
    report = {
        "script_name": Path(__file__).name,
        "read_only": True,
        "formal_replay_targets_modified": False,
        "deleted_frames": 0,
        "smoothed_frames": 0,
        "interpolated_frames": 0,
        "episode_path": str(hdf5_path),
        "mapping_path": str(mapping_path),
        "frame_window": {
            "start": resolved_start,
            "end": resolved_end,
            "target_steps": int(resolved_end - resolved_start - 1),
        },
        "hdf5_rate_hz": float(hdf5_rate_hz),
        "spike_threshold_rad_s": float(spike_threshold_rad_s),
        "strong_velocity_threshold_rad_s": float(strong_velocity_threshold_rad_s),
        "accel_warning_threshold_rad_s2": float(accel_warning_threshold_rad_s2),
        "cluster_gap_steps": int(cluster_gap_steps),
        "overall_classification": overall_classification,
        "overall_recommendation": _recommendation(overall_classification),
        "per_joint": per_joint,
        "joint_rows": sorted(
            rows,
            key=lambda row: (
                CLASSIFICATION_SEVERITY.get(str(row["classification"]), -1),
                float(row["max_abs_target_velocity"] or 0.0),
            ),
            reverse=True,
        ),
        "notes": [
            "This scanner reads raw HDF5 qpos and mapping only. It does not start Isaac Sim.",
            "It does not delete, smooth, interpolate, or otherwise modify the replay target sequence.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "hdf5_command_quality.json"
    csv_path = output_dir / "hdf5_command_quality_per_joint.csv"
    md_path = output_dir / "hdf5_command_quality.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(csv_path, report["joint_rows"])
    _write_markdown(md_path, report)
    report["json"] = str(json_path)
    report["csv"] = str(csv_path)
    report["markdown"] = str(md_path)
    return report


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "joint",
        "dataset_index",
        "unit",
        "classification",
        "recommendation",
        "spike_count",
        "strong_spike_count",
        "accel_warning_count",
        "spike_fraction",
        "cluster_count",
        "max_abs_target_velocity",
        "max_abs_target_velocity_step",
        "max_abs_target_velocity_hdf5_frame",
        "max_abs_target_acceleration",
        "max_abs_target_acceleration_step",
        "max_abs_target_acceleration_hdf5_frame",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_window_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "window_start_frame",
        "window_end_frame",
        "duration_s",
        "classification",
        "recommendation",
        "total_spikes",
        "total_strong_spikes",
        "total_accel_warnings",
        "max_abs_target_velocity",
        "max_abs_target_velocity_joint",
        "max_abs_target_velocity_hdf5_frame",
        "max_abs_target_acceleration",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_window_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# HDF5 Command Quality Window Scan",
        "",
        "This is a read-only preflight scan over sliding HDF5 windows.",
        "",
        f"- episode: `{report['episode_path']}`",
        f"- scan range: `{report['scan_range']}`",
        f"- window size: `{report['window_size_frames']}` frames",
        f"- window stride: `{report['window_stride_frames']}` frames",
        f"- window count: `{report['window_count']}`",
        f"- velocity threshold: `{report['spike_threshold_rad_s']}` rad/s",
        "",
        "## Best Windows",
        "",
        "| start | end | class | spikes | strong | max vel | max vel joint | max vel frame |",
        "| ---: | ---: | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in report["best_windows"]:
        lines.append(
            "| {start} | {end} | `{cls}` | {spikes} | {strong} | {vel} | {joint} | {frame} |".format(
                start=row["window_start_frame"],
                end=row["window_end_frame"],
                cls=row["classification"],
                spikes=row["total_spikes"],
                strong=row["total_strong_spikes"],
                vel=_fmt(row["max_abs_target_velocity"]),
                joint=row["max_abs_target_velocity_joint"],
                frame=row["max_abs_target_velocity_hdf5_frame"],
            )
        )
    lines.extend(
        [
            "",
            "## Worst Windows",
            "",
            "| start | end | class | spikes | strong | max vel | max vel joint | max vel frame |",
            "| ---: | ---: | --- | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for row in report["worst_windows"]:
        lines.append(
            "| {start} | {end} | `{cls}` | {spikes} | {strong} | {vel} | {joint} | {frame} |".format(
                start=row["window_start_frame"],
                end=row["window_end_frame"],
                cls=row["classification"],
                spikes=row["total_spikes"],
                strong=row["total_strong_spikes"],
                vel=_fmt(row["max_abs_target_velocity"]),
                joint=row["max_abs_target_velocity_joint"],
                frame=row["max_abs_target_velocity_hdf5_frame"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# HDF5 Command Quality Scan",
        "",
        "This is a read-only preflight scan. It does not start Isaac Sim and does not modify replay targets.",
        "",
        "## Summary",
        "",
        f"- episode: `{report['episode_path']}`",
        f"- mapping: `{report['mapping_path']}`",
        f"- frame window: `{report['frame_window']}`",
        f"- HDF5 rate: `{report['hdf5_rate_hz']}` Hz",
        f"- velocity threshold: `{report['spike_threshold_rad_s']}` rad/s",
        f"- strong velocity threshold: `{report['strong_velocity_threshold_rad_s']}` rad/s",
        f"- acceleration warning threshold: `{report['accel_warning_threshold_rad_s2']}` rad/s^2",
        f"- overall classification: `{report['overall_classification']}`",
        f"- recommendation: `{report['overall_recommendation']}`",
        f"- formal replay targets modified: `{report['formal_replay_targets_modified']}`",
        f"- deleted/smoothed/interpolated frames: `{report['deleted_frames']}` / `{report['smoothed_frames']}` / `{report['interpolated_frames']}`",
        "",
        "## Per-Joint Rows",
        "",
        "| joint | class | max vel | vel frame | spikes | strong | accel warn | clusters | max accel |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["joint_rows"]:
        lines.append(
            "| {joint} | `{cls}` | {vel} | {frame} | {spikes} | {strong} | {accel_warn} | {clusters} | {accel} |".format(
                joint=row["joint"],
                cls=row["classification"],
                vel=_fmt(row["max_abs_target_velocity"]),
                frame=row["max_abs_target_velocity_hdf5_frame"],
                spikes=row["spike_count"],
                strong=row["strong_spike_count"],
                accel_warn=row["accel_warning_count"],
                clusters=row["cluster_count"],
                accel=_fmt(row["max_abs_target_acceleration"]),
            )
        )
    top = report["joint_rows"][0] if report["joint_rows"] else {}
    top_joint = (report["per_joint"] or {}).get(top.get("joint"), {})
    lines.extend(
        [
            "",
            "## Top Joint Clusters",
            "",
            f"- top joint: `{top.get('joint')}`",
            "",
            "| cluster | steps | count | seconds | peak frame | peak velocity | peak acceleration |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for i, cluster in enumerate(top_joint.get("spike_clusters") or [], start=1):
        lines.append(
            "| {i} | {start}-{end} | {count} | {seconds} | {frame} | {vel} | {accel} |".format(
                i=i,
                start=cluster.get("cluster_start_step"),
                end=cluster.get("cluster_end_step"),
                count=cluster.get("cluster_count"),
                seconds=_fmt(cluster.get("cluster_span_seconds")),
                frame=cluster.get("peak_hdf5_frame"),
                vel=_fmt(cluster.get("peak_abs_target_velocity")),
                accel=_fmt(cluster.get("peak_abs_target_acceleration")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only raw HDF5 command quality preflight scanner.")
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--hdf5-rate-hz", type=float, default=50.0)
    parser.add_argument("--spike-threshold-rad-s", type=float, default=2.0)
    parser.add_argument("--strong-velocity-threshold-rad-s", type=float, default=3.0)
    parser.add_argument("--accel-warning-threshold-rad-s2", type=float, default=100.0)
    parser.add_argument("--cluster-gap-steps", type=int, default=3)
    parser.add_argument(
        "--window-size-frames",
        type=int,
        help="Optional sliding-window scan size in HDF5 frames. Writes extra window JSON/CSV/Markdown.",
    )
    parser.add_argument("--window-stride-frames", type=int, default=50)
    args = parser.parse_args()
    report = scan_hdf5_command_quality(
        hdf5_path=args.hdf5,
        mapping_path=args.mapping,
        output_dir=args.output_dir,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        hdf5_rate_hz=args.hdf5_rate_hz,
        spike_threshold_rad_s=args.spike_threshold_rad_s,
        strong_velocity_threshold_rad_s=args.strong_velocity_threshold_rad_s,
        accel_warning_threshold_rad_s2=args.accel_warning_threshold_rad_s2,
        cluster_gap_steps=args.cluster_gap_steps,
    )
    response = {"json": report["json"], "csv": report["csv"], "markdown": report["markdown"]}
    if args.window_size_frames:
        window_report = scan_hdf5_command_windows(
            hdf5_path=args.hdf5,
            mapping_path=args.mapping,
            output_dir=args.output_dir,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            hdf5_rate_hz=args.hdf5_rate_hz,
            window_size_frames=args.window_size_frames,
            window_stride_frames=args.window_stride_frames,
            spike_threshold_rad_s=args.spike_threshold_rad_s,
            strong_velocity_threshold_rad_s=args.strong_velocity_threshold_rad_s,
            accel_warning_threshold_rad_s2=args.accel_warning_threshold_rad_s2,
        )
        response["windows_json"] = window_report["json"]
        response["windows_csv"] = window_report["csv"]
        response["windows_markdown"] = window_report["markdown"]
    print(json.dumps(response, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
