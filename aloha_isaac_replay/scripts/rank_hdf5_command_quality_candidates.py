from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from aloha_isaac_replay.scripts.scan_hdf5_command_quality import (
    CLASSIFICATION_SEVERITY,
    scan_hdf5_command_windows,
)


def _rank_key(row: dict[str, Any]) -> tuple[int, int, int, float]:
    return (
        CLASSIFICATION_SEVERITY.get(str(row["classification"]), 99),
        int(row["total_spikes"]),
        int(row["total_strong_spikes"]),
        float(row["max_abs_target_velocity"]),
    )


def rank_hdf5_command_quality_candidates(
    *,
    hdf5_paths: list[Path],
    mapping_path: Path,
    output_dir: Path,
    hdf5_rate_hz: float = 50.0,
    window_size_frames: int = 570,
    window_stride_frames: int = 50,
    spike_threshold_rad_s: float = 2.0,
    strong_velocity_threshold_rad_s: float = 3.0,
    accel_warning_threshold_rad_s2: float = 100.0,
    top_per_episode: int = 5,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for hdf5_path in sorted(hdf5_paths):
        episode_output_dir = output_dir / "episodes" / hdf5_path.stem
        try:
            report = scan_hdf5_command_windows(
                hdf5_path=hdf5_path,
                mapping_path=mapping_path,
                output_dir=episode_output_dir,
                hdf5_rate_hz=hdf5_rate_hz,
                window_size_frames=window_size_frames,
                window_stride_frames=window_stride_frames,
                spike_threshold_rad_s=spike_threshold_rad_s,
                strong_velocity_threshold_rad_s=strong_velocity_threshold_rad_s,
                accel_warning_threshold_rad_s2=accel_warning_threshold_rad_s2,
            )
        except Exception as exc:  # pragma: no cover - surfaced in report for batch robustness.
            skipped.append({"episode_path": str(hdf5_path), "reason": f"{type(exc).__name__}: {exc}"})
            continue

        windows = sorted(report["windows"], key=_rank_key)
        best = windows[0] if windows else None
        if best:
            episode_rows.append(
                {
                    "episode_path": str(hdf5_path),
                    "window_count": int(report["window_count"]),
                    "best_classification": best["classification"],
                    "best_total_spikes": int(best["total_spikes"]),
                    "best_total_strong_spikes": int(best["total_strong_spikes"]),
                    "best_max_abs_target_velocity": float(best["max_abs_target_velocity"]),
                    "best_max_abs_target_velocity_joint": best["max_abs_target_velocity_joint"],
                    "best_window_start_frame": int(best["window_start_frame"]),
                    "best_window_end_frame": int(best["window_end_frame"]),
                    "episode_windows_json": str(episode_output_dir / "hdf5_command_quality_windows.json"),
                }
            )
        for row in windows[: max(1, int(top_per_episode))]:
            candidate_rows.append(
                {
                    "episode_path": str(hdf5_path),
                    "window_start_frame": int(row["window_start_frame"]),
                    "window_end_frame": int(row["window_end_frame"]),
                    "duration_s": float(row["duration_s"]),
                    "classification": row["classification"],
                    "recommendation": row["recommendation"],
                    "total_spikes": int(row["total_spikes"]),
                    "total_strong_spikes": int(row["total_strong_spikes"]),
                    "total_accel_warnings": int(row["total_accel_warnings"]),
                    "max_abs_target_velocity": float(row["max_abs_target_velocity"]),
                    "max_abs_target_velocity_joint": row["max_abs_target_velocity_joint"],
                    "max_abs_target_velocity_hdf5_frame": row["max_abs_target_velocity_hdf5_frame"],
                    "max_abs_target_acceleration": float(row["max_abs_target_acceleration"]),
                }
            )

    candidate_rows = sorted(candidate_rows, key=_rank_key)
    episode_rows = sorted(
        episode_rows,
        key=lambda row: (
            CLASSIFICATION_SEVERITY.get(str(row["best_classification"]), 99),
            int(row["best_total_spikes"]),
            int(row["best_total_strong_spikes"]),
            float(row["best_max_abs_target_velocity"]),
        ),
    )
    report = {
        "script_name": Path(__file__).name,
        "read_only": True,
        "formal_replay_targets_modified": False,
        "mapping_path": str(mapping_path),
        "hdf5_rate_hz": float(hdf5_rate_hz),
        "window_size_frames": int(window_size_frames),
        "window_stride_frames": int(window_stride_frames),
        "spike_threshold_rad_s": float(spike_threshold_rad_s),
        "strong_velocity_threshold_rad_s": float(strong_velocity_threshold_rad_s),
        "accel_warning_threshold_rad_s2": float(accel_warning_threshold_rad_s2),
        "episode_count": len(hdf5_paths),
        "scanned_episode_count": len(episode_rows),
        "skipped_episode_count": len(skipped),
        "skipped": skipped,
        "best_episode_rows": episode_rows,
        "best_candidate_windows": candidate_rows[:100],
    }
    json_path = output_dir / "hdf5_command_quality_candidate_ranking.json"
    csv_path = output_dir / "hdf5_command_quality_candidate_windows.csv"
    episodes_csv_path = output_dir / "hdf5_command_quality_episode_summary.csv"
    md_path = output_dir / "hdf5_command_quality_candidate_ranking.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(csv_path, candidate_rows)
    _write_episode_csv(episodes_csv_path, episode_rows)
    _write_markdown(md_path, report)
    report["json"] = str(json_path)
    report["csv"] = str(csv_path)
    report["episodes_csv"] = str(episodes_csv_path)
    report["markdown"] = str(md_path)
    return report


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "episode_path",
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


def _write_episode_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "episode_path",
        "window_count",
        "best_classification",
        "best_total_spikes",
        "best_total_strong_spikes",
        "best_max_abs_target_velocity",
        "best_max_abs_target_velocity_joint",
        "best_window_start_frame",
        "best_window_end_frame",
        "episode_windows_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# HDF5 Command Quality Candidate Ranking",
        "",
        "This is a read-only batch preflight. It does not start Isaac Sim and does not modify replay targets.",
        "",
        f"- mapping: `{report['mapping_path']}`",
        f"- episodes requested/scanned/skipped: `{report['episode_count']}` / `{report['scanned_episode_count']}` / `{report['skipped_episode_count']}`",
        f"- window size: `{report['window_size_frames']}` frames",
        f"- window stride: `{report['window_stride_frames']}` frames",
        f"- velocity threshold: `{report['spike_threshold_rad_s']}` rad/s",
        "",
        "## Best Candidate Windows",
        "",
        "| episode | start | end | class | spikes | strong | max vel | joint | frame |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in report["best_candidate_windows"][:30]:
        lines.append(
            "| `{episode}` | {start} | {end} | `{cls}` | {spikes} | {strong} | {vel:.4f} | {joint} | {frame} |".format(
                episode=Path(row["episode_path"]).name,
                start=row["window_start_frame"],
                end=row["window_end_frame"],
                cls=row["classification"],
                spikes=row["total_spikes"],
                strong=row["total_strong_spikes"],
                vel=float(row["max_abs_target_velocity"]),
                joint=row["max_abs_target_velocity_joint"],
                frame=row["max_abs_target_velocity_hdf5_frame"],
            )
        )
    if report["skipped"]:
        lines.extend(["", "## Skipped", ""])
        for row in report["skipped"]:
            lines.append(f"- `{row['episode_path']}`: {row['reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _collect_hdf5_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.hdf5:
        paths.extend(args.hdf5)
    if args.hdf5_dir:
        paths.extend(sorted(args.hdf5_dir.glob(args.glob)))
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise ValueError("no HDF5 files selected")
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank raw HDF5 replay windows by command smoothness.")
    parser.add_argument("--hdf5", type=Path, action="append", help="HDF5 file to scan. Can be repeated.")
    parser.add_argument("--hdf5-dir", type=Path, help="Directory containing HDF5 files.")
    parser.add_argument("--glob", default="*.hdf5")
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hdf5-rate-hz", type=float, default=50.0)
    parser.add_argument("--window-size-frames", type=int, default=570)
    parser.add_argument("--window-stride-frames", type=int, default=50)
    parser.add_argument("--spike-threshold-rad-s", type=float, default=2.0)
    parser.add_argument("--strong-velocity-threshold-rad-s", type=float, default=3.0)
    parser.add_argument("--accel-warning-threshold-rad-s2", type=float, default=100.0)
    parser.add_argument("--top-per-episode", type=int, default=5)
    args = parser.parse_args()
    report = rank_hdf5_command_quality_candidates(
        hdf5_paths=_collect_hdf5_paths(args),
        mapping_path=args.mapping,
        output_dir=args.output_dir,
        hdf5_rate_hz=args.hdf5_rate_hz,
        window_size_frames=args.window_size_frames,
        window_stride_frames=args.window_stride_frames,
        spike_threshold_rad_s=args.spike_threshold_rad_s,
        strong_velocity_threshold_rad_s=args.strong_velocity_threshold_rad_s,
        accel_warning_threshold_rad_s2=args.accel_warning_threshold_rad_s2,
        top_per_episode=args.top_per_episode,
    )
    print(
        json.dumps(
            {
                "json": report["json"],
                "csv": report["csv"],
                "episodes_csv": report["episodes_csv"],
                "markdown": report["markdown"],
                "scanned_episode_count": report["scanned_episode_count"],
                "skipped_episode_count": report["skipped_episode_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
