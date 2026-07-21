from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


LEFT_GRIPPER_INDEX = 6
RIGHT_GRIPPER_INDEX = 13


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _percentile_summary(values: np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"min": None, "p50": None, "p90": None, "max": None, "mean": None}
    return {
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _clusters(mask: np.ndarray) -> list[dict[str, int]]:
    indices = np.flatnonzero(np.asarray(mask, dtype=bool))
    if indices.size == 0:
        return []
    rows: list[dict[str, int]] = []
    start = prev = int(indices[0])
    for item in indices[1:]:
        step = int(item)
        if step == prev + 1:
            prev = step
            continue
        rows.append({"start_local_frame": start, "end_local_frame": prev, "length_frames": prev - start + 1})
        start = prev = step
    rows.append({"start_local_frame": start, "end_local_frame": prev, "length_frames": prev - start + 1})
    return rows


def _load_hdf5_channels(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        required = ["observations/qpos", "action"]
        missing = [name for name in required if name not in h5]
        if missing:
            raise KeyError(f"{path} is missing required datasets: {missing}")
        qpos = np.asarray(h5["observations/qpos"], dtype=np.float64)
        action = np.asarray(h5["action"], dtype=np.float64)
        effort = (
            np.asarray(h5["observations/effort"], dtype=np.float64)
            if "observations/effort" in h5
            else np.full_like(qpos, np.nan)
        )
    if qpos.ndim != 2 or action.ndim != 2 or qpos.shape[0] != action.shape[0]:
        raise ValueError(f"expected qpos/action shape (T, D) with same T, got qpos={qpos.shape}, action={action.shape}")
    if qpos.shape[1] <= RIGHT_GRIPPER_INDEX or action.shape[1] <= RIGHT_GRIPPER_INDEX:
        raise ValueError(f"expected at least 14 qpos/action columns, got qpos={qpos.shape}, action={action.shape}")
    if effort.shape != qpos.shape:
        raise ValueError(f"expected effort shape {qpos.shape}, got {effort.shape}")
    return {"qpos": qpos, "action": action, "effort": effort}


def analyze_loaded_gripper_calibration(
    *,
    hdf5_path: Path,
    output_dir: Path,
    side: str = "left",
    start_frame: int = 0,
    end_frame: int | None = None,
    rate_hz: float = 50.0,
    close_action_threshold: float = 0.12,
    qpos_action_gap_threshold: float = 0.25,
    effort_abs_threshold: float = 100.0,
    qpos_plateau_delta_threshold: float = 0.01,
) -> dict[str, Any]:
    channels = _load_hdf5_channels(hdf5_path)
    qpos = channels["qpos"]
    action = channels["action"]
    effort = channels["effort"]
    if side not in {"left", "right"}:
        raise ValueError(f"side must be left or right, got {side!r}")
    idx = LEFT_GRIPPER_INDEX if side == "left" else RIGHT_GRIPPER_INDEX
    end = qpos.shape[0] if end_frame is None else int(end_frame)
    start = int(start_frame)
    if start < 0 or end > qpos.shape[0] or end <= start:
        raise ValueError(f"invalid frame window start={start}, end={end}, frames={qpos.shape[0]}")

    q = qpos[start:end, idx]
    a = action[start:end, idx]
    e = effort[start:end, idx]
    dq = np.zeros_like(q)
    if q.size > 1:
        dq[1:] = np.diff(q)
    close_intent = a <= float(close_action_threshold)
    command_qpos_gap = q - a
    command_gap_large = command_qpos_gap >= float(qpos_action_gap_threshold)
    effort_loaded = np.abs(e) >= float(effort_abs_threshold)
    qpos_plateau = np.abs(dq) <= float(qpos_plateau_delta_threshold)
    loaded_mask = close_intent & command_gap_large & effort_loaded & qpos_plateau

    cluster_rows = []
    for cluster in _clusters(loaded_mask):
        local_slice = slice(cluster["start_local_frame"], cluster["end_local_frame"] + 1)
        cluster_rows.append(
            {
                **cluster,
                "start_hdf5_frame": int(start + cluster["start_local_frame"]),
                "end_hdf5_frame": int(start + cluster["end_local_frame"]),
                "duration_s": float(cluster["length_frames"] / float(rate_hz)),
                "qpos_mean": float(np.mean(q[local_slice])),
                "action_mean": float(np.mean(a[local_slice])),
                "effort_abs_mean": float(np.mean(np.abs(e[local_slice]))),
                "qpos_action_gap_mean": float(np.mean(command_qpos_gap[local_slice])),
            }
        )

    report: dict[str, Any] = {
        "hdf5_path": str(hdf5_path),
        "side": side,
        "gripper_index": int(idx),
        "frame_window": [start, end],
        "sample_count": int(q.size),
        "rate_hz": float(rate_hz),
        "thresholds": {
            "close_action_threshold": float(close_action_threshold),
            "qpos_action_gap_threshold": float(qpos_action_gap_threshold),
            "effort_abs_threshold": float(effort_abs_threshold),
            "qpos_plateau_delta_threshold": float(qpos_plateau_delta_threshold),
        },
        "loaded_close_plateau_frame_count": int(np.count_nonzero(loaded_mask)),
        "loaded_close_plateau_fraction": float(np.count_nonzero(loaded_mask) / max(1, q.size)),
        "loaded_close_plateau_clusters": cluster_rows,
        "longest_loaded_close_plateau_cluster": max(cluster_rows, key=lambda row: row["length_frames"])
        if cluster_rows
        else None,
        "qpos_summary": _percentile_summary(q),
        "action_summary": _percentile_summary(a),
        "effort_abs_summary": _percentile_summary(np.abs(e)),
        "qpos_action_gap_summary": _percentile_summary(command_qpos_gap),
        "qpos_delta_abs_summary": _percentile_summary(np.abs(dq)),
        "interpretation": {
            "status": "LOADED_CLOSE_PLATEAU_DETECTED" if cluster_rows else "NO_LOADED_CLOSE_PLATEAU_DETECTED",
            "qpos_is_not_direct_loaded_pad_gap_sensor": bool(cluster_rows),
            "notes": (
                "A loaded close plateau means action commands continued toward closed, observed gripper qpos "
                "stopped or moved slowly, and effort was high. This supports treating qpos-to-finger-gap as "
                "uncalibrated under soft-bottle load."
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "loaded_gripper_calibration.json"
    md_path = output_dir / "loaded_gripper_calibration.md"
    _write_json(json_path, report)
    md_path.write_text(
        "\n".join(
            [
                "# Loaded Gripper Calibration Diagnostic",
                "",
                f"- HDF5: `{hdf5_path}`",
                f"- side: `{side}`",
                f"- frame window: `{start}` to `{end}`",
                f"- status: `{report['interpretation']['status']}`",
                f"- loaded plateau frames: `{report['loaded_close_plateau_frame_count']}`",
                f"- longest cluster: `{report['longest_loaded_close_plateau_cluster']}`",
                "",
                "This diagnostic does not prove Isaac grasp success. It only identifies frames where qpos is "
                "likely a loaded gripper-joint state rather than a free-space finger-pad gap measurement.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    report["json"] = str(json_path)
    report["markdown"] = str(md_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze HDF5 qpos/action/effort loaded gripper close plateaus.")
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--close-action-threshold", type=float, default=0.12)
    parser.add_argument("--qpos-action-gap-threshold", type=float, default=0.25)
    parser.add_argument("--effort-abs-threshold", type=float, default=100.0)
    parser.add_argument("--qpos-plateau-delta-threshold", type=float, default=0.01)
    args = parser.parse_args()
    report = analyze_loaded_gripper_calibration(
        hdf5_path=args.hdf5,
        output_dir=args.output_dir,
        side=args.side,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        rate_hz=args.rate_hz,
        close_action_threshold=args.close_action_threshold,
        qpos_action_gap_threshold=args.qpos_action_gap_threshold,
        effort_abs_threshold=args.effort_abs_threshold,
        qpos_plateau_delta_threshold=args.qpos_plateau_delta_threshold,
    )
    print(json.dumps({"status": report["interpretation"]["status"], "json": report["json"], "markdown": report["markdown"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
