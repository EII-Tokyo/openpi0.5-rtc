#!/usr/bin/env python3
"""Extract the user-confirmed episode 18 grasp window without signal mixing."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

FRAME_RATE_STATUS = "NOT_EMITTED_UNTIL_SOURCE_PROVEN"
ACTION_DATASET = "/action"
QPOS_DATASET = "/observations/qpos"
LEFT_ARM_SLICE = slice(0, 6)
LEFT_GRIPPER_INDEX = 6


@dataclass(frozen=True)
class EpisodeWindow:
    source_path: str
    source_sha256: str
    frames: np.ndarray
    action: np.ndarray
    qpos: np.ndarray


@dataclass(frozen=True)
class GripperPhases:
    close_command_start_frame: int
    readback_response_start_frame: int
    command_direction: str
    readback_direction: str
    action_threshold: float
    qpos_threshold: float
    baseline_sample_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_1d(values: np.ndarray | list[float], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{name} must be a one-dimensional signal")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def load_episode_window(
    path: str | Path,
    start_frame: int,
    end_frame_inclusive: int,
    *,
    expected_sha256: str | None = None,
) -> EpisodeWindow:
    """Load separate 14-D action and qpos arrays for an inclusive frame range."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"episode path is not a file: {source}")
    if start_frame < 0 or end_frame_inclusive < start_frame:
        raise ValueError("invalid inclusive frame range")

    actual_sha256 = _sha256(source)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            "episode SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    try:
        with h5py.File(source, "r") as handle:
            if ACTION_DATASET not in handle:
                raise ValueError(f"missing action dataset {ACTION_DATASET}")
            if QPOS_DATASET not in handle:
                raise ValueError(f"missing qpos dataset {QPOS_DATASET}")
            action_dataset = handle[ACTION_DATASET]
            qpos_dataset = handle[QPOS_DATASET]
            if action_dataset.ndim != 2 or action_dataset.shape[1] != 14:
                raise ValueError(
                    f"action dataset must be [N,14], got {action_dataset.shape}"
                )
            if qpos_dataset.ndim != 2 or qpos_dataset.shape[1] != 14:
                raise ValueError(
                    f"qpos dataset must be [N,14], got {qpos_dataset.shape}"
                )
            stop = end_frame_inclusive + 1
            if stop > action_dataset.shape[0] or stop > qpos_dataset.shape[0]:
                raise ValueError(
                    "inclusive frame range exceeds action or qpos length"
                )
            action = np.asarray(
                action_dataset[start_frame:stop],
                dtype=np.float64,
            )
            qpos = np.asarray(
                qpos_dataset[start_frame:stop],
                dtype=np.float64,
            )
    except OSError as exc:
        raise ValueError(f"cannot read episode HDF5: {source}") from exc

    if not np.isfinite(action).all() or not np.isfinite(qpos).all():
        raise ValueError("episode window contains NaN or Inf")
    action.setflags(write=False)
    qpos.setflags(write=False)
    frames = np.arange(start_frame, end_frame_inclusive + 1, dtype=np.int64)
    frames.setflags(write=False)
    return EpisodeWindow(
        source_path=str(source),
        source_sha256=actual_sha256,
        frames=frames,
        action=action,
        qpos=qpos,
    )


def robust_change_threshold(
    baseline_signal: np.ndarray | list[float],
) -> float:
    """Return median absolute first difference plus five baseline MADs."""
    signal = _finite_1d(baseline_signal, name="baseline_signal")
    differences = np.diff(signal)
    absolute = np.abs(differences)
    median_absolute_difference = float(np.median(absolute))
    mad = float(np.median(np.abs(absolute - median_absolute_difference)))
    scale = max(1.0, float(np.max(np.abs(signal))))
    epsilon = float(np.finfo(np.float64).eps * scale)
    threshold = median_absolute_difference + 5.0 * mad
    if mad == 0.0:
        threshold += epsilon
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("robust change threshold is not finite and positive")
    return threshold


def _first_change(
    signal: np.ndarray,
    *,
    first_frame: int,
    baseline_sample_count: int,
) -> tuple[int, str, float]:
    threshold = robust_change_threshold(signal[:baseline_sample_count])
    differences = np.diff(signal)
    candidate_indices = np.flatnonzero(np.abs(differences) > threshold)
    if candidate_indices.size == 0:
        raise ValueError("no gripper change exceeds the robust baseline")
    difference_index = int(candidate_indices[0])
    direction = "increasing" if differences[difference_index] > 0.0 else "decreasing"
    destination_frame = first_frame + difference_index + 1
    return destination_frame, direction, threshold


def detect_gripper_phases(
    action_gripper: np.ndarray | list[float],
    qpos_gripper: np.ndarray | list[float],
    *,
    first_frame: int,
) -> GripperPhases:
    """Detect command and readback change points independently."""
    action = _finite_1d(action_gripper, name="action_gripper")
    qpos = _finite_1d(qpos_gripper, name="qpos_gripper")
    if action.shape != qpos.shape:
        raise ValueError("action and qpos gripper signals must have equal shape")
    if first_frame < 0:
        raise ValueError("first_frame must be non-negative")
    baseline_sample_count = min(10, max(3, action.size // 4))
    if baseline_sample_count >= action.size:
        raise ValueError("signal is too short for baseline and change detection")

    command_frame, command_direction, action_threshold = _first_change(
        action,
        first_frame=first_frame,
        baseline_sample_count=baseline_sample_count,
    )
    readback_frame, readback_direction, qpos_threshold = _first_change(
        qpos,
        first_frame=first_frame,
        baseline_sample_count=baseline_sample_count,
    )
    return GripperPhases(
        close_command_start_frame=command_frame,
        readback_response_start_frame=readback_frame,
        command_direction=command_direction,
        readback_direction=readback_direction,
        action_threshold=action_threshold,
        qpos_threshold=qpos_threshold,
        baseline_sample_count=baseline_sample_count,
    )


def build_frame_records(
    window: EpisodeWindow,
    phases: GripperPhases,
) -> list[dict[str, Any]]:
    """Build JSON-compatible records with command and readback kept separate."""
    records: list[dict[str, Any]] = []
    previous_action = window.action[0, LEFT_ARM_SLICE]
    previous_qpos = window.qpos[0, LEFT_ARM_SLICE]
    for index, frame_value in enumerate(window.frames):
        frame = int(frame_value)
        action_arm = window.action[index, LEFT_ARM_SLICE]
        qpos_arm = window.qpos[index, LEFT_ARM_SLICE]
        action_step_norm = (
            0.0
            if index == 0
            else float(np.linalg.norm(action_arm - previous_action))
        )
        qpos_step_norm = (
            0.0 if index == 0 else float(np.linalg.norm(qpos_arm - previous_qpos))
        )
        labels = [
            (
                "close_command_active"
                if frame >= phases.close_command_start_frame
                else "before_close_command"
            ),
            (
                "readback_response_active"
                if frame >= phases.readback_response_start_frame
                else "readback_not_responding"
            ),
        ]
        records.append(
            {
                "frame": frame,
                "action_left_arm_6d": [float(value) for value in action_arm],
                "qpos_left_arm_6d": [float(value) for value in qpos_arm],
                "action_left_gripper": float(
                    window.action[index, LEFT_GRIPPER_INDEX]
                ),
                "qpos_left_gripper": float(
                    window.qpos[index, LEFT_GRIPPER_INDEX]
                ),
                "action_step_norm": action_step_norm,
                "qpos_step_norm": qpos_step_norm,
                "phase_labels": labels,
            }
        )
        previous_action = action_arm
        previous_qpos = qpos_arm
    return records


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_episode_reports(
    window: EpisodeWindow,
    phases: GripperPhases,
    *,
    json_output: str | Path,
    csv_output: str | Path,
) -> dict[str, Any]:
    """Write deterministic JSON and CSV evidence for one episode window."""
    records = build_frame_records(window, phases)
    ordered = (
        phases.close_command_start_frame
        <= phases.readback_response_start_frame
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS" if ordered else "FAIL",
        "source": {
            "path": window.source_path,
            "sha256": window.source_sha256,
            "action_dataset": ACTION_DATASET,
            "qpos_dataset": QPOS_DATASET,
        },
        "window": {
            "start_frame": int(window.frames[0]),
            "end_frame_inclusive": int(window.frames[-1]),
        },
        "frame_count": len(records),
        "frame_rate_status": FRAME_RATE_STATUS,
        "signal_semantics": {
            "action": "COMMAND",
            "qpos": "RUNTIME_READBACK",
            "mixed": False,
            "left_arm_indices": list(range(6)),
            "left_gripper_index": LEFT_GRIPPER_INDEX,
        },
        "phase_detection": {
            "close_command_start_frame": phases.close_command_start_frame,
            "readback_response_start_frame": (
                phases.readback_response_start_frame
            ),
            "command_direction": phases.command_direction,
            "readback_direction": phases.readback_direction,
            "action_threshold": phases.action_threshold,
            "qpos_threshold": phases.qpos_threshold,
            "baseline_sample_count": phases.baseline_sample_count,
            "ordered": ordered,
        },
        "frames": records,
    }
    json_text = json.dumps(
        report,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    _atomic_text(Path(json_output), f"{json_text}\n")

    csv_path = Path(csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_name(f".{csv_path.name}.tmp-{os.getpid()}")
    fieldnames = list(records[0])
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["action_left_arm_6d"] = json.dumps(
                row["action_left_arm_6d"],
                separators=(",", ":"),
            )
            row["qpos_left_arm_6d"] = json.dumps(
                row["qpos_left_arm_6d"],
                separators=(",", ":"),
            )
            row["phase_labels"] = "|".join(row["phase_labels"])
            writer.writerow(row)
    os.replace(temporary, csv_path)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--start-frame", required=True, type=int)
    parser.add_argument("--end-frame-inclusive", required=True, type=int)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--csv-output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    window = load_episode_window(
        args.input,
        args.start_frame,
        args.end_frame_inclusive,
        expected_sha256=args.expected_sha256,
    )
    phases = detect_gripper_phases(
        window.action[:, LEFT_GRIPPER_INDEX],
        window.qpos[:, LEFT_GRIPPER_INDEX],
        first_frame=args.start_frame,
    )
    report = write_episode_reports(
        window,
        phases,
        json_output=args.json_output,
        csv_output=args.csv_output,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "frame_count": report["frame_count"],
                "phase_detection": report["phase_detection"],
                "json_output": str(args.json_output.resolve()),
                "csv_output": str(args.csv_output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
