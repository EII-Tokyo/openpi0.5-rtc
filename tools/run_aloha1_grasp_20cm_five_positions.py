#!/usr/bin/env python3
"""Run five fixed-seed Bottle500 grasp positions in fresh Isaac processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ISAAC_PYTHON = ROOT / ".venv_issac/bin/python"
ISAAC_LAUNCHER = ROOT / "tools/run_aloha1_grasp_20cm_gui.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_five_position_summary(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate the five machine runs before visual/user review."""

    ids = [str(record.get("position_id")) for record in records]
    process_ids = [
        int(run.get("process_id", -1))
        for record in records
        for run in (
            record.get("primary", {}),
            record.get("collider_repeat", {}),
        )
        if int(run.get("process_id", -1)) > 0
    ]
    per_position_gates: list[dict[str, Any]] = []
    for record in records:
        primary = record.get("primary", {})
        collider = record.get("collider_repeat", {})
        gates = {
            "primary_machine_pass": (
                primary.get("machine_status") == "PASS"
            ),
            "collider_repeat_machine_pass": (
                collider.get("machine_status") == "PASS"
            ),
            "paired_signature_equal": (
                primary.get("deterministic_signature")
                == collider.get("deterministic_signature")
                and primary.get("deterministic_signature") is not None
            ),
            "raw_and_annotated_video_present": (
                primary.get("video_count") == 2
            ),
            "collision_screenshot_records_complete": (
                collider.get("collision_record_count") == 24
            ),
        }
        per_position_gates.append(
            {
                "position_id": record.get("position_id"),
                "gates": gates,
                "status": (
                    "PASS" if all(gates.values()) else "FAIL"
                ),
            }
        )
    machine_pass_count = sum(
        item["status"] == "PASS" for item in per_position_gates
    )
    machine_pass = (
        len(records) == 5
        and len(set(ids)) == 5
        and len(process_ids) == 10
        and len(set(process_ids)) == 10
        and machine_pass_count == 5
    )
    return {
        "status": "PARTIAL" if machine_pass else "FAIL",
        "machine_status": "PASS" if machine_pass else "FAIL",
        "machine_pass_count": machine_pass_count,
        "required_position_count": 5,
        "video_count": sum(
            record.get("primary", {}).get("video_count") == 2
            for record in records
        ),
        "fresh_process_count": len(set(process_ids)),
        "per_position_gates": per_position_gates,
        "visual_model_review": "NOT_RUN",
        "user_confirmation": "NOT_RUN",
        "promotion_status": (
            "AWAITING_VISUAL_MODEL_REVIEW"
            if machine_pass
            else "BLOCKED_BY_MACHINE_FAILURE"
        ),
        "task8": "NOT_RUN",
    }


def _run_logged(
    *,
    command: list[str],
    log_path: Path,
    timeout_s: float,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("wb") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        process_id = int(process.pid)
        timed_out = False
        try:
            exit_code = int(process.wait(timeout=timeout_s))
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                exit_code = int(process.wait(timeout=20.0))
            except subprocess.TimeoutExpired:
                process.kill()
                exit_code = int(process.wait(timeout=20.0))
    return {
        "process_id": process_id,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "runtime_seconds": time.perf_counter() - started,
        "command": command,
        "log_absolute_path": str(log_path.resolve()),
        "log_sha256": _sha256(log_path),
    }


def _read_run_evidence(
    *,
    artifact_root: Path,
    process: dict[str, Any],
    collision_repeat: bool,
) -> dict[str, Any]:
    result = dict(process)
    runtime_path = artifact_root / "aloha1_grasp_20cm_runtime.json"
    candidate_path = (
        artifact_root
        / "video_attempt_001/video/candidate_manifest.json"
    )
    if not runtime_path.is_file():
        result.update(
            {
                "machine_status": "MISSING_REPORT",
                "video_count": 0,
                "collision_record_count": 0,
            }
        )
        return result
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    result.update(
        {
            "machine_status": runtime.get("status"),
            "machine_reason": runtime.get("reason"),
            "deterministic_signature": runtime.get(
                "deterministic_signature"
            ),
            "runtime_report_absolute_path": str(runtime_path.resolve()),
            "runtime_report_sha256": _sha256(runtime_path),
            "metrics": runtime.get("metrics"),
            "bottle_random_position": runtime.get(
                "bottle_random_position"
            ),
            "stage": runtime.get("stage"),
        }
    )
    if not candidate_path.is_file():
        result.update(
            {
                "candidate_status": "MISSING",
                "video_count": 0,
                "collision_record_count": 0,
            }
        )
        return result
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    videos = list(candidate.get("videos", []))
    collision = candidate.get("collision_evidence", {})
    result.update(
        {
            "candidate_status": candidate.get("status"),
            "candidate_manifest_absolute_path": str(
                candidate_path.resolve()
            ),
            "candidate_manifest_sha256": _sha256(candidate_path),
            "videos": videos if not collision_repeat else [],
            "video_count": len(videos) if not collision_repeat else 0,
            "collision_status": collision.get("status"),
            "collision_records": (
                list(collision.get("records", []))
                if collision_repeat
                else []
            ),
            "collision_record_count": (
                len(collision.get("records", []))
                if collision_repeat
                else 0
            ),
        }
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument(
        "--additional-lift-margin-m",
        type=float,
        default=0.0,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    preflight_path = args.preflight.resolve(strict=True)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    additional_lift_margin_m = float(args.additional_lift_margin_m)
    if (
        not math.isfinite(additional_lift_margin_m)
        or additional_lift_margin_m < 0.0
    ):
        raise ValueError(
            "additional lift margin must be finite and non-negative"
        )
    if not math.isclose(
        float(preflight.get("additional_lift_margin_m", 0.0)),
        additional_lift_margin_m,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError(
            "runner lift margin does not match frozen preflight"
        )
    if (
        preflight.get("status") != "PASS"
        or preflight.get("selected_position_count") != 5
    ):
        raise RuntimeError("five-position preflight is not PASS")
    if not ISAAC_PYTHON.is_file():
        raise RuntimeError(f"missing project Isaac Python: {ISAAC_PYTHON}")
    artifact_root = args.artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_path = args.output.resolve()
    records: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = {
        "schema_version": 1,
        "status": "PARTIAL",
        "machine_status": "NOT_RUN",
        "preflight": {
            "absolute_path": str(preflight_path),
            "sha256": _sha256(preflight_path),
        },
        "additional_lift_margin_m": additional_lift_margin_m,
        "positions": records,
        "task8": "NOT_RUN",
    }
    _atomic_json(output_path, aggregate)
    for selected in preflight["selected_positions"]:
        position_id = str(selected["position_id"])
        offset = [float(value) for value in selected["offset_xy_m"]]
        position_root = artifact_root / position_id
        primary_root = position_root / "primary"
        collider_root = position_root / "collider_repeat"
        common = [
            str(ISAAC_PYTHON),
            str(ISAAC_LAUNCHER),
            "--autorun",
            "--close-after-terminal",
            "--bottle-offset-x-m",
            repr(offset[0]),
            "--bottle-offset-y-m",
            repr(offset[1]),
            "--additional-lift-margin-m",
            repr(additional_lift_margin_m),
        ]
        primary_command = [
            *common,
            "--artifact-root",
            str(primary_root),
            "--skip-collider-evidence",
        ]
        primary_process = _run_logged(
            command=primary_command,
            log_path=position_root / "primary.log",
            timeout_s=float(args.timeout_s),
        )
        primary = _read_run_evidence(
            artifact_root=primary_root,
            process=primary_process,
            collision_repeat=False,
        )
        collider_command = [
            *common,
            "--artifact-root",
            str(collider_root),
        ]
        collider_process = _run_logged(
            command=collider_command,
            log_path=position_root / "collider_repeat.log",
            timeout_s=float(args.timeout_s),
        )
        collider = _read_run_evidence(
            artifact_root=collider_root,
            process=collider_process,
            collision_repeat=True,
        )
        records.append(
            {
                "position_id": position_id,
                "preflight_candidate_index": int(
                    selected["candidate_index"]
                ),
                "offset_xy_m": offset,
                "preflight_bottle_position_world_m": selected[
                    "bottle_position_world_m"
                ],
                "preflight_ik": selected["ik"],
                "primary": primary,
                "collider_repeat": collider,
                "visual_model_review": "NOT_RUN",
                "user_confirmation": "NOT_RUN",
            }
        )
        aggregate = {
            "schema_version": 1,
            **build_five_position_summary(records),
            "preflight": {
                "absolute_path": str(preflight_path),
                "sha256": _sha256(preflight_path),
            },
            "additional_lift_margin_m": additional_lift_margin_m,
            "positions": records,
            "boundaries": {
                "real_robot": False,
                "remote_103": False,
                "source_stage_modified": False,
                "final_collider_modified": False,
                "task8": "NOT_RUN",
            },
        }
        _atomic_json(output_path, aggregate)
    print(
        json.dumps(
            {
                "status": aggregate["status"],
                "machine_status": aggregate["machine_status"],
                "output": str(output_path),
            },
            sort_keys=True,
        )
    )
    return 0 if aggregate["machine_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
