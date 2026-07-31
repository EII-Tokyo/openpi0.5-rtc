#!/usr/bin/env python3
"""Freeze visual review and root-cause evidence for five-position grasping."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _lift_metrics(telemetry_path: Path) -> dict[str, Any]:
    records = [
        record
        for record in _read_jsonl(telemetry_path)
        if record["phase"] == "VERTICAL_LIFT"
    ]
    if not records:
        raise RuntimeError(f"no VERTICAL_LIFT records: {telemetry_path}")
    clearance = [
        float(record["observation"]["clearance_m"])
        for record in records
    ]
    ee = [
        float(record["observation"]["ee_vertical_displacement_m"])
        for record in records
    ]
    force_by_side: dict[str, list[float]] = {
        "left": [],
        "right": [],
    }
    for record in records:
        for side, force_records in force_by_side.items():
            impulse = 0.0
            for contact in record.get("contacts", []):
                paths = (
                    str(contact.get("collider0_path", ""))
                    + str(contact.get("collider1_path", ""))
                )
                if (
                    f"{side}_finger" in paths
                    and "Bottle500" in paths
                ):
                    value = float(contact.get("impulse_ns", 0.0))
                    if math.isfinite(value) and value > 0.0:
                        impulse += value
            force_records.append(impulse * 60.0)
    return {
        "lift_frame_count": len(records),
        "clearance_start_m": clearance[0],
        "clearance_end_m": clearance[-1],
        "maximum_clearance_m": max(clearance),
        "ee_displacement_start_m": ee[0],
        "ee_displacement_end_m": ee[-1],
        "relative_vertical_slip_change_m": (
            (ee[-1] - ee[0]) - (clearance[-1] - clearance[0])
        ),
        "mean_estimated_normal_force_n": {
            side: sum(values) / len(values)
            for side, values in force_by_side.items()
        },
        "method": (
            "SUM_POSITIVE_FINGER_BOTTLE_NORMAL_IMPULSE_PER_FRAME_"
            "TIMES_60HZ;_RELATIVE_SLIP_FROM_EE_AND_BOTTLE_CLEARANCE_"
            "DELTA"
        ),
    }


def _image_record(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        size = list(image.size)
    return {
        "absolute_path": str(path.resolve()),
        "sha256": _sha256(path),
        "resolution": size,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ALOHA 20 cm five-position video review",
        "",
        f"- Visual evidence status: **{report['visual_evidence_status']}**",
        f"- Five-position acceptance: **{report['acceptance_status']}**",
        f"- Machine passes: **{report['machine_pass_count']}/5**",
        "- Task 8: **NOT_RUN**",
        "",
        "## Per-position result",
        "",
        "| Position | Machine | Reason | Visual evidence |",
        "|---|---:|---|---:|",
    ]
    lines.extend(
        (
            "| {position_id} | {machine_status} | {machine_reason} | "
            "{visual_review_status} |".format(**record)
        )
        for record in report["positions"]
    )
    diagnosis = report["root_cause_diagnosis"]
    lines.extend(
        [
            "",
            "## Root-cause boundary",
            "",
            (
                "- Position 2 retained bilateral solver contact but reached "
                f"only {diagnosis['failed_position_maximum_clearance_m']:.9f} m "
                "against the unchanged 0.200 m gate."
            ),
            (
                "- Its measured relative vertical slip change was "
                f"{diagnosis['failed_position_relative_slip_m']:.9f} m."
            ),
            (
                "- A diagnostic-only +2 mm lift reached the height gate but "
                f"failed hold after "
                f"{diagnosis['lift_margin_diagnostic_hold_duration_s']:.6f} s "
                "with drop "
                f"{diagnosis['lift_margin_diagnostic_hold_drop_m']:.9f} m."
            ),
            (
                "- Classification: "
                f"**{diagnosis['classification']}**. The extra lift is not "
                "promoted."
            ),
            "",
            "The raw and annotated videos were reviewed through complete "
            "frame contact-sheet montages plus annotated phase keyframes. "
            "Every view shows the full arm and the gripper/bottle inset; "
            "position 2 is correctly labeled as machine FAIL. Visual review "
            "validates evidence quality, not physical acceptance.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--lift-margin-diagnostic", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    results_path = args.results.resolve(strict=True)
    artifact_root = args.artifact_root.resolve(strict=True)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if len(results.get("positions", [])) != 5:
        raise RuntimeError("five-position result does not contain five runs")
    positions: list[dict[str, Any]] = []
    lift_comparison: dict[str, Any] = {}
    for position in results["positions"]:
        position_id = str(position["position_id"])
        primary_root = artifact_root / position_id / "primary"
        candidate_path = (
            primary_root
            / "video_attempt_001/video/candidate_manifest.json"
        ).resolve(strict=True)
        candidate = json.loads(
            candidate_path.read_text(encoding="utf-8")
        )
        videos = []
        for video in candidate["videos"]:
            path = Path(video["absolute_path"]).resolve(strict=True)
            if _sha256(path) != str(video["sha256"]):
                raise RuntimeError(f"video hash mismatch: {path}")
            videos.append(
                {
                    "kind": str(video["kind"]),
                    "absolute_path": str(path),
                    "sha256": _sha256(path),
                    "frame_count": int(video["frame_count"]),
                    "fps": int(video["fps"]),
                }
            )
        raw_montage = (
            artifact_root
            / "visual_review_montages"
            / f"{position_id}_full_video_contact_sheet_montage.png"
        ).resolve(strict=True)
        annotated_montage = (
            artifact_root
            / "visual_review_montages"
            / f"{position_id}_annotated_keyframes_montage.png"
        ).resolve(strict=True)
        telemetry_path = (
            primary_root / "aloha1_grasp_20cm_telemetry.jsonl"
        ).resolve(strict=True)
        lift_comparison[position_id] = _lift_metrics(telemetry_path)
        positions.append(
            {
                "position_id": position_id,
                "machine_status": str(
                    position["primary"]["machine_status"]
                ),
                "machine_reason": str(
                    position["primary"]["machine_reason"]
                ),
                "deterministic_repeat": bool(
                    position["primary"]["deterministic_signature"]
                    == position["collider_repeat"][
                        "deterministic_signature"
                    ]
                ),
                "visual_review_status": "PASS",
                "visual_review_scope": (
                    "EVIDENCE_QUALITY_ONLY_NOT_PHYSICAL_ACCEPTANCE"
                ),
                "visual_checks": {
                    "full_arm_visible": True,
                    "gripper_and_bottle_visible": True,
                    "open_contact_lift_hold_or_failure_distinguishable": True,
                    "overview_and_closeup_synchronized": True,
                    "machine_status_annotation_matches_report": True,
                    "critical_geometry_not_occluded": True,
                },
                "retake_reasons": [],
                "videos": videos,
                "raw_full_video_review_montage": _image_record(raw_montage),
                "annotated_keyframe_review_montage": _image_record(
                    annotated_montage
                ),
            }
        )
    diagnostic_path = args.lift_margin_diagnostic.resolve(strict=True)
    diagnostic = json.loads(
        diagnostic_path.read_text(encoding="utf-8")
    )
    failed = lift_comparison["position_02"]
    diagnosis = {
        "classification": (
            "POSITION_DEPENDENT_CONTINUOUS_SLIP_OR_ROTATIONAL_"
            "INSTABILITY_NOT_RESOLVED"
        ),
        "failed_position": "position_02",
        "failed_position_maximum_clearance_m": float(
            failed["maximum_clearance_m"]
        ),
        "failed_position_relative_slip_m": float(
            failed["relative_vertical_slip_change_m"]
        ),
        "failed_position_mean_estimated_normal_force_n": failed[
            "mean_estimated_normal_force_n"
        ],
        "height_gate_m": 0.200,
        "height_deficit_m": (
            0.200 - float(failed["maximum_clearance_m"])
        ),
        "lift_margin_diagnostic": {
            "absolute_path": str(diagnostic_path),
            "sha256": _sha256(diagnostic_path),
            "additional_lift_margin_m": float(
                diagnostic["runtime"]["trajectory"][
                    "additional_lift_margin_m"
                ]
            ),
            "status": str(diagnostic["status"]),
            "reason": str(diagnostic["reason"]),
        },
        "lift_margin_diagnostic_hold_duration_s": float(
            diagnostic["metrics"]["hold_duration_s"]
        ),
        "lift_margin_diagnostic_hold_drop_m": float(
            diagnostic["metrics"]["hold_drop_m"]
        ),
        "extra_lift_promoted": False,
        "unchanged": [
            "collider",
            "friction",
            "drive",
            "mimic",
            "bottle_mass",
            "bottle_diameter",
            "physics_timestep",
            "acceptance_gates",
        ],
    }
    machine_pass_count = sum(
        record["machine_status"] == "PASS" for record in positions
    )
    report = {
        "schema_version": 1,
        "status": "PASS",
        "visual_evidence_status": "PASS",
        "acceptance_status": (
            "PASS" if machine_pass_count == 5 else "FAIL"
        ),
        "machine_pass_count": machine_pass_count,
        "review_method": (
            "VISUAL_MODEL_COMPLETE_RAW_CONTACT_SHEET_MONTAGE_AND_"
            "ANNOTATED_PHASE_KEYFRAME_REVIEW"
        ),
        "source_results": {
            "absolute_path": str(results_path),
            "sha256_before_review": _sha256(results_path),
        },
        "positions": positions,
        "lift_comparison": lift_comparison,
        "root_cause_diagnosis": diagnosis,
        "user_confirmation": "NOT_RUN",
        "task8": "NOT_RUN",
    }
    for position in results["positions"]:
        position["visual_model_review"] = "PASS"
    results["visual_model_review"] = "PASS"
    results["visual_review_report_absolute_path"] = str(
        args.output_json.resolve()
    )
    results["user_confirmation"] = "NOT_RUN"
    _atomic_json(results_path, results)
    report["source_results"]["sha256_after_review"] = _sha256(results_path)
    _atomic_json(args.output_json.resolve(), report)
    args.output_md.resolve().write_text(
        _markdown(report),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
