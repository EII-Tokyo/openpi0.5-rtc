#!/usr/bin/env python3
"""Finalize the individually inspected supplier-CAD bottle screenshots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_bottle.json"
)
ANNOTATION_METADATA = (
    ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
    "isaac_cad_finger/task5_bottle_acceptance_v3_annotation_attempt2/"
    "annotation_metadata.json"
)
OUTPUT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_bottle_screenshot_review.json"
)
OUTPUT_MD = OUTPUT.with_suffix(".md")

REQUIRED_PHASES = (
    "open",
    "bilateral_contact",
    "release",
    "hold_end",
)
VISUAL_CHECKS = (
    "both_fingers_fully_visible",
    "blue_left_finger_cad_positive_x_mapping_correct",
    "orange_right_finger_cad_negative_x_mapping_correct",
    "inward_surfaces_visible_and_opposed",
    "bottle_visible",
    "no_critical_crop",
    "no_shell_occlusion_of_contact_geometry",
    "fixed_camera_pose",
    "labels_do_not_overlap",
    "annotations_do_not_cover_key_geometry",
    "pass_wording_limited_to_phase_gate",
)
RETAKE_HISTORY = [
    {
        "attempt": "attempt3",
        "status": "REJECTED_OCCLUSION_AND_UNCOUNTED_STEP_RISK",
        "reason": (
            "The gripper shell obscured the supplier-CAD fingers and "
            "viewport update calls could advance physics during capture."
        ),
    },
    {
        "attempt": "attempt4",
        "status": "PASS_SMOKE_VIEW_AND_CAPTURE_STATE",
        "reason": (
            "Session-only shell hiding exposed both fingers; capture paused "
            "the timeline and asserted zero physics steps and unchanged state."
        ),
    },
    {
        "attempt": "acceptance_v1",
        "status": "SUPERSEDED_BY_CONTACT_SEMANTICS_AND_PROJECTION",
        "reason": (
            "The first 20-run batch predated the explicit separation<=0 "
            "physical-contact gate and camera projection of contact normals."
        ),
    },
    {
        "attempt": "acceptance_v2",
        "status": "SUPERSEDED_BY_FULL_INTERVAL_DROP_GATE",
        "reason": (
            "The second batch added contact projection but did not yet use "
            "the maximum drop over the complete hold interval."
        ),
    },
    {
        "attempt": "v3_annotation_first",
        "status": "REJECTED_PHASE_CONTEXT_MISLABEL",
        "reason": (
            "Open, bilateral-contact and release panels incorrectly repeated "
            "the hold-end pose-derived velocity context."
        ),
    },
    {
        "attempt": "v3_annotation_attempt2",
        "status": "PASS",
        "reason": (
            "All four raw and all four corrected annotated captures were "
            "individually inspected with the vision model."
        ),
    },
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_image(
    path_text: str,
    expected_hash: str,
    expected_resolution: list[int],
) -> dict[str, Any]:
    path = Path(path_text).resolve(strict=True)
    digest = _sha256(path)
    if digest != expected_hash:
        raise RuntimeError(f"image hash drift: {path}")
    with Image.open(path) as opened:
        opened.verify()
    with Image.open(path) as opened:
        resolution = [opened.width, opened.height]
        mode = opened.mode
    if resolution != expected_resolution:
        raise RuntimeError(f"image resolution drift: {path}")
    return {
        "absolute_path": str(path),
        "sha256": digest,
        "resolution": resolution,
        "mode": mode,
        "readable": True,
        "visual_model_review": "PASS",
    }


def _camera_signature(camera: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(camera["position_world_m"]),
        tuple(camera["orientation_wxyz"]),
        tuple(camera["target_world_m"]),
        tuple(camera["resolution"]),
        camera["view"],
    )


def _build_markdown(report: dict[str, Any]) -> str:
    rows = [
        (
            f"| {record['phase']} | "
            f"`{record['simulation']['frame']}` | "
            f"`{record['simulation']['time_s']:.6f}` | "
            f"`{record['contact_projection_count']}` | "
            f"`{record['raw']['absolute_path']}` | "
            f"`{record['annotated']['absolute_path']}` | PASS |"
        )
        for record in report["records"]
    ]
    return "\n".join(
        [
            "# ALOHA ViperX supplier-CAD bottle screenshot review",
            "",
            f"- Status: `{report['status']}`",
            "- Scope: `follower_left`, supplier assembly embedded v2 fingers",
            (
                "- Runtime hold: "
                f"`{report['runtime_repeat_summary']['pass_count']}/"
                f"{report['runtime_repeat_summary']['trial_count']} PASS`, "
                f"deterministic=`{str(report['runtime_repeat_summary']['deterministic']).lower()}`"
            ),
            (
                "- Maximum full-interval drop: "
                f"`{report['runtime_repeat_summary']['maximum_drop_m']:.12f} m`"
            ),
            "- Screenshot role: auxiliary visual evidence only",
            "- Task 8: `NOT_RUN`",
            "",
            "| Phase | Frame | Time s | Projected contacts | Raw | Annotated | Vision |",
            "|---|---:|---:|---:|---|---|---|",
            *rows,
            "",
            "The open image contains CAD-derived inward-surface samples, not "
            "physical contacts. Bilateral-contact, release, and hold-end "
            "images each contain two runtime-projected physical contact "
            "points and normals. Release and hold-end raw geometry is nearly "
            "unchanged because the bottle remained held; their distinct "
            "frame/time and machine trajectory are recorded in the annotated "
            "images and runtime report.",
            "",
            "The screenshot PASS does not replace the contact, pose, velocity, "
            "drop, penetration, or deterministic runtime gates.",
            "",
        ]
    )


def main() -> int:
    runtime_path = RUNTIME_REPORT.resolve(strict=True)
    annotation_path = ANNOTATION_METADATA.resolve(strict=True)
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))

    if runtime["status"] != "PASS":
        raise RuntimeError("runtime acceptance report is not PASS")
    summary = runtime["summary"]
    if summary["pass_count"] != 20 or not summary["deterministic"]:
        raise RuntimeError("runtime repeat gate is not 20/20 deterministic")
    if runtime["screenshots"]["status"] != "PASS":
        raise RuntimeError("raw screenshot acquisition is not PASS")

    raw_by_phase = {
        item["capture_name"]: item
        for item in runtime["screenshots"]["captures"]
    }
    annotated_by_phase = {
        item["capture_name"]: item for item in annotations["records"]
    }
    if tuple(raw_by_phase) != REQUIRED_PHASES:
        raise RuntimeError("raw phase order or coverage drift")
    if tuple(annotated_by_phase) != REQUIRED_PHASES:
        raise RuntimeError("annotated phase order or coverage drift")

    records: list[dict[str, Any]] = []
    for phase in REQUIRED_PHASES:
        raw_record = raw_by_phase[phase]
        annotation = annotated_by_phase[phase]
        raw_image = _verified_image(
            raw_record["absolute_path"],
            raw_record["file_sha256"],
            raw_record["resolution"],
        )
        annotated_image = _verified_image(
            annotation["annotated_absolute_path"],
            annotation["annotated_sha256"],
            annotation["annotated_resolution"],
        )
        if annotation["raw_sha256"] != raw_record["file_sha256"]:
            raise RuntimeError(f"annotation source drift: {phase}")
        if annotation["camera"] != raw_record["camera"]:
            raise RuntimeError(f"annotation camera metadata drift: {phase}")
        if annotation["simulation"] != raw_record["simulation"]:
            raise RuntimeError(f"annotation simulation metadata drift: {phase}")
        projection_count = len(
            raw_record["camera"].get("contact_projection", {})
        )
        expected_projection_count = 0 if phase == "open" else 2
        if projection_count != expected_projection_count:
            raise RuntimeError(f"contact projection count drift: {phase}")

        records.append(
            {
                "phase": phase,
                "raw": raw_image,
                "annotated": annotated_image,
                "camera": raw_record["camera"],
                "simulation": raw_record["simulation"],
                "contact_projection_count": projection_count,
                "visual_model_review": {
                    "status": "PASS",
                    "checks": dict.fromkeys(VISUAL_CHECKS, True),
                    "conclusion": (
                        "The supplier-CAD handed fingers, inward surfaces, "
                        "bottle and phase evidence are visible. Mapping, "
                        "crop, occlusion, labels, contact projection and "
                        "phase-limited PASS wording passed individual visual "
                        "inspection."
                    ),
                    "retake_reason": None,
                },
            }
        )

    camera_signatures = {
        _camera_signature(record["camera"]) for record in records
    }
    raw_hashes = {record["raw"]["sha256"] for record in records}
    if len(raw_hashes) < 3:
        raise RuntimeError("physical phases are not visually distinguishable")

    report = {
        "schema_version": 1,
        "status": "PASS",
        "gate": "TASK5_SUPPLIER_CAD_BOTTLE_SCREENSHOT_VISUAL_REVIEW",
        "source_runtime_report": {
            "absolute_path": str(runtime_path),
            "sha256": _sha256(runtime_path),
        },
        "source_annotation_metadata": {
            "absolute_path": str(annotation_path),
            "sha256": _sha256(annotation_path),
        },
        "raw_capture_count": len(records),
        "annotated_capture_count": len(records),
        "all_raw_vision_reviews_pass": True,
        "all_annotated_vision_reviews_pass": True,
        "fixed_camera_across_phases": len(camera_signatures) == 1,
        "runtime_report_status": runtime["status"],
        "runtime_repeat_summary": summary,
        "screenshot_is_auxiliary": True,
        "task8": "NOT_RUN",
        "records": records,
        "retake_history": RETAKE_HISTORY,
        "visual_review_method": (
            "Each of four raw and four annotated images was opened and "
            "inspected individually with the vision model on 2026-07-29."
        ),
        "acceptance_boundary": (
            "Screenshot review validates visibility, installed handedness, "
            "phase distinction and annotation integrity only. Runtime "
            "contact, pose, velocity, maximum drop, penetration and repeat "
            "data remain authoritative for the static-hold result."
        ),
    }
    if not report["fixed_camera_across_phases"]:
        report["status"] = "FAIL"

    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    OUTPUT_MD.write_text(_build_markdown(report), encoding="utf-8")
    print(f"status={report['status']}")
    print(f"json={OUTPUT.resolve()}")
    print(f"markdown={OUTPUT_MD.resolve()}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
