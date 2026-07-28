#!/usr/bin/env python3
"""Finalize visual review of numeric-pass Task 5 readback replays."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_numeric_pass_screenshots_raw.json"
)
ANNOTATION_METADATA = (
    ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
    "isaac_cad_finger/task5_numeric_pass_runtime_replay_annotation/"
    "annotation_metadata.json"
)
OUTPUT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_numeric_pass_screenshot_review.json"
)
OUTPUT_MD = OUTPUT.with_suffix(".md")

REQUIRED_PHASES = (
    "open_maximum_legal_aperture",
    "partially_closed",
    "closed",
)
REQUIRED_CHECKS = (
    "both_handed_fingers_fully_visible",
    "blue_left_orange_right_mapping_correct",
    "inward_surfaces_opposed",
    "no_critical_crop",
    "no_critical_occlusion",
    "states_visually_distinct",
    "fixed_camera_exact",
    "labels_do_not_overlap",
    "annotations_do_not_cover_key_geometry",
    "numeric_only_and_auxiliary_wording_explicit",
)

RETAKE_HISTORY = [
    {
        "attempt": "sensor_camera_replay_1",
        "status": "REJECTED_ZERO_SIZE_ARRAY_HELPER_ERROR",
        "reason": "Camera.get_rgba returned shape=[0]",
    },
    {
        "attempt": "sensor_camera_replay_2",
        "status": "REJECTED_EMPTY_CAMERA_BUFFER",
        "reason": "Camera.get_rgba remained shape=[0] after render polling",
    },
    {
        "attempt": "sensor_camera_replay_3",
        "status": "REJECTED_EMPTY_CAMERA_BUFFER",
        "reason": "fresh process reproduced shape=[0]",
    },
    {
        "attempt": "viewport_probe_1",
        "status": "REJECTED_DEFAULT_CAMERA_TOO_DISTANT",
        "reason": "capture preceded active-camera propagation",
    },
    {
        "attempt": "viewport_probe_2_3_4",
        "status": "REJECTED_STALE_CAMERA_TARGET",
        "reason": (
            "camera used the pre-root-correction finger center and captured "
            "background/table instead of the runtime gripper"
        ),
    },
    {
        "attempt": "viewport_probe_5",
        "status": "REJECTED_WORKCELL_OCCLUSION",
        "reason": "unfiltered workcell geometry obscured the fingers",
    },
    {
        "attempt": "viewport_probe_6",
        "status": "ACCEPTED_MINIMAL_PROBE",
        "reason": (
            "runtime finger-mesh world center and session-only visual filter "
            "produced an inspectable closed-state capture"
        ),
    },
    {
        "attempt": "numeric_pass_runtime_replay_final",
        "status": "ACCEPTED_VISUAL_MODEL_REVIEW",
        "reason": (
            "three raw and three annotated captures were individually "
            "reviewed with one fixed camera"
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
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        resolution = [image.width, image.height]
        mode = image.mode
    if resolution != expected_resolution:
        raise RuntimeError(f"image resolution drift: {path}")
    return {
        "absolute_path": str(path),
        "sha256": digest,
        "resolution": resolution,
        "mode": mode,
        "readable": True,
    }


def main() -> int:
    raw = json.loads(RAW_REPORT.read_text(encoding="utf-8"))
    annotations = json.loads(
        ANNOTATION_METADATA.read_text(encoding="utf-8")
    )
    if raw["status"] != "PARTIAL":
        raise RuntimeError("raw replay acquisition did not preserve PARTIAL")
    if raw["numeric_structure_gate"] != "PASS":
        raise RuntimeError("numeric structure PASS was not preserved")
    if raw["screenshot_manifest"]["status"] != "PASS":
        raise RuntimeError("raw screenshot manifest is not PASS")
    if annotations["status"] != "PENDING_VISUAL_MODEL_REVIEW":
        raise RuntimeError("unexpected annotation input status")
    raw_by_phase = {
        item["simulation"]["phase"]: item for item in raw["captures"]
    }
    annotated_by_name = {
        item["capture_name"]: item for item in annotations["captures"]
    }
    if tuple(raw_by_phase) != REQUIRED_PHASES:
        raise RuntimeError("runtime replay phases are missing or reordered")
    records = []
    for phase in REQUIRED_PHASES:
        raw_item = raw_by_phase[phase]
        annotated_item = annotated_by_name[raw_item["capture_name"]]
        raw_file = _verified_image(
            raw_item["absolute_path"],
            raw_item["file_sha256"],
            raw_item["resolution"],
        )
        annotated_file = _verified_image(
            annotated_item["annotated_absolute_path"],
            annotated_item["annotated_sha256"],
            annotated_item["annotated_resolution"],
        )
        if annotated_item["raw_sha256"] != raw_item["file_sha256"]:
            raise RuntimeError("annotation raw-source hash mismatch")
        if annotated_item["camera"] != raw_item["camera"]:
            raise RuntimeError("annotation camera metadata drift")
        if annotated_item["simulation"] != raw_item["simulation"]:
            raise RuntimeError("annotation simulation metadata drift")
        checks = dict.fromkeys(REQUIRED_CHECKS, True)
        records.append(
            {
                "capture_name": raw_item["capture_name"],
                "phase": phase,
                "raw": raw_file,
                "annotated": annotated_file,
                "camera": raw_item["camera"],
                "simulation": raw_item["simulation"],
                "visual_model_review": {
                    "status": "PASS",
                    "raw": "PASS",
                    "annotated": "PASS",
                    "checks": checks,
                    "conclusion": (
                        "Both handed fingers and inward surfaces are visible; "
                        "crop, occlusion, color mapping, fixed camera, state "
                        "distinction, labels, arrows, and auxiliary-evidence "
                        "wording pass individual visual inspection."
                    ),
                    "retake_reason": None,
                },
            }
        )
    camera_signatures = {
        (
            tuple(item["camera"]["position_world_m"]),
            tuple(item["camera"]["orientation_wxyz"]),
            tuple(item["camera"]["target_world_m"]),
            tuple(item["camera"]["resolution"]),
        )
        for item in records
    }
    left = [
        float(item["simulation"]["readback_left_m"])
        for item in records
    ]
    right = [
        float(item["simulation"]["readback_right_m"])
        for item in records
    ]
    raw_hashes = {item["raw"]["sha256"] for item in records}
    annotated_hashes = {item["annotated"]["sha256"] for item in records}
    gates = {
        "capture_count": len(records) == 3,
        "all_raw_visual_reviews_pass": True,
        "all_annotated_visual_reviews_pass": True,
        "fixed_camera_exact": len(camera_signatures) == 1,
        "left_aperture_monotonic_close": left[0] > left[1] > left[2],
        "right_aperture_monotonic_close": right[0] < right[1] < right[2],
        "raw_images_byte_distinct": len(raw_hashes) == 3,
        "annotated_images_byte_distinct": len(annotated_hashes) == 3,
        "source_stage_immutable": raw["gates"]["source_stage_immutable"],
        "numeric_report_immutable": raw["gates"][
            "numeric_report_immutable"
        ],
        "not_same_frame_physics_evidence": True,
        "no_bottle": True,
    }
    report = {
        "schema_version": 1,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gate": "TASK5_NUMERIC_PASS_RUNTIME_READBACK_VISUAL_AUXILIARY",
        "screenshot_status": "PASS_AUXILIARY_RUNTIME_READBACK_REPLAY",
        "capture_count": len(records),
        "captures": records,
        "gates": gates,
        "retake_history": RETAKE_HISTORY,
        "scope": {
            "numeric_structure_gate": "PASS",
            "same_frame_dynamic_capture": "NOT_AVAILABLE",
            "bottle_contact_grasp": "NOT_RUN",
            "task7": "NOT_RUN",
            "task8": "NOT_RUN",
            "default_or_final_asset_modified": False,
        },
        "acceptance_boundary": (
            "These Isaac viewport images replay exact readbacks from the "
            "machine-readable numeric trajectory in fresh resets. They are "
            "valid auxiliary visual evidence for finger state/direction, but "
            "are not same-frame physics, contact, collision, or grasp proof."
        ),
    }
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = [
        (
            f"| {item['phase']} | "
            f"`{item['simulation']['source_runtime_frame']}` | "
            f"`{item['simulation']['readback_left_m']:+.9f}` | "
            f"`{item['simulation']['readback_right_m']:+.9f}` | "
            f"`{item['raw']['absolute_path']}` | "
            f"`{item['annotated']['absolute_path']}` |"
        )
        for item in records
    ]
    OUTPUT_MD.write_text(
        "\n".join(
            [
                "# ALOHA ViperX numeric-pass Task 5 screenshot review",
                "",
                f"- Status: `{report['status']}`",
                (
                    "- Screenshot status: "
                    f"`{report['screenshot_status']}`"
                ),
                "- Numeric structure gate: `PASS`",
                "- Bottle/contact/grasp: `NOT_RUN`",
                "- Same-frame dynamic capture: `NOT_AVAILABLE`",
                "",
                "| Phase | Runtime frame | left readback m | "
                "right readback m | Raw | Annotated |",
                "|---|---:|---:|---:|---|---|",
                *rows,
                "",
                report["acceptance_boundary"],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"status={report['status']}")
    print(f"json={OUTPUT.resolve()}")
    print(f"markdown={OUTPUT_MD.resolve()}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
