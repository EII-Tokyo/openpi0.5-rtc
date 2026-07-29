#!/usr/bin/env python3
"""Freeze the completed vision-model review of Hydra protoPath screenshots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = (
    ROOT
    / ".codex/artifacts/20260729-aloha1-signal-correspondence"
    / "hydra_protopath_diagnosis"
)
REPORT_ROOT = ROOT / "reports/aloha1_mapping"
MAIN_REPORT = REPORT_ROOT / "aloha1_hydra_protopath_diagnosis.json"
JSON_REPORT = REPORT_ROOT / "aloha1_hydra_protopath_screenshot_review.json"
MD_REPORT = REPORT_ROOT / "aloha1_hydra_protopath_screenshot_review.md"

PHASES = (
    "home_reference",
    "small_up_start",
    "small_up_max",
    "small_down_return",
    "waist_positive",
    "waist_negative",
)
VARIANTS = ("A", "B", "C1", "C2", "C3_RESUME1", "C4", "B_REPEAT", "RESTORE")
D_RETAKE_HISTORY = (
    ("D", "REJECTED_VIEW_TOO_DISTANT", "both followers were too small to inspect link-mesh completeness"),
    ("D_RETAKE1", "REJECTED_CAMERA_NOT_ACTIVE", "camera change did not affect the active viewport"),
    ("D_RETAKE2", "REJECTED_CAMERA_NOT_ACTIVE", "focal-length change did not affect the active viewport"),
    ("D_RETAKE3", "REJECTED_CAPTURE_NOT_CREATED", "local Sdf import error prevented capture"),
    ("D_RETAKE4", "REJECTED_CAMERA_NOT_ACTIVE", "viewport remained on the default distant camera"),
    ("D_RETAKE5", "REJECTED_WRONG_TARGET", "active camera exposed only a close workcell-frame segment"),
    ("D_RETAKE6", "REJECTED_CROPPED", "follower base and distal links were cropped"),
    ("D_RETAKE7", "REJECTED_OCCLUDED_AND_CROPPED", "rack occluded the gripper and the base was cropped"),
    (
        "D_RETAKE8",
        "PASS",
        "session-only environment visibility isolation exposes both complete materialized followers",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_record(path: Path, *, variant: str, phase: str, note: str) -> dict[str, object]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        dimensions = [int(image.width), int(image.height)]
    return {
        "variant": variant,
        "phase": phase,
        "raw_path": str(path.resolve()),
        "sha256": _sha256(path),
        "dimensions_px": dimensions,
        "review_method": "VISION_MODEL_INDIVIDUAL_IMAGE_REVIEW",
        "status": "PASS",
        "detection_target": (
            "complete follower visual geometry, unclipped gripper, and visibly distinct Task 7A pose"
            if variant != "D_RETAKE8"
            else "complete left/right materialized visual meshes without workcell occlusion"
        ),
        "vision_review": note,
        "acceptance_boundary": (
            "screenshot evidence only; runtime protoPath counts and mesh inventories remain authoritative"
        ),
    }


def main() -> int:
    records: list[dict[str, object]] = []
    for variant in VARIANTS:
        root = ARTIFACT_ROOT / variant / "exact_capture/screenshots_raw/follower_left"
        for phase in PHASES:
            path = root / f"follower_left_{phase}_raw.png"
            records.append(
                _image_record(
                    path,
                    variant=variant,
                    phase=phase,
                    note=(
                        "follower, finger pair, and workcell context are visible without cropping; "
                        "home/up/down/waist phases have visually distinguishable poses"
                    ),
                )
            )

    d_path = ARTIFACT_ROOT / "D_RETAKE8/native_raw.png"
    records.append(
        _image_record(
            d_path,
            variant="D_RETAKE8",
            phase="materialized_visuals",
            note=(
                "blue and orange materialized followers are both fully in frame; the diagnostic "
                "environment was hidden session-only to remove rack occlusion"
            ),
        )
    )

    retakes = []
    for variant, status, reason in D_RETAKE_HISTORY:
        path = ARTIFACT_ROOT / variant / "native_raw.png"
        retakes.append(
            {
                "variant": variant,
                "status": status,
                "reason": reason,
                "raw_path": str(path.resolve()) if path.exists() else None,
                "sha256": _sha256(path) if path.exists() else None,
            }
        )

    report = {
        "schema_version": 1,
        "status": "PASS",
        "review_method": "VISION_MODEL_INDIVIDUAL_IMAGE_REVIEW",
        "accepted_capture_count": len(records),
        "expected_accepted_capture_count": 49,
        "all_accepted_images_individually_viewed": True,
        "records": records,
        "retake_history": retakes,
        "scope": "Hydra protoPath controlled diagnosis screenshot evidence",
        "not_claimed": [
            "visual PASS does not override runtime protoPath errors",
            "variant D is not a final asset fix",
            "no physics or collision PASS is inferred from screenshots",
        ],
    }
    if len(records) != 49:
        raise RuntimeError(f"expected 49 accepted screenshots, got {len(records)}")
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# ALOHA 1 Hydra protoPath screenshot review",
        "",
        f"- Status: **{report['status']}**",
        f"- Review method: `{report['review_method']}`",
        f"- Accepted screenshots reviewed individually: `{len(records)}/49`",
        "- Boundary: screenshot review is auxiliary; runtime error counts and mesh inventories are authoritative.",
        "",
        "## Accepted evidence",
        "",
        "| Variant | Accepted images | Vision result |",
        "|---|---:|---|",
    ]
    for variant in (*VARIANTS, "D_RETAKE8"):
        count = sum(record["variant"] == variant for record in records)
        lines.append(f"| {variant} | {count} | PASS |")
    lines.extend(
        [
            "",
            "## Variant D retake history",
            "",
            "| Attempt | Status | Reason |",
            "|---|---|---|",
        ]
    )
    lines.extend(
        f"| {item['variant']} | {item['status']} | {item['reason']} |"
        for item in retakes
    )
    lines.extend(
        [
            "",
            f"Machine report: `{JSON_REPORT.resolve()}`",
            f"Accepted D close view: `{d_path.resolve()}`",
            "",
        ]
    )
    MD_REPORT.write_text("\n".join(lines))

    main_report = json.loads(MAIN_REPORT.read_text())
    main_report["visual_self_review"] = {
        "status": "PASS",
        "review_method": report["review_method"],
        "accepted_capture_count": len(records),
        "report_json": str(JSON_REPORT.resolve()),
        "report_md": str(MD_REPORT.resolve()),
        "variant_d_accepted_retake": str(d_path.resolve()),
    }
    MAIN_REPORT.write_text(json.dumps(main_report, indent=2, sort_keys=True) + "\n")
    main_md = REPORT_ROOT / "aloha1_hydra_protopath_diagnosis.md"
    marker = "## Vision-model screenshot review"
    main_text = main_md.read_text()
    if marker not in main_text:
        main_text += "\n".join(
            [
                "",
                marker,
                "",
                "- Status: **PASS**",
                "- Method: every accepted raw image was opened and reviewed individually by the vision model.",
                f"- Accepted evidence: `{len(records)}/49` images.",
                f"- Review report: `{JSON_REPORT.resolve()}`",
                f"- Variant D accepted retake: `{d_path.resolve()}`",
                "- Screenshot PASS is auxiliary and does not replace runtime protoPath/error/mesh evidence.",
                "",
            ]
        )
        main_md.write_text(main_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
