#!/usr/bin/env python3
"""Finalize raw and annotated Task 7 pose screenshot evidence."""

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
    "aloha_viper_cad_finger_task7_pose_screenshots_raw.json"
)
ANNOTATION_METADATA = (
    ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
    "task7_robot_scope/pose_evidence_attempt5/annotation_metadata_v2.json"
)
ANNOTATED_DECISIONS = (
    ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
    "task7_robot_scope/pose_evidence_attempt5/"
    "annotated_visual_review_decisions_v2.json"
)
NUMERIC_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_drive_probe_arm_max_force_over_combined.json"
)
BOTTLE_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_bottle.json"
)
BOTTLE_SCREENSHOTS = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_bottle_screenshot_review.json"
)
TASK7_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task7_validation.json"
)
OUTPUT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task7_pose_screenshot_review.json"
)
OUTPUT_MD = OUTPUT.with_suffix(".md")
RIGHT_ARM_BLOCKER = (
    "HARD_BLOCKER_APPROVED_STAGE_MISSING_FOLLOWER_RIGHT"
)

RETAKE_HISTORY = [
    {
        "attempt": "pose_evidence_rejected_attempt1",
        "status": "REJECTED_FRAMING_DEBUG_GEOMETRY_AND_OCCLUSION",
        "reason": (
            "full-arm view framed only the wrist/gripper; camera-focus and "
            "other non-target geometry remained; closeup was occluded"
        ),
    },
    {
        "attempt": "pose_evidence_attempt2",
        "status": "REJECTED_NON_TARGET_RED_GEOMETRY_AND_TABLE_EDGE",
        "reason": (
            "instance-proxy bounds fixed the full-arm framing, but a red "
            "debug object and table edge remained"
        ),
    },
    {
        "attempt": "pose_evidence_attempt3",
        "status": "REJECTED_BLUE_SITE_DEBUG_AXES",
        "reason": "site debug axes remained above the gripper",
    },
    {
        "attempt": "pose_evidence_attempt4",
        "status": "REJECTED_SITE_ROOT_PATH_BOUNDARY",
        "reason": (
            "hiding gripper-prop did not remove the axes; audit traced the "
            "issue to the /sites container root"
        ),
    },
    {
        "attempt": "pose_evidence_attempt5_raw",
        "status": "PASS_INDIVIDUAL_VISUAL_MODEL_REVIEW",
        "reason": (
            "six raw images show the complete follower_left or unobstructed "
            "finger closeup with distinct certified states"
        ),
    },
    {
        "attempt": "pose_evidence_attempt5_annotated_v1",
        "status": "REJECTED_ANNOTATION_OCCLUDES_INWARD_SURFACE",
        "reason": (
            "closed closeup arrows and centerline crossed the inner contact "
            "region"
        ),
    },
    {
        "attempt": "pose_evidence_attempt5_annotated_v2",
        "status": "PASS_INDIVIDUAL_VISUAL_MODEL_REVIEW",
        "reason": (
            "arrows moved above geometry and centerline breaks around the "
            "finger boxes"
        ),
    },
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def _image_record(
    path: Path,
    *,
    expected_hash: str,
    expected_resolution: list[int],
    visual_review: str,
) -> dict[str, Any]:
    path = path.resolve(strict=True)
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
        "visual_model_review": visual_review,
    }


def main() -> int:
    raw = _load(RAW_REPORT)
    annotations = _load(ANNOTATION_METADATA)
    decisions = _load(ANNOTATED_DECISIONS)
    numeric = _load(NUMERIC_REPORT)
    bottle = _load(BOTTLE_REPORT)
    bottle_screenshots = _load(BOTTLE_SCREENSHOTS)
    task7 = _load(TASK7_REPORT)
    if raw["capture_status"] != "PASS":
        raise RuntimeError("raw screenshot acquisition is not PASS")
    if annotations["status"] != "PENDING_ANNOTATED_VISUAL_MODEL_REVIEW":
        raise RuntimeError("unexpected annotation metadata status")
    if decisions["status"] != "PASS":
        raise RuntimeError("annotated visual review is not PASS")
    if numeric["status"] != "PASS":
        raise RuntimeError("numeric Task 5 structure input is not PASS")
    if bottle["status"] != "PASS":
        raise RuntimeError("Task 5 bottle input is not PASS")
    if bottle_screenshots["status"] != "PASS":
        raise RuntimeError("Task 5 bottle screenshots are not PASS")
    if task7["status"] != "PARTIAL":
        raise RuntimeError("Task 7 status changed from expected PARTIAL")

    raw_by_name = {
        item["capture_name"]: item for item in raw["captures"]
    }
    annotated_by_name = {
        item["capture_name"]: item for item in annotations["records"]
    }
    expected_names = set(raw_by_name)
    if set(annotated_by_name) != expected_names:
        raise RuntimeError("raw/annotated capture set mismatch")
    if set(decisions["records"]) != expected_names:
        raise RuntimeError("annotated decision set mismatch")

    records = []
    for name, raw_item in raw_by_name.items():
        annotated_item = annotated_by_name[name]
        decision = decisions["records"][name]
        if decision["status"] != "PASS":
            raise RuntimeError(f"annotated visual review failed: {name}")
        if annotated_item["raw_sha256"] != raw_item["file_sha256"]:
            raise RuntimeError(f"annotation raw hash mismatch: {name}")
        if annotated_item["camera"] != raw_item["camera"]:
            raise RuntimeError(f"annotation camera drift: {name}")
        if annotated_item["simulation"] != raw_item["simulation"]:
            raise RuntimeError(f"annotation simulation drift: {name}")
        raw_review = annotated_item["raw_visual_review"]
        if raw_review["status"] != "PASS":
            raise RuntimeError(f"raw visual review failed: {name}")
        records.append(
            {
                "capture_name": name,
                "view": raw_item["camera"]["view"],
                "phase": raw_item["simulation"]["phase"],
                "raw": _image_record(
                    Path(raw_item["absolute_path"]),
                    expected_hash=raw_item["file_sha256"],
                    expected_resolution=raw_item["resolution"],
                    visual_review="PASS",
                ),
                "annotated": _image_record(
                    Path(annotated_item["annotated_absolute_path"]),
                    expected_hash=annotated_item["annotated_sha256"],
                    expected_resolution=annotated_item[
                        "annotated_resolution"
                    ],
                    visual_review="PASS",
                ),
                "camera": raw_item["camera"],
                "simulation": raw_item["simulation"],
                "raw_visual_review": raw_review,
                "annotated_visual_review": decision,
            }
        )

    camera_signatures_by_view: dict[str, set[tuple[Any, ...]]] = {}
    hashes_by_view: dict[str, set[str]] = {}
    for item in records:
        camera = item["camera"]
        camera_signatures_by_view.setdefault(item["view"], set()).add(
            (
                tuple(camera["position_world_m"]),
                tuple(camera["orientation_wxyz"]),
                tuple(camera["target_world_m"]),
                tuple(camera["resolution"]),
            )
        )
        hashes_by_view.setdefault(item["view"], set()).add(
            item["raw"]["sha256"]
        )
    fixed_camera = all(
        len(signatures) == 1
        for signatures in camera_signatures_by_view.values()
    )
    states_distinct = all(
        len(hashes) == 3 for hashes in hashes_by_view.values()
    )
    left_status = (
        "PASS"
        if (
            len(records) == 6
            and fixed_camera
            and states_distinct
            and all(
                item["raw"]["visual_model_review"] == "PASS"
                and item["annotated"]["visual_model_review"] == "PASS"
                for item in records
            )
        )
        else "FAIL"
    )
    report = {
        "schema_version": 1,
        "status": "PARTIAL" if left_status == "PASS" else "FAIL",
        "left_arm": {
            "status": left_status,
            "records": records,
            "fixed_camera_within_each_view": fixed_camera,
            "states_visually_distinct": states_distinct,
            "certified_action": "symmetric_close",
            "certified_phases": [
                "open_maximum_legal_aperture",
                "partially_closed",
                "closed",
            ],
        },
        "right_arm": {
            "status": "NOT_RUN",
            "blocker": RIGHT_ARM_BLOCKER,
            "reason": (
                "The user-approved frozen supplier-CAD review Stage contains "
                "follower_left only. No transform, placement, or supplier-CAD "
                "right-arm Stage was approved, so no image was fabricated."
            ),
        },
        "source_runtime_evidence": {
            "numeric_structure": numeric["status"],
            "numeric_report": str(NUMERIC_REPORT.resolve()),
            "numeric_report_sha256": _sha256(NUMERIC_REPORT),
            "bottle_static_hold": bottle["status"],
            "bottle_report": str(BOTTLE_REPORT.resolve()),
            "bottle_report_sha256": _sha256(BOTTLE_REPORT),
            "bottle_screenshot_review": str(
                BOTTLE_SCREENSHOTS.resolve()
            ),
            "bottle_screenshot_review_sha256": _sha256(
                BOTTLE_SCREENSHOTS
            ),
            "bottle_phases": [
                item["phase"]
                for item in bottle_screenshots["records"]
            ],
            "task7_validation": str(TASK7_REPORT.resolve()),
            "task7_validation_sha256": _sha256(TASK7_REPORT),
        },
        "retake_history": RETAKE_HISTORY,
        "raw_capture_root": str(
            (
                ROOT
                / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
                "task7_robot_scope/pose_evidence_attempt5/screenshots_raw"
            ).resolve()
        ),
        "annotated_capture_root": str(
            (
                ROOT
                / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
                "task7_robot_scope/pose_evidence_attempt5/"
                "screenshots_annotated_v2"
            ).resolve()
        ),
        "screenshot_role": "AUXILIARY_EVIDENCE_NOT_PHYSICS_ACCEPTANCE",
        "task7": task7["status"],
        "task8": "NOT_RUN",
        "final_default_collider_modified": False,
    }
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = [
        (
            f"| {item['view']} | {item['phase']} | "
            f"`{item['raw']['absolute_path']}` | "
            f"`{item['annotated']['absolute_path']}` | PASS |"
        )
        for item in records
    ]
    OUTPUT_MD.write_text(
        "\n".join(
            [
                "# ALOHA ViperX Task 7 pose screenshot review",
                "",
                f"- Overall status: `{report['status']}`",
                f"- follower_left: `{left_status}`",
                "- follower_right: `NOT_RUN`",
                f"- blocker: `{RIGHT_ARM_BLOCKER}`",
                f"- Task 7: `{report['task7']}`",
                "- Task 8: `NOT_RUN`",
                "",
                "| View | Phase | Raw | Annotated | Visual review |",
                "|---|---|---|---|---|",
                *rows,
                "",
                (
                    "The six images are auxiliary evidence for the already "
                    "machine-verified follower_left symmetric-close poses. "
                    "The existing bottle report remains authoritative for "
                    "bilateral contact, release and static hold. No "
                    "follower_right image was manufactured because the "
                    "approved Stage does not contain it."
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"status={report['status']}")
    print(f"left_arm={left_status}")
    print(f"right_arm={report['right_arm']['status']}")
    print(f"json={OUTPUT.resolve()}")
    print(f"markdown={OUTPUT_MD.resolve()}")
    return 0 if report["status"] == "PARTIAL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
