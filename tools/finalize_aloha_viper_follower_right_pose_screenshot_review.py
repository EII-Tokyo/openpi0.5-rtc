#!/usr/bin/env python3
"""Finalize follower_right raw/annotated visual evidence after review."""

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
    "aloha_viper_follower_right_pose_screenshots_raw.json"
)
NUMERIC_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_follower_right_one_joint_validation.json"
)
ARTIFACT_ROOT = (
    ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
    "follower_right_pose_evidence/attempt4_final"
)
ANNOTATION_METADATA = ARTIFACT_ROOT / "annotation_metadata_v2.json"
RAW_DECISIONS = ARTIFACT_ROOT / "raw_visual_review_decisions.json"
ANNOTATED_DECISIONS = (
    ARTIFACT_ROOT / "annotated_visual_review_decisions_v2.json"
)
OUTPUT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_follower_right_pose_screenshot_review.json"
)
SCOPE = "ROBOT_LOCAL_DIAGNOSTIC_NOT_WORKCELL_PLACEMENT"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def _image(path: Path, expected_hash: str, review: str) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if _sha256(path) != expected_hash:
        raise RuntimeError(f"reviewed image hash drift: {path}")
    with Image.open(path) as opened:
        opened.verify()
    with Image.open(path) as opened:
        resolution = [opened.width, opened.height]
        mode = opened.mode
    return {
        "absolute_path": str(path),
        "sha256": expected_hash,
        "resolution": resolution,
        "mode": mode,
        "readable": True,
        "visual_model_review": review,
    }


def main() -> int:
    raw = _load(RAW_REPORT)
    numeric = _load(NUMERIC_REPORT)
    annotations = _load(ANNOTATION_METADATA)
    raw_decisions = _load(RAW_DECISIONS)
    annotated_decisions = _load(ANNOTATED_DECISIONS)
    if raw["capture_status"] != "PASS":
        raise RuntimeError("raw screenshot machine gate is not PASS")
    if raw_decisions["status"] != "PASS":
        raise RuntimeError("raw visual-model review is not PASS")
    if annotations["status"] != "PENDING_ANNOTATED_VISUAL_MODEL_REVIEW":
        raise RuntimeError("unexpected annotation metadata status")
    if annotated_decisions["status"] != "PASS":
        raise RuntimeError("annotated visual-model review is not PASS")
    if numeric["scope"] != SCOPE:
        raise RuntimeError("numeric report is not follower_right robot-local")

    raw_by_name = {
        item["capture_name"]: item for item in raw["captures"]
    }
    annotation_by_name = {
        item["capture_name"]: item for item in annotations["records"]
    }
    names = set(raw_by_name)
    if (
        set(annotation_by_name) != names
        or set(raw_decisions["records"]) != names
        or set(annotated_decisions["records"]) != names
    ):
        raise RuntimeError("raw/annotation/review capture sets differ")

    records = []
    for name, raw_item in raw_by_name.items():
        annotation = annotation_by_name[name]
        raw_review = raw_decisions["records"][name]
        annotated_review = annotated_decisions["records"][name]
        if (
            raw_review["status"] != "PASS"
            or annotated_review["status"] != "PASS"
        ):
            raise RuntimeError(f"visual review failed for {name}")
        if annotation["raw_sha256"] != raw_item["file_sha256"]:
            raise RuntimeError(f"raw/annotation hash mismatch for {name}")
        if annotation["camera"] != raw_item["camera"]:
            raise RuntimeError(f"camera metadata drift for {name}")
        if annotation["simulation"] != raw_item["simulation"]:
            raise RuntimeError(f"simulation metadata drift for {name}")
        records.append(
            {
                "capture_name": name,
                "phase": raw_item["simulation"]["phase"],
                "view": raw_item["camera"]["view"],
                "numeric_status": raw_item["simulation"]["numeric_status"],
                "raw": _image(
                    Path(raw_item["absolute_path"]),
                    raw_item["file_sha256"],
                    "PASS",
                ),
                "annotated": _image(
                    Path(annotation["annotated_absolute_path"]),
                    annotation["annotated_sha256"],
                    "PASS",
                ),
                "camera": raw_item["camera"],
                "simulation": raw_item["simulation"],
                "raw_visual_review": raw_review,
                "annotated_visual_review": annotated_review,
            }
        )

    arm_pass = all(
        item["status"] == "PASS"
        for item in numeric["arm_one_joint_cases"]
    )
    other_runtime_pass = all(
        numeric[key]["status"] == "PASS"
        for key in ("first_frame_jump", "static_pose_hold", "determinism")
    )
    mimic_pass = (
        numeric["gripper_validation"]["maximum_mimic_residual_m"] <= 0.001
    )
    numeric_status = (
        "PASS"
        if arm_pass
        and other_runtime_pass
        and numeric["gripper_validation"]["status"] == "PASS"
        else "PARTIAL"
        if arm_pass and other_runtime_pass
        else "FAIL"
    )
    camera_signatures: dict[str, set[tuple[Any, ...]]] = {}
    image_hashes: dict[str, set[str]] = {}
    for item in records:
        camera = item["camera"]
        camera_signatures.setdefault(item["view"], set()).add(
            (
                tuple(camera["position_world_m"]),
                tuple(camera["orientation_wxyz"]),
                tuple(camera["target_world_m"]),
            )
        )
        image_hashes.setdefault(item["view"], set()).add(
            item["raw"]["sha256"]
        )
    fixed_camera = all(
        len(signatures) == 1
        for signatures in camera_signatures.values()
    )
    distinct_states = (
        len(image_hashes["full_arm_oblique"]) == 3
        and len(image_hashes["gripper_closeup"]) == 4
    )
    visual_pass = (
        len(records) == 7
        and fixed_camera
        and distinct_states
        and all(
            item["raw"]["visual_model_review"] == "PASS"
            and item["annotated"]["visual_model_review"] == "PASS"
            for item in records
        )
    )
    report = {
        "schema_version": 1,
        "status": (
            "PARTIAL"
            if visual_pass and numeric_status == "PARTIAL"
            else "PASS"
            if visual_pass and numeric_status == "PASS"
            else "FAIL"
        ),
        "scope": SCOPE,
        "visual_installation_pose_gate": (
            "PASS" if visual_pass else "FAIL"
        ),
        "numeric_runtime_status": numeric_status,
        "arm_one_joint_direction_range": "PASS" if arm_pass else "FAIL",
        "gripper_motion_direction": numeric["gripper_validation"][
            "motion_direction"
        ],
        "aperture_monotonicity": numeric["gripper_validation"][
            "aperture_monotonicity"
        ],
        "mimic_accuracy": "PASS" if mimic_pass else "FAIL",
        "maximum_mimic_residual_m": numeric["gripper_validation"][
            "maximum_mimic_residual_m"
        ],
        "first_frame_jump": numeric["first_frame_jump"]["status"],
        "static_pose_hold": numeric["static_pose_hold"]["status"],
        "determinism": numeric["determinism"]["status"],
        "records": records,
        "fixed_camera_within_each_view": fixed_camera,
        "states_visually_distinct": distinct_states,
        "retake_history": [
            *raw_decisions["retake_history"],
            *annotated_decisions["rejected_annotation_history"],
        ],
        "raw_capture_root": str(
            (ARTIFACT_ROOT / "screenshots_raw").resolve()
        ),
        "annotated_capture_root": str(
            (ARTIFACT_ROOT / "screenshots_annotated_v2").resolve()
        ),
        "source_numeric_report": {
            "absolute_path": str(NUMERIC_REPORT.resolve()),
            "sha256": _sha256(NUMERIC_REPORT),
        },
        "source_stage_immutable": raw["source_stage_immutable"],
        "source_numeric_report_immutable": raw[
            "source_numeric_report_immutable"
        ],
        "workcell_placement_verified": False,
        "hard_blockers": [
            "HARD_BLOCKER_FOLLOWER_RIGHT_WORKCELL_INSTALL_TRANSFORM"
        ],
        "screenshot_role": "AUXILIARY_EVIDENCE_NOT_RUNTIME_ACCEPTANCE",
        "task8": "NOT_RUN",
    }
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = [
        (
            f"| {item['phase']} | {item['view']} | "
            f"`{item['numeric_status']}` | "
            f"`{item['raw']['absolute_path']}` | "
            f"`{item['annotated']['absolute_path']}` | PASS |"
        )
        for item in records
    ]
    OUTPUT.with_suffix(".md").write_text(
        "\n".join(
            [
                "# follower_right robot-local pose screenshot review",
                "",
                f"- Overall: `{report['status']}`",
                "- Visual installation/pose gate: `PASS`",
                f"- Numeric runtime: `{numeric_status}`",
                f"- Mimic accuracy: `{report['mimic_accuracy']}`",
                f"- Scope: `{SCOPE}`",
                "- Workcell placement: `NOT_VERIFIED`",
                "- Task 8: `NOT_RUN`",
                "",
                "| Phase | View | Numeric | Raw | Annotated | Visual |",
                "|---|---|---|---|---|---|",
                *rows,
                "",
                "The visual PASS proves only that the robot-local supplier "
                "finger installation and replayed poses are reviewable. It "
                "does not override the numeric mimic failure or prove a "
                "dual-arm workcell placement.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"status={report['status']}")
    print(f"record_count={len(records)}")
    print(f"output={OUTPUT}")
    return 0 if report["status"] in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
