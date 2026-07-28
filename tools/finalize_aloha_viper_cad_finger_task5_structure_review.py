#!/usr/bin/env python3
"""Write the finalized Task 5 no-bottle screenshot review reports."""

from __future__ import annotations

import json
from pathlib import Path

from tools.aloha1_mapping.cad_finger_task5_structure_review import REQUIRED_CHECKS
from tools.aloha1_mapping.cad_finger_task5_structure_review import build_review_report
from tools.aloha1_mapping.cad_finger_task5_structure_review import render_markdown

ROOT = Path(__file__).resolve().parents[1]
RAW_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_structure.json"
)
ANNOTATION_METADATA = (
    ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
    "isaac_cad_finger/task5_structure/annotation_v2/"
    "annotation_metadata.json"
)
ASSET_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_asset.json"
)
OUTPUT_JSON = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_structure_screenshot_review.json"
)
OUTPUT_MD = OUTPUT_JSON.with_suffix(".md")


def _decisions(capture_names: list[str]) -> dict[str, dict[str, object]]:
    checks = dict.fromkeys(REQUIRED_CHECKS, True)
    return {
        capture_name: {
            "raw": "PASS",
            "annotated": "PASS",
            "conclusion": "PASS",
            "checks": dict(checks),
            "notes": (
                "Vision-model inspection: both handed fingers and inward "
                "surfaces are visible; mapping, crop, occlusion, state "
                "distinction, camera pairing, labels, and arrows pass. "
                "Verdict is limited to the structure visual gate."
            ),
        }
        for capture_name in capture_names
    }


RETAKE_HISTORY = [
    {
        "attempt": "raw_attempt12",
        "status": "REJECTED_OCCLUSION_AND_CROPPING",
        "reason": "critical gripper geometry was occluded or cropped",
    },
    {
        "attempt": "raw_attempt13",
        "status": "REJECTED_CAMERA_FOCUS_OCCLUSION",
        "reason": "camera-focus debug geometry obscured the target",
    },
    {
        "attempt": "raw_attempt14",
        "status": "REJECTED_GRIPPER_PROP_OCCLUSION",
        "reason": "gripper-prop geometry obscured the target",
    },
    {
        "attempt": "raw_attempt15",
        "status": "REJECTED_VISIBILITY_NOT_PROPAGATED",
        "reason": "child visibility did not hide all debug geometry",
    },
    {
        "attempt": "raw_attempt16",
        "status": "REJECTED_INCOMPLETE_LINK_FILTER",
        "reason": "non-target link filtering was incomplete",
    },
    {
        "attempt": "raw_attempt17",
        "status": "REJECTED_GPRIM_FILTER_GAP",
        "reason": "render Gprims escaped the visual filter",
    },
    {
        "attempt": "raw_attempt18",
        "status": "REJECTED_INSTANCE_PROXY_DEBUG_GEOMETRY",
        "reason": "instance-proxy debug geometry remained visible",
    },
    {
        "attempt": "raw_attempt19",
        "status": "REJECTED_BASE_OBLIQUE_TOO_DISTANT",
        "reason": "base-oblique framing did not expose finger detail",
    },
    {
        "attempt": "raw_attempt20",
        "status": "REJECTED_BASE_OBLIQUE_OFF_CENTER",
        "reason": "base-oblique target was off center",
    },
    {
        "attempt": "raw_attempt21",
        "status": "REJECTED_PROJECTION_PROBE_ONLY",
        "reason": "camera projection was probed but evidence was not final",
    },
    {
        "attempt": "raw_attempt22",
        "status": "REJECTED_POST_DYNAMIC_BODY_DRIFT",
        "reason": "post-failure body transforms corrupted static evidence",
    },
    {
        "attempt": "raw_attempt23",
        "status": "ACCEPTED_VISUAL_MODEL_REVIEW",
        "reason": (
            "fresh World reset and full legal-pose reinjection produced "
            "twelve individually reviewed raw images"
        ),
    },
    {
        "attempt": "annotation_v1",
        "status": "REJECTED_ANNOTATION_LEADER_OCCLUSION",
        "reason": (
            "a magenta inward-surface annotation leader crossed key finger "
            "geometry in base-oblique views"
        ),
    },
    {
        "attempt": "annotation_v2",
        "status": "ACCEPTED_VISUAL_MODEL_REVIEW",
        "reason": (
            "all twelve annotated images were individually reviewed; labels "
            "and arrows do not obscure key geometry"
        ),
    },
]


def main() -> int:
    raw_report = json.loads(RAW_REPORT.read_text(encoding="utf-8"))
    annotation_metadata = json.loads(
        ANNOTATION_METADATA.read_text(encoding="utf-8")
    )
    asset_report = json.loads(ASSET_REPORT.read_text(encoding="utf-8"))
    capture_names = [
        record["capture_name"] for record in raw_report["captures"]
    ]
    report = build_review_report(
        raw_report=raw_report,
        annotation_metadata=annotation_metadata,
        decisions=_decisions(capture_names),
        retake_history=RETAKE_HISTORY,
        approved_source_stage=asset_report["source_stage"],
    )
    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    OUTPUT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(f"status={report['status']}")
    print(f"json={OUTPUT_JSON.resolve()}")
    print(f"markdown={OUTPUT_MD.resolve()}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
