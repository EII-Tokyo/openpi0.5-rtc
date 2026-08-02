#!/usr/bin/env python3
"""Annotate repeated Task 7 helper-body candidate failures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import textwrap
from typing import Any

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from tools.aloha1_mapping.task7_failure_screenshot_gate import classify_review


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def _bbox(projection: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int] | None:
    minimum = projection.get("minimum_px")
    maximum = projection.get("maximum_px")
    if minimum is None or maximum is None:
        return None
    x0 = max(0, min(width - 1, int(minimum[0]) - 10))
    y0 = max(0, min(height - 1, int(minimum[1]) - 10))
    x1 = max(0, min(width - 1, int(maximum[0]) + 10))
    y1 = max(0, min(height - 1, int(maximum[1]) + 10))
    return x0, y0, x1, y1


def _draw_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int] | None,
    color: tuple[int, int, int],
    label: str,
) -> None:
    if box is None:
        return
    draw.rectangle(box, outline=color, width=5)
    draw.rectangle((box[0], box[1], box[0] + 300, box[1] + 38), fill=(0, 0, 0, 210))
    draw.text((box[0] + 8, box[1] + 5), label, fill=color, font=_font(22))


def _annotate(raw: Path, destination: Path, capture: dict[str, Any], report: dict[str, Any]) -> None:
    image = Image.open(raw).convert("RGBA")
    width, height = image.size
    panel_width = 520
    canvas = Image.new("RGBA", (width + panel_width, height), (18, 20, 25, 255))
    canvas.alpha_composite(image, (0, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")
    if capture["view"] == "whole_arm_oblique":
        _draw_box(
            draw,
            _bbox(capture["projections"]["base"], width, height),
            (255, 50, 50),
            "BASE collider group",
        )
    gripper_box = _bbox(capture["projections"]["gripper"], width, height)
    _draw_box(
        draw,
        gripper_box,
        (255, 170, 20),
        "GRIPPER + FINGER colliders",
    )
    helpers = capture["projections"]["helpers"]
    if helpers["minimum_px"] is not None:
        center_x = int((helpers["minimum_px"][0] + helpers["maximum_px"][0]) / 2)
        center_y = int((helpers["minimum_px"][1] + helpers["maximum_px"][1]) / 2)
        draw.ellipse((center_x - 14, center_y - 14, center_x + 14, center_y + 14), fill=(210, 80, 255, 235))
        if capture["view"] == "whole_arm_oblique" and gripper_box is not None:
            label_x = min(width - 140, center_x + 45)
            label_y = min(height - 45, gripper_box[3] + 18)
        else:
            label_x = min(width - 140, center_x + 28)
            label_y = max(65, center_y - 62)
        draw.line(
            (center_x, center_y, label_x - 8, label_y + 12),
            fill=(210, 80, 255),
            width=5,
        )
        draw.text(
            (label_x, label_y),
            "H1-H3",
            fill=(235, 175, 255),
            font=_font(20),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    title = "REJECTED CANDIDATE — VIRTUAL HELPER RIGID-BODY REMOVAL"
    draw.rectangle((0, 0, width, 58), fill=(95, 0, 0, 225))
    draw.text((22, 13), title, fill=(255, 255, 255), font=_font(26))
    x = width + 24
    y = 24
    heading = _font(23)
    body = _font(18)
    small = _font(15)
    draw.text((x, y), "Machine verdict: FAIL", fill=(255, 90, 90), font=heading)
    y += 43
    facts = [
        f"Follower: {capture['follower']}",
        f"View: {capture['view']}",
        "Isaac Sim 5.1.0.0 / Kit 107.3.3",
        "PhysX 107.3.26",
        "Fresh processes: 2",
        f"Repeated new clash findings: {report['failure']['non_adjacent_clash_count']}",
        f"Signature: {report['failure']['deterministic_signature'][:20]}…",
    ]
    for line in facts:
        draw.text((x, y), line, fill=(235, 235, 235), font=body)
        y += 30
    y += 12
    draw.text((x, y), "What failed", fill=(255, 205, 80), font=heading)
    y += 38
    explanation = (
        "Removing RigidBodyAPI/MassAPI from the three empty fixed helper frames "
        "changes the validator's body adjacency/ownership graph. It removes the "
        "three original missing-collider findings, but introduces 57 deterministic "
        "NonAdjacentCollisionMeshesDoNotClash findings."
    )
    for line in textwrap.wrap(explanation, width=48):
        draw.text((x, y), line, fill=(240, 240, 240), font=body)
        y += 27
    y += 14
    draw.text((x, y), "Evidence meaning", fill=(120, 210, 255), font=heading)
    y += 38
    meaning = (
        "Colored geometry is a session-only visual clone of authored colliders. "
        "Boxes identify collider groups; the purple marker identifies helper-frame "
        "locations. It is not a contact point. This capture did not step physics, "
        "apply a legal finger qpos, or read back the articulation. Finger orientation "
        "and finger-pair collision response are NOT EVALUATED. Runtime parameters and "
        "final assets were not changed."
    )
    for line in textwrap.wrap(meaning, width=48):
        draw.text((x, y), line, fill=(220, 230, 240), font=body)
        y += 25
    y += 12
    draw.text((x, y), "Decision: DO NOT PROMOTE", fill=(255, 100, 100), font=heading)
    y += 40
    stage_lines = textwrap.wrap(report["stage"]["absolute_path"], width=58)
    draw.text((x, y), "Stage:", fill=(180, 180, 180), font=small)
    y += 22
    for line in stage_lines[:5]:
        draw.text((x, y), line, fill=(160, 160, 160), font=small)
        y += 20
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(destination, format="PNG", optimize=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-report", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument(
        "--review-status",
        choices=("PENDING", "PASS"),
        default="PENDING",
    )
    args = parser.parse_args()
    capture_report = args.capture_report.resolve(strict=True)
    report = json.loads(capture_report.read_text(encoding="utf-8"))
    annotated_root = capture_report.parent / "screenshots_annotated"
    records = []
    for capture in report["captures"]:
        raw = Path(capture["raw_absolute_path"]).resolve(strict=True)
        annotated = annotated_root / raw.name.replace("_raw.png", "_annotated.png")
        _annotate(raw, annotated, capture, report)
        record = {
            **capture,
            "annotated_absolute_path": str(annotated.resolve(strict=True)),
            "annotated_sha256": _sha256(annotated),
            "visual_model_review": args.review_status,
            "visual_review_checks": {
                "whole_arm_or_failure_region_visible": args.review_status,
                "collision_overlay_visible": args.review_status,
                "labels_do_not_hide_failure_region": args.review_status,
                "failure_reason_marked": args.review_status,
                "raw_and_annotated_are_distinct": args.review_status,
            },
        }
        record["review_classification"] = classify_review(
            requested_visual_status=args.review_status,
            capture=record,
        )
        record["visual_model_review"] = record["review_classification"][
            "visual_model_review"
        ]
        records.append(record)
    classifications = [record["review_classification"] for record in records]
    overall_status = (
        "PASS"
        if classifications and all(item["status"] == "PASS" for item in classifications)
        else "PARTIAL"
    )
    geometry_status = (
        "PASS"
        if classifications
        and all(
            item["finger_installation_and_collision_gate"] == "PASS"
            for item in classifications
        )
        else "NOT_RUN"
    )
    final = {
        **report,
        "status": overall_status,
        "reason": (
            "VISUAL_FAILURE_EVIDENCE_LEGIBLE_BUT_FINGER_GEOMETRY_NOT_EVALUATED"
            if args.review_status == "PASS" and geometry_status != "PASS"
            else (
                "VISUAL_AND_FINGER_GEOMETRY_GATES_VERIFIED"
                if overall_status == "PASS"
                else "PENDING_VISUAL_MODEL_REVIEW"
            )
        ),
        "capture_report": {
            "absolute_path": str(capture_report),
            "sha256": _sha256(capture_report),
        },
        "captures": records,
        "visual_model_review": (
            "PASS"
            if overall_status == "PASS"
            else (
                "PASS_LEGIBILITY_ONLY"
                if args.review_status == "PASS"
                else "PENDING"
            )
        ),
        "visual_evidence_legibility": args.review_status,
        "finger_installation_and_collision_gate": geometry_status,
        "scope_boundary": (
            "The overlay images show the rejected helper-body candidate and its "
            "authored collider groups. They do not establish legal finger qpos, "
            "supplier-CAD palm orientation, finger-pair collision response, or "
            "runtime grasp validity unless a separate finger_geometry_gate passes."
        ),
    }
    output = args.output_report.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": final["status"], "captures": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
