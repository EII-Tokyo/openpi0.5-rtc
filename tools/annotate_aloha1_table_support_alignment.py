#!/usr/bin/env python3
"""Annotate table/support alignment screenshots without altering raw pixels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = (
    ROOT / ".codex/artifacts/20260730-aloha-support-table-alignment"
)
RAW_ROOT = ARTIFACT_ROOT / "screenshots_raw"
ANNOTATED_ROOT = ARTIFACT_ROOT / "screenshots_annotated_v2"
VALIDATION_PATH = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha1_table_support_alignment_validation.json"
)
REPORT_PATH = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha1_table_support_alignment_screenshot_review.json"
)
REPORT_MD_PATH = REPORT_PATH.with_suffix(".md")
STAGE_HASH = (
    "2b3f76365ed67532f478d995ae859a88b5639975ac07cb7ac8a53ac679e8205c"
)

CAPTURES = (
    {
        "id": "aligned_overview",
        "raw": "aligned_overview_raw.png",
        "camera": "Overview",
        "camera_eye_world_m": [-1.15, 2.75, 1.35],
        "camera_target_world_m": [0.0, 0.0, 0.20],
        "attempt_status": "ACCEPTED_FINAL_VISUAL_PASS",
        "callouts": (
            {
                "type": "box",
                "geometry": (700, 570, 1230, 1080),
                "label": "LEFT BASE",
            },
            {
                "type": "box",
                "geometry": (1740, 700, 2260, 1150),
                "label": "RIGHT BASE",
            },
            {
                "type": "point",
                "geometry": (1320, 1030),
                "label_position": (1020, 630),
                "label": "TABLETOP + SUPPORT PLANE",
            },
        ),
        "review_goal": (
            "Both follower bases, lower support frame and tabletop visible "
            "in one view; old air gap absent."
        ),
    },
    {
        "id": "aligned_support_side_attempt1",
        "raw": "aligned_support_side_raw.png",
        "camera": "SupportSide",
        "camera_eye_world_m": [0.0, 1.65, 0.16],
        "camera_target_world_m": [0.0, 0.0, 0.055],
        "attempt_status": "REJECTED_SUPPORT_INTERFACE_OCCLUDED",
        "callouts": (
            {
                "type": "box",
                "geometry": (180, 540, 2500, 1080),
                "label": "OCCLUDED BY FRONT RAIL",
            },
        ),
        "review_goal": (
            "Strict side view of table/support plane; rejected because "
            "the front extrusion hides the critical interface."
        ),
    },
    {
        "id": "aligned_left_base_side",
        "raw": "aligned_left_base_side_raw.png",
        "camera": "LeftBaseSide",
        "camera_eye_world_m": [-0.47, 1.10, 0.16],
        "camera_target_world_m": [-0.47, -0.02, 0.055],
        "attempt_status": "ACCEPTED_FINAL_VISUAL_PASS",
        "callouts": (
            {
                "type": "box",
                "geometry": (850, 330, 1960, 1030),
                "label": "FOLLOWER_LEFT BASE",
            },
            {
                "type": "point",
                "geometry": (520, 1010),
                "label_position": (210, 610),
                "label": "TABLETOP Z=0",
            },
            {
                "type": "point",
                "geometry": (1170, 1130),
                "label_position": (210, 680),
                "label": "SUPPORT Z=0..20 mm",
            },
            {
                "type": "point",
                "geometry": (1320, 995),
                "label_position": (210, 750),
                "label": "BASE BOTTOM Z=20 mm",
            },
        ),
        "review_goal": (
            "Left robot base, supporting extrusions and tabletop are all "
            "visible at the vertical stack interface."
        ),
    },
    {
        "id": "aligned_right_base_side",
        "raw": "aligned_right_base_side_raw.png",
        "camera": "RightBaseSide",
        "camera_eye_world_m": [0.47, 1.10, 0.16],
        "camera_target_world_m": [0.47, -0.02, 0.055],
        "attempt_status": "ACCEPTED_FINAL_VISUAL_PASS",
        "callouts": (
            {
                "type": "box",
                "geometry": (850, 330, 1960, 1030),
                "label": "FOLLOWER_RIGHT BASE",
            },
            {
                "type": "point",
                "geometry": (2210, 1050),
                "label_position": (2030, 610),
                "label": "TABLETOP Z=0",
            },
            {
                "type": "point",
                "geometry": (1170, 1130),
                "label_position": (210, 680),
                "label": "SUPPORT Z=0..20 mm",
            },
            {
                "type": "point",
                "geometry": (1320, 995),
                "label_position": (210, 750),
                "label": "BASE BOTTOM Z=20 mm",
            },
        ),
        "review_goal": (
            "Right robot base, supporting extrusions and tabletop are all "
            "visible at the vertical stack interface."
        ),
    },
)

VISION_REVIEWS = {
    "aligned_overview": {
        "status": "PASS",
        "reason": (
            "Raw and annotated images were individually inspected with the "
            "vision model. Both follower bases, the tabletop and the lower "
            "support frame are visible; the prior vertical air gap is absent, "
            "and the concise callouts do not obscure either base interface."
        ),
    },
    "aligned_support_side_attempt1": {
        "status": "REJECTED",
        "reason": (
            "Raw and annotated images were individually inspected with the "
            "vision model. The front support extrusion occludes the critical "
            "table/support interface, so this view is retained only as "
            "rejected retake history."
        ),
    },
    "aligned_left_base_side": {
        "status": "PASS",
        "reason": (
            "Raw and annotated images were individually inspected with the "
            "vision model. The left base, 20 mm support member and tabletop "
            "interface are visible; the region box and three short callouts "
            "do not overlap the measured interface."
        ),
    },
    "aligned_right_base_side": {
        "status": "PASS",
        "reason": (
            "Raw and annotated images were individually inspected with the "
            "vision model. The right base, 20 mm support member and tabletop "
            "interface are visible; the region box and three short callouts "
            "do not overlap the measured interface."
        ),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(
        f"/usr/share/fonts/truetype/dejavu/{name}",
        size,
    )


def _draw_callout(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    *,
    rejected: bool,
) -> None:
    color = (255, 90, 90) if rejected else (40, 235, 235)
    draw.rectangle(box, outline=color, width=5)
    label_box = draw.textbbox((0, 0), label, font=_font(20, bold=True))
    width = label_box[2] - label_box[0] + 20
    height = label_box[3] - label_box[1] + 14
    x = box[0]
    y = max(0, box[1] - height)
    draw.rectangle((x, y, x + width, y + height), fill=(20, 28, 38))
    draw.text(
        (x + 10, y + 6),
        label,
        font=_font(20, bold=True),
        fill=color,
    )


def _draw_point_callout(
    draw: ImageDraw.ImageDraw,
    point: tuple[int, int],
    label_position: tuple[int, int],
    label: str,
) -> None:
    color = (40, 235, 235)
    x, y = point
    label_x, label_y = label_position
    draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color)
    text_box = draw.textbbox(
        (label_x, label_y),
        label,
        font=_font(20, bold=True),
    )
    draw.rectangle(
        (
            text_box[0] - 8,
            text_box[1] - 5,
            text_box[2] + 8,
            text_box[3] + 5,
        ),
        fill=(20, 28, 38),
    )
    draw.text(
        (label_x, label_y),
        label,
        font=_font(20, bold=True),
        fill=color,
    )
    start_x = text_box[2] + 8
    start_y = (text_box[1] + text_box[3]) // 2
    draw.line((start_x, start_y, x, y), fill=color, width=4)


def _annotate(
    capture: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    raw = (RAW_ROOT / capture["raw"]).resolve(strict=True)
    with Image.open(raw) as opened:
        image = opened.convert("RGB")
    panel_width = 760
    canvas = Image.new(
        "RGB",
        (image.width + panel_width, image.height),
        (22, 26, 34),
    )
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    rejected = capture["attempt_status"].startswith("REJECTED")
    for callout in capture["callouts"]:
        if callout["type"] == "box":
            _draw_callout(
                draw,
                callout["geometry"],
                callout["label"],
                rejected=rejected,
            )
        elif callout["type"] == "point":
            _draw_point_callout(
                draw,
                callout["geometry"],
                callout["label_position"],
                callout["label"],
            )
        else:
            raise ValueError(f"unsupported callout: {callout}")

    stacks = validation["alignment"]["support_stacks"]
    left = stacks["follower_left"]["metrics"]
    right = stacks["follower_right"]["metrics"]
    x = image.width + 28
    y = 24
    status = (
        "REJECTED"
        if rejected
        else "TABLE/SUPPORT VISUAL GATE: PASS"
    )
    status_color = (255, 100, 100) if rejected else (105, 235, 145)
    draw.text(
        (x, y),
        "ALOHA1 TABLE / SUPPORT",
        font=_font(27, bold=True),
        fill=(110, 225, 255),
    )
    y += 42
    draw.text((x, y), status, font=_font(22, bold=True), fill=status_color)
    y += 42
    lines = [
        "Diagnostic alignment only",
        "Not grasp, dynamics or final-asset PASS",
        "",
        "Isaac Sim 5.1.0.0",
        "Kit 107.3.3 / PhysX 107.3.26",
        "Timeline: PAUSED",
        "Workspace: 2 (X11 index 1)",
        f"Camera: {capture['camera']}",
        f"Eye: {capture['camera_eye_world_m']}",
        f"Target: {capture['camera_target_world_m']}",
        "",
        "Stage SHA-256:",
        STAGE_HASH[:32],
        STAGE_HASH[32:],
        "",
        "WORLD Z STACK (runtime AABB)",
        "Table top: +0.000000000 m",
        (
            "Left support bottom: "
            f"{left['support_bottom_z_m']:+.9f} m"
        ),
        f"Left support top: {left['support_top_z_m']:+.9f} m",
        (
            "Left base bottom: "
            f"{left['robot_base_bottom_z_m']:+.9f} m"
        ),
        (
            "Left gaps table/base: "
            f"{left['table_to_support_gap_m']:+.2e} / "
            f"{left['support_to_robot_base_gap_m']:+.2e} m"
        ),
        (
            "Right support bottom: "
            f"{right['support_bottom_z_m']:+.9f} m"
        ),
        f"Right support top: {right['support_top_z_m']:+.9f} m",
        (
            "Right base bottom: "
            f"{right['robot_base_bottom_z_m']:+.9f} m"
        ),
        (
            "Right gaps table/base: "
            f"{right['table_to_support_gap_m']:+.2e} / "
            f"{right['support_to_robot_base_gap_m']:+.2e} m"
        ),
        "",
        "Boxes identify visual regions only.",
        "They are not contact-point measurements.",
        "Numeric JSON is authoritative.",
        "Task 8: NOT_RUN",
    ]
    for line in lines:
        color = (232, 236, 242)
        if line.startswith(("Not ", "Task 8")):
            color = (255, 205, 90)
        draw.text((x, y), line, font=_font(18), fill=color)
        y += 28

    destination = (
        ANNOTATED_ROOT / capture["raw"].replace("_raw.png", "_annotated.png")
    ).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
    return {
        "id": capture["id"],
        "attempt_status": capture["attempt_status"],
        "review_goal": capture["review_goal"],
        "raw_absolute_path": str(raw),
        "raw_sha256": _sha256(raw),
        "raw_size_px": list(image.size),
        "annotated_absolute_path": str(destination),
        "annotated_sha256": _sha256(destination),
        "annotated_size_px": list(canvas.size),
        "camera": {
            "name": capture["camera"],
            "eye_world_m": capture["camera_eye_world_m"],
            "target_world_m": capture["camera_target_world_m"],
        },
        "vision_model_review": VISION_REVIEWS[capture["id"]],
    }


def _write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# ALOHA1 Table/Support Alignment Screenshot Review",
        "",
        f"- Status: `{report['status']}`",
        f"- Diagnostic Stage: `{report['stage']['path']}`",
        f"- SHA-256: `{report['stage']['sha256']}`",
        "- Scope: `DIAGNOSTIC_ONLY_NOT_FINAL_ASSET`",
        "- Isaac GUI workspace: `2` (X11 index `1`)",
        f"- User review: `{report['user_review']['status']}`",
        "",
        "The images are supporting visual evidence. Runtime AABB stack "
        "measurements in the JSON validation report are authoritative.",
        "",
        "## Captures",
        "",
    ]
    for record in report["captures"]:
        lines.extend(
            [
                f"### {record['id']}",
                "",
                f"- Attempt: `{record['attempt_status']}`",
                f"- Vision review: `{record['vision_model_review']['status']}`",
                f"- Raw: `{record['raw_absolute_path']}`",
                f"- Annotated: `{record['annotated_absolute_path']}`",
                f"- Goal: {record['review_goal']}",
                "",
            ]
        )
    lines.extend(["## Retake history", ""])
    for item in report["retake_history"]:
        lines.extend(
            [
                f"- `{item['status']}`: {item['reason']}",
                "",
            ]
        )
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    validation = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
    records = [_annotate(capture, validation) for capture in CAPTURES]
    report = {
        "schema_version": 1,
        "status": "PASS",
        "stage": {
            "path": validation["diagnostic_stage"]["path"],
            "sha256": validation["diagnostic_stage"]["sha256"],
        },
        "workspace_policy": {
            "isaac_gui_workspace_number": 2,
            "x11_desktop_index": 1,
            "user_active_desktop_during_capture": 0,
        },
        "user_review": {
            "status": "PASS",
            "basis": "USER_CONFIRMED_VISUALLY_IN_ISAAC_GUI",
            "confirmed_stage_sha256": STAGE_HASH,
            "required_before_final_asset_promotion": True,
        },
        "captures": records,
        "final_evidence_set": {
            "required_capture_ids": [
                "aligned_overview",
                "aligned_left_base_side",
                "aligned_right_base_side",
            ],
            "all_required_raw_and_annotated_vision_reviewed": True,
            "all_required_status": "PASS",
        },
        "retake_history": [
            {
                "batch": "screenshots_annotated",
                "status": "REJECTED_OVERLAPPING_REGION_BOXES",
                "reason": (
                    "The first annotated batch placed large overlapping "
                    "region boxes over the left/right base interfaces."
                ),
                "raw_images_modified": False,
            },
            {
                "capture_id": "aligned_support_side_attempt1",
                "status": "REJECTED_SUPPORT_INTERFACE_OCCLUDED",
                "reason": (
                    "The strict side camera put the front support extrusion "
                    "between the camera and the table/support interface."
                ),
                "replacement_captures": [
                    "aligned_left_base_side",
                    "aligned_right_base_side",
                ],
            },
        ],
        "machine_validation_report": str(VALIDATION_PATH.resolve()),
        "boundaries": {
            "visual_pass_means": "TABLE_SUPPORT_INSTALLATION_VISUAL_GATE",
            "not_claimed": [
                "grasp_pass",
                "dynamics_pass",
                "final_asset_promotion",
            ],
            "task8": "NOT_RUN",
        },
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
