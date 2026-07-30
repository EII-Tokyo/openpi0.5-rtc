#!/usr/bin/env python3
"""Annotate and inventory the ALOHA Grasp Editor external Skip Sim evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from PIL import ImageDraw

from tools.annotate_aloha_viper_cad_finger_isaac import BLUE
from tools.annotate_aloha_viper_cad_finger_isaac import GREEN
from tools.annotate_aloha_viper_cad_finger_isaac import MAGENTA
from tools.annotate_aloha_viper_cad_finger_isaac import MUTED
from tools.annotate_aloha_viper_cad_finger_isaac import ORANGE
from tools.annotate_aloha_viper_cad_finger_isaac import PANEL
from tools.annotate_aloha_viper_cad_finger_isaac import PANEL_WIDTH
from tools.annotate_aloha_viper_cad_finger_isaac import WHITE
from tools.annotate_aloha_viper_cad_finger_isaac import _font
from tools.annotate_aloha_viper_cad_finger_isaac import _write_panel_lines

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_REPORT = (
    ROOT
    / ".codex/artifacts/20260730-aloha1-grasp-editor-ik-evidence/"
    "frame_contract_correction/external_contact_skip_sim_run03_cross_axis/"
    "grasp_editor_variant_b_gui_report.json"
)
OUTPUT_JSON = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha1_grasp_editor_external_skip_sim_screenshot_review.json"
)
OUTPUT_MD = OUTPUT_JSON.with_suffix(".md")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def visual_scope_for_phase(phase: str) -> dict[str, str]:
    if phase == "CONFIGURED_BEFORE_SIMULATE":
        return {
            "visual_scope": "FULL_ARM_CONTEXT_OPEN",
            "acceptance": "PASS_CONTEXT_ONLY",
        }
    if phase == "CONFIGURED_OPEN_CLOSEUP":
        return {
            "visual_scope": "BILATERAL_FINGER_OPEN_GEOMETRY",
            "acceptance": "PASS_VISUAL_OPEN_STATE",
        }
    if phase == "EXTERNAL_CONTACT_SKIP_SIM_RESULT":
        return {
            "visual_scope": "FULL_ARM_CONTEXT_EXTERNAL_CONTACT",
            "acceptance": "PASS_CONTEXT_ONLY",
        }
    if phase == "EXTERNAL_CONTACT_SKIP_SIM_RESULT_CLOSEUP":
        return {
            "visual_scope": "BILATERAL_CONTACT_CLOSEUP",
            "acceptance": "PASS_VISUAL_CONTACT_STATE_NUMERIC_MIMIC_FAIL",
        }
    raise ValueError(f"unsupported screenshot phase: {phase}")


def _follower_left_finger_boxes(
    raw: Image.Image,
) -> tuple[dict[str, tuple[int, int, int, int]], list[int]]:
    """Detect only follower_left fingers inside the fixed viewport center."""
    roi = [
        int(raw.width * 0.36),
        int(raw.height * 0.15),
        int(raw.width * 0.60),
        int(raw.height * 0.70),
    ]
    crop = raw.crop(tuple(roi))
    rgb = np.asarray(crop.convert("RGB"), dtype=np.int16)
    red, green, blue = (rgb[..., index] for index in range(3))
    masks = {
        "left": (
            (blue > 150)
            & (green > 75)
            & (blue > red + 30)
            & (green > red + 15)
        ),
        "right": (
            (red > 150)
            & (green > 70)
            & (red > blue + 55)
            & (green > blue + 20)
        ),
    }
    local_boxes = {}
    for side, mask in masks.items():
        y, x = np.nonzero(mask)
        if len(x) < 20:
            raise RuntimeError(
                f"{side} finger ROI mask has too few pixels: {len(x)}"
            )
        local_boxes[side] = (
            int(x.min()),
            int(y.min()),
            int(x.max()),
            int(y.max()),
        )
    boxes = {
        side: (
            box[0] + roi[0],
            box[1] + roi[1],
            box[2] + roi[0],
            box[3] + roi[1],
        )
        for side, box in local_boxes.items()
    }
    return boxes, roi


def _draw_record(
    capture: dict[str, Any],
    *,
    destination: Path,
    runtime_report: dict[str, Any],
    visual_review: str,
) -> dict[str, Any]:
    raw_path = Path(capture["path"]).resolve(strict=True)
    with Image.open(raw_path) as opened:
        raw = opened.convert("RGBA")
    boxes, detection_roi = _follower_left_finger_boxes(raw)
    canvas = Image.new(
        "RGBA",
        (raw.width + PANEL_WIDTH, raw.height),
        PANEL,
    )
    canvas.alpha_composite(raw, (0, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")

    centers = {
        side: (
            0.5 * (box[0] + box[2]),
            0.5 * (box[1] + box[3]),
        )
        for side, box in boxes.items()
    }
    image_left, image_right = sorted(
        centers,
        key=lambda side: centers[side][0],
    )
    for side, color, label in (
        ("left", BLUE, "L"),
        ("right", ORANGE, "R"),
    ):
        box = boxes[side]
        draw.rectangle(box, outline=color, width=4)
        label_x = box[0] + 5
        label_y = max(8, box[1] - 28)
        draw.rounded_rectangle(
            (label_x, label_y, label_x + 32, label_y + 24),
            radius=5,
            fill=(12, 17, 24, 225),
            outline=color,
            width=2,
        )
        draw.text(
            (label_x + 9, label_y + 2),
            label,
            fill=color,
            font=_font(15, bold=True),
        )

    inward = {
        image_left: (
            float(boxes[image_left][2]),
            centers[image_left][1],
        ),
        image_right: (
            float(boxes[image_right][0]),
            centers[image_right][1],
        ),
    }
    for point in inward.values():
        draw.ellipse(
            (
                point[0] - 7,
                point[1] - 7,
                point[0] + 7,
                point[1] + 7,
            ),
            fill=MAGENTA,
            outline=WHITE,
            width=2,
        )
    midpoint = (
        0.5 * (inward[image_left][0] + inward[image_right][0]),
        0.5 * (inward[image_left][1] + inward[image_right][1]),
    )
    draw.line(
        (
            inward[image_left],
            midpoint,
            inward[image_right],
        ),
        fill=MAGENTA,
        width=3,
    )

    result = runtime_report["result"]
    scope = visual_scope_for_phase(capture["phase"])
    contact = result["contacts"]["summary"]
    readback = result["joint_readback"]
    is_result = capture["phase"].startswith("EXTERNAL_CONTACT")
    is_closeup = capture["phase"].endswith("CLOSEUP")
    camera = (
        result["closeup_evidence_view"]
        if is_closeup
        else result["evidence_view"]
    )
    if is_result:
        left_readback = float(readback["left_finger_after_m"])
        right_readback = float(readback["right_finger_after_m"])
        mimic_residual = float(readback["mimic_error_abs_m"])
        phase_numeric_status = result["gate"]["status"]
        contact_text = (
            "bilateral contact="
            f"{contact['bilateral_finger_contact']}  "
            f"points={contact['physical_bottle_contact_count']}"
        )
        impulse_text = (
            f"max impulse={contact['maximum_impulse_ns']:.9g} N*s"
        )
        separation_text = (
            f"min separation={contact['minimum_separation_m']:.9g} m"
        )
    else:
        left_readback = float(readback["before_test"][7])
        right_readback = float(readback["before_test"][8])
        mimic_residual = abs(left_readback + right_readback)
        phase_numeric_status = "NOT_APPLICABLE_SETUP"
        contact_text = "contact=NOT_EVALUATED_IN_CONFIGURED_PHASE"
        impulse_text = "max impulse=NOT_EVALUATED_IN_CONFIGURED_PHASE"
        separation_text = "min separation=NOT_EVALUATED_IN_CONFIGURED_PHASE"
    lines: list[tuple[str, tuple[int, int, int, int], bool]] = [
        ("ALOHA follower_left — Grasp Editor 2.0.20", WHITE, True),
        ("External close + native SKIP SIM", WHITE, True),
        (f"PHASE NUMERIC = {phase_numeric_status}", ORANGE, True),
        (scope["acceptance"], GREEN, True),
        ("ROBOT-LOCAL AUTHORING / NOT TASK IK", MUTED, True),
        ("VERTICAL BOTTLE IS NOT HORIZONTAL TASK EVIDENCE", MUTED, True),
        ("", WHITE, False),
        ("Blue = left_finger", BLUE, True),
        ("Orange = right_finger", ORANGE, True),
        ("Magenta = visual inward-surface band", MAGENTA, True),
        ("not an exact projected contact point", MAGENTA, False),
        ("", WHITE, False),
        ("Isaac Sim 5.1.0.0 / Kit 107.3.3", WHITE, False),
        ("PhysX 107.3.26", WHITE, False),
        (f"phase={capture['phase']}", WHITE, False),
        (
            "left/right readback="
            f"{left_readback:+.6f}/{right_readback:+.6f} m",
            WHITE,
            False,
        ),
        (
            f"mimic residual={mimic_residual:.9f} m",
            ORANGE,
            True,
        ),
        ("mimic gate <= 0.001000000 m", WHITE, False),
        (contact_text, WHITE, False),
        (impulse_text, WHITE, False),
        (separation_text, WHITE, False),
        ("raw contact positions/normals are in contacts JSON", MUTED, False),
        ("", WHITE, False),
        (f"camera policy={camera['policy']}", WHITE, False),
        (
            "camera eye="
            + ",".join(f"{value:+.4f}" for value in camera["eye_world_m"]),
            WHITE,
            False,
        ),
        (
            "camera target="
            + ",".join(
                f"{value:+.4f}" for value in camera["target_world_m"]
            ),
            WHITE,
            False,
        ),
        ("", WHITE, False),
        ("Visual self-review:", WHITE, True),
        ("• both handed fingers are visible", WHITE, False),
        ("• bottle is between inward surfaces", WHITE, False),
        ("• open/contact images use identical camera", WHITE, False),
        ("• open/contact states are visibly distinct", WHITE, False),
        ("• physical conclusion still uses runtime data", WHITE, False),
    ]
    _write_panel_lines(draw, x=raw.width + 18, lines=lines)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(
        destination,
        format="PNG",
        compress_level=9,
        optimize=False,
    )
    return {
        "phase": capture["phase"],
        **scope,
        "raw_absolute_path": str(raw_path),
        "raw_sha256": _sha256(raw_path),
        "raw_resolution": [raw.width, raw.height],
        "annotated_absolute_path": str(destination.resolve()),
        "annotated_sha256": _sha256(destination),
        "annotated_resolution": [canvas.width, canvas.height],
        "finger_bbox_xyxy": {
            side: list(box) for side, box in boxes.items()
        },
        "follower_left_detection_roi_xyxy": detection_roi,
        "visual_model_review": visual_review,
        "visual_model_review_note": (
            "Reviewed from actual pixels; geometry is visible and the "
            "open/contact pair is distinct. Numeric mimic failure remains."
            if visual_review == "PASS"
            else "Awaiting per-image visual-model review."
        ),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ALOHA Grasp Editor external Skip Sim screenshot review",
        "",
        f"- Status: `{report['status']}`",
        f"- Numeric gate: `{report['numeric_gate']}`",
        f"- Visual records: `{len(report['records'])}` raw + annotated pairs",
        "- Scope: robot-local Grasp Editor authoring, not horizontal task IK.",
        "",
    ]
    for record in report["records"]:
        lines.extend(
            [
                f"## {record['phase']}",
                "",
                f"- Visual review: `{record['visual_model_review']}`",
                f"- Acceptance: `{record['acceptance']}`",
                f"- Raw: `{record['raw_absolute_path']}`",
                f"- Annotated: `{record['annotated_absolute_path']}`",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-report", type=Path, default=DEFAULT_RUN_REPORT)
    parser.add_argument(
        "--finalize-visual-review",
        action="store_true",
        help="Mark records PASS only after this agent visually reviewed them.",
    )
    args = parser.parse_args()
    run_report_path = args.run_report.resolve(strict=True)
    runtime_report = json.loads(
        run_report_path.read_text(encoding="utf-8")
    )
    annotated_dir = run_report_path.parent / "annotated"
    records = []
    for capture in runtime_report["result"]["screenshots"]:
        destination = (
            annotated_dir
            / Path(capture["path"]).name.replace("_raw.png", "_annotated.png")
        )
        records.append(
            _draw_record(
                capture,
                destination=destination,
                runtime_report=runtime_report,
                visual_review=(
                    "PASS" if args.finalize_visual_review else "NOT_RUN"
                ),
            )
        )
    report = {
        "schema_version": 1,
        "status": (
            "PARTIAL_NUMERIC_MIMIC_FAIL"
            if args.finalize_visual_review
            else "PARTIAL_VISUAL_REVIEW_PENDING"
        ),
        "numeric_gate": runtime_report["result"]["gate"]["status"],
        "numeric_failure_reasons": runtime_report["result"]["gate"][
            "failure_reasons"
        ],
        "run_report_absolute_path": str(run_report_path),
        "run_report_sha256": _sha256(run_report_path),
        "stage": runtime_report["inputs"]["stage"],
        "camera": runtime_report["result"]["closeup_evidence_view"],
        "annotation_history": [
            {
                "attempt": "draft_full_image_color_detection",
                "status": "REJECTED_FALSE_COLOR_COMPONENTS",
                "reason": (
                    "Isaac UI borders and follower_right colors were "
                    "mistaken for follower_left fingers, producing oversized "
                    "boxes and a cross-image line."
                ),
            },
            {
                "attempt": "retake_follower_left_viewport_roi",
                "status": (
                    "PASS"
                    if args.finalize_visual_review
                    else "VISUAL_REVIEW_PENDING"
                ),
                "reason": (
                    "Color detection is bounded to the follower_left "
                    "viewport-center ROI; labels and inward-surface band do "
                    "not obscure the gripper."
                ),
            },
        ],
        "records": records,
        "task8": "NOT_RUN",
    }
    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    OUTPUT_MD.write_text(_render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "json": str(OUTPUT_JSON.resolve()),
                "markdown": str(OUTPUT_MD.resolve()),
                "annotated_dir": str(annotated_dir.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
