#!/usr/bin/env python3
"""Annotate supplier-CAD Task 5 bottle screenshots from runtime readback."""

from __future__ import annotations

import argparse
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
from tools.annotate_aloha_viper_cad_finger_isaac import _arrow_head
from tools.annotate_aloha_viper_cad_finger_isaac import _finger_boxes
from tools.annotate_aloha_viper_cad_finger_isaac import _font
from tools.annotate_aloha_viper_cad_finger_isaac import _sha256
from tools.annotate_aloha_viper_cad_finger_isaac import _wrap
from tools.annotate_aloha_viper_cad_finger_isaac import _write_panel_lines

ROOT = Path(__file__).resolve().parents[1]
RAW_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_bottle.json"
)
OUTPUT_ROOT = (
    ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
    "isaac_cad_finger/task5_bottle_acceptance_v2_annotation"
)


def _bottle_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    """Detect the green diagnostic bottle in the rendered evidence."""

    rgba = np.asarray(image.convert("RGBA"), dtype=np.int16)
    red = rgba[..., 0]
    green = rgba[..., 1]
    blue = rgba[..., 2]
    mask = (
        (green >= 105)
        & (green >= red + 25)
        & (green >= blue + 12)
    )
    rows, columns = np.nonzero(mask)
    if columns.size < 100:
        raise RuntimeError("green bottle projection is not detectable")
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()),
        int(rows.max()),
    )


def _draw_contact_projection(
    draw: ImageDraw.ImageDraw,
    projection: dict[str, Any],
) -> None:
    for side, color in (("left", BLUE), ("right", ORANGE)):
        record = projection.get(side)
        if record is None:
            continue
        point = tuple(float(value) for value in record["contact_pixel_xy"])
        endpoint = tuple(
            float(value)
            for value in record["normal_endpoint_pixel_xy"]
        )
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
        draw.line((point, endpoint), fill=color, width=4)
        _arrow_head(draw, tip=endpoint, toward=point)


def _phase_gate(phase: str) -> tuple[str, tuple[int, int, int, int]]:
    gates = {
        "open": (
            "PHASE EVIDENCE — NOT HOLD PASS",
            WHITE,
        ),
        "bilateral_contact": (
            "PASS — PHYSICAL BILATERAL CONTACT",
            GREEN,
        ),
        "release": (
            "PASS — RELEASED WITHOUT CONSTRAINT",
            GREEN,
        ),
        "hold_end": (
            "PASS — 20/20 STATIC HOLD GATE",
            GREEN,
        ),
    }
    return gates[phase]


def _draw_annotations(
    capture: dict[str, Any],
    report: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    raw_path = Path(capture["absolute_path"]).resolve(strict=True)
    with Image.open(raw_path) as opened:
        raw = opened.convert("RGBA")
    boxes = _finger_boxes(raw)
    bottle_box = _bottle_bbox(raw)
    canvas = Image.new(
        "RGBA",
        (raw.width + PANEL_WIDTH, raw.height),
        PANEL,
    )
    canvas.alpha_composite(raw, (0, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")

    for side, color, label in (
        ("left", BLUE, "L"),
        ("right", ORANGE, "R"),
    ):
        box = boxes[side]
        draw.rectangle(box, outline=color, width=4)
        image_left = box[0] < raw.width / 2
        label_x = box[0] + 8 if image_left else box[2] - 40
        label_y = box[1] + 8
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
    draw.rectangle(bottle_box, outline=GREEN, width=3)

    projection = capture["camera"].get("contact_projection", {})
    if projection:
        _draw_contact_projection(draw, projection)
    else:
        centers = {
            side: (
                (box[0] + box[2]) / 2.0,
                (box[1] + box[3]) / 2.0,
            )
            for side, box in boxes.items()
        }
        for side, box in boxes.items():
            image_left = centers[side][0] < raw.width / 2
            point = (
                float(box[2] - 8 if image_left else box[0] + 8),
                float(centers[side][1]),
            )
            draw.ellipse(
                (
                    point[0] - 6,
                    point[1] - 6,
                    point[0] + 6,
                    point[1] + 6,
                ),
                fill=MAGENTA,
                outline=WHITE,
                width=2,
            )

    phase = capture["capture_name"]
    simulation = capture["simulation"]
    bottle = simulation["bottle"]
    gate_text, gate_color = _phase_gate(phase)
    drop_m = (
        float(report["first_trial"]["released_hold"]["drop_m"])
        if phase == "hold_end"
        else 0.0
    )
    pose_velocity_text = (
        "pose-derived final vz="
        f"{float(report['first_trial']['released_hold']['pose_derived_vertical_velocity']['final_m_s']):+.6f} m/s"
        if phase == "hold_end"
        else "pose-derived hold velocity=N/A before hold interval"
    )
    first_trial = report["first_trial"]
    fixed = first_trial["contacts"]
    contact_lines = []
    if phase != "open":
        left = (
            fixed["left_all"]["last_contact"]
            if phase == "hold_end"
            else fixed["left_fixed"]["first_contact"]
        )
        right = (
            fixed["right_all"]["last_contact"]
            if phase == "hold_end"
            else fixed["right_fixed"]["first_contact"]
        )
        contact_lines = [
            (
                "L/R separation="
                f"{left['separation_m'] * 1e3:+.4f}/"
                f"{right['separation_m'] * 1e3:+.4f} mm",
                WHITE,
                False,
            ),
            (
                "L/R |N impulse|="
                f"{left.get('normal_impulse_n_s', 0.0):.6f}/"
                f"{right.get('normal_impulse_n_s', 0.0):.6f} N·s",
                WHITE,
                False,
            ),
        ]

    lines: list[tuple[str, tuple[int, int, int, int], bool]] = [
        ("ALOHA ViperX follower_left", WHITE, True),
        ("Supplier-CAD finger — 20 g bottle Task 5", WHITE, True),
        (gate_text, gate_color, True),
        (
            "FIXED BOTTLE; NOT HOLD"
            if phase == "bilateral_contact"
            else "NO FIXED JOINT / SURFACE GRIPPER / PARENT",
            ORANGE if phase == "bilateral_contact" else GREEN,
            True,
        ),
        ("", WHITE, False),
        ("Blue = left_finger / CAD +X", BLUE, True),
        ("Orange = right_finger / CAD -X", ORANGE, True),
        ("Green = 20 g bottle proxy", GREEN, True),
        (
            "Magenta dot = projected physical contact"
            if projection
            else "Magenta dot = CAD-derived inward sample",
            MAGENTA,
            True,
        ),
        (
            "Colored arrow = projected Contact Report normal"
            if projection
            else "not a physical contact point",
            MAGENTA,
            False,
        ),
        ("", WHITE, False),
        (
            f"phase={phase}  frame={simulation['frame']}",
            WHITE,
            True,
        ),
        (f"time={simulation['time_s']:.6f} s", WHITE, False),
        (
            f"bottle z={float(bottle['z_m']):.9f} m",
            WHITE,
            False,
        ),
        (
            f"API vertical velocity={float(bottle['vertical_velocity_m_s']):+.6f} m/s",
            WHITE,
            False,
        ),
        (
            pose_velocity_text,
            WHITE,
            False,
        ),
        (
            f"angular speed={float(bottle['angular_speed_rad_s']):.6f} rad/s",
            WHITE,
            False,
        ),
        (
            f"max drop={drop_m * 1e3:.6f} mm / gate=10.000000 mm",
            GREEN if phase == "hold_end" else WHITE,
            phase == "hold_end",
        ),
        (
            "contact L/R="
            f"{simulation['contact_state'].get('left', False)}/"
            f"{simulation['contact_state'].get('right', False)}",
            WHITE,
            False,
        ),
    ]
    lines.extend(contact_lines)
    lines.extend(
        [
            ("", WHITE, False),
            ("Isaac Sim 5.1.0.0 / Kit 107.3.3", WHITE, False),
            ("PhysX 107.3.26 / 60 Hz", WHITE, False),
            ("collider=supplier CAD v2 convexHull", WHITE, False),
            ("friction=0.7 TEMPORARY_UNCALIBRATED", ORANGE, False),
            (
                "solve_articulation_contact_last=True",
                WHITE,
                False,
            ),
            (
                "capture paused: zero physics steps; state unchanged",
                MUTED,
                False,
            ),
            ("view=fixed_tip_end_contact  1280x900", WHITE, False),
            ("", WHITE, False),
        ]
    )
    lines.extend(
        (value, MUTED, False)
        for value in _wrap(
            "Stage: ",
            simulation["stage_absolute_path"],
        )
    )
    lines.extend(
        (value, MUTED, False)
        for value in _wrap(
            "Stage SHA-256: ",
            simulation["stage_sha256"],
            width=54,
        )
    )
    lines.extend(
        [
            ("", WHITE, False),
            ("Evidence boundary:", WHITE, True),
            ("• screenshot is auxiliary evidence", MUTED, False),
            (
                "• machine runtime data are authoritative",
                MUTED,
                False,
            ),
            (
                "• hold result: 20/20 exact-signature repeat",
                MUTED,
                False,
            ),
        ]
    )
    _write_panel_lines(draw, x=raw.width + 18, lines=lines)

    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(
        destination,
        format="PNG",
        compress_level=9,
        optimize=False,
    )
    return {
        "capture_name": phase,
        "raw_absolute_path": str(raw_path),
        "raw_sha256": _sha256(raw_path),
        "raw_resolution": [raw.width, raw.height],
        "annotated_absolute_path": str(destination.resolve()),
        "annotated_sha256": _sha256(destination),
        "annotated_resolution": [canvas.width, canvas.height],
        "finger_bbox_xyxy": {
            side: list(box) for side, box in boxes.items()
        },
        "bottle_bbox_xyxy": list(bottle_box),
        "contact_projection": projection,
        "camera": capture["camera"],
        "simulation": simulation,
        "raw_visual_self_review": (
            "PASS_BY_VISION_MODEL_2026-07-29"
        ),
        "annotated_visual_self_review": "PENDING_VISUAL_MODEL_REVIEW",
        "retake_reason": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-report", type=Path, default=RAW_REPORT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    report_path = args.raw_report.resolve(strict=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["run_mode"] != "ACCEPTANCE" or report["status"] != "PASS":
        raise RuntimeError("annotation requires a PASS acceptance report")
    if report["summary"]["pass_count"] != 20:
        raise RuntimeError("annotation requires 20/20 hold trials")
    if report["screenshots"]["status"] != "PASS":
        raise RuntimeError("raw screenshot acquisition did not pass")
    captures = report["screenshots"]["captures"]
    if {item["capture_name"] for item in captures} != {
        "open",
        "bilateral_contact",
        "release",
        "hold_end",
    }:
        raise RuntimeError("required physical phases are incomplete")

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"annotation output already exists: {output_root}"
        )
    annotated_root = output_root / "screenshots_annotated"
    records = [
        _draw_annotations(
            capture,
            report,
            annotated_root
            / f"{capture['capture_name']}_annotated.png",
        )
        for capture in captures
    ]
    metadata = {
        "schema_version": 1,
        "status": "PARTIAL",
        "raw_report": {
            "absolute_path": str(report_path),
            "sha256": _sha256(report_path),
        },
        "records": records,
        "visual_model_review": "PENDING_VISUAL_MODEL_REVIEW",
        "acceptance_boundary": (
            "ANNOTATIONS ARE AUXILIARY; CONTACT, POSE, VELOCITY, DROP, "
            "PENETRATION AND DETERMINISM COME FROM THE RUNTIME REPORT"
        ),
    }
    metadata_path = output_root / "annotation_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"status={metadata['status']}")
    print(f"metadata={metadata_path}")
    print(f"annotated_root={annotated_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
