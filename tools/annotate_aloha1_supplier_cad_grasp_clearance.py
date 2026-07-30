#!/usr/bin/env python3
"""Annotate static complete-gripper CAD clearance screenshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

PANEL_WIDTH = 660
COLORS = {
    "white": (244, 247, 251, 255),
    "muted": (174, 185, 201, 255),
    "red": (255, 70, 70, 255),
    "green": (70, 235, 125, 255),
    "blue": (60, 135, 255, 255),
    "orange": (255, 135, 35, 255),
    "cyan": (45, 230, 240, 255),
    "magenta": (238, 90, 245, 255),
    "yellow": (255, 220, 65, 255),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-metadata", type=Path, required=True)
    parser.add_argument("--clearance-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _font(size: int, *, bold: bool = False) -> Any:
    name = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    return ImageFont.truetype(name, size=size)


def _point(record: dict[str, Any], name: str) -> tuple[float, float]:
    value = record["projected_points_px"][name]
    return float(value[0]), float(value[1])


def _dot(
    left: list[float],
    right: list[float],
) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _project(
    point: list[float],
    camera: dict[str, Any],
    *,
    width: int,
    height: int,
) -> tuple[float, float]:
    relative = [
        point[index] - camera["target_m"][index]
        for index in range(3)
    ]
    horizontal = _dot(relative, camera["image_right_gripper"])
    vertical = _dot(relative, camera["image_up_gripper"])
    return (
        (0.5 + horizontal / camera["ortho_width_m"]) * width,
        (0.5 - vertical / camera["ortho_height_m"]) * height,
    )


def _marker(
    draw: ImageDraw.ImageDraw,
    point: tuple[float, float],
    *,
    color: tuple[int, int, int, int],
    label: str,
    offset: tuple[int, int],
) -> None:
    x, y = point
    radius = 8
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=color,
        outline=COLORS["white"],
        width=2,
    )
    text_position = (x + offset[0], y + offset[1])
    draw.line(
        (x, y, text_position[0], text_position[1] + 8),
        fill=color,
        width=3,
    )
    draw.text(
        text_position,
        label,
        font=_font(22, bold=True),
        fill=color,
        stroke_width=2,
        stroke_fill=(15, 18, 24, 255),
    )


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    *,
    x0: int,
    height: int,
    record: dict[str, Any],
    report: dict[str, Any],
) -> None:
    draw.rectangle(
        (x0, 0, x0 + PANEL_WIDTH, height),
        fill=(18, 22, 31, 255),
    )
    state = record["state"]
    state_status = "FAIL" if state == "rejected_run13" else "PASS"
    status_color = COLORS["red"] if state_status == "FAIL" else COLORS["green"]
    lines = [
        ("ALOHA1 COMPLETE GRIPPER CAD CLEARANCE", COLORS["white"], True),
        ("STATIC GEOMETRY EVIDENCE — NOT ISAAC HOLD", COLORS["yellow"], True),
        (f"State: {state}", COLORS["white"], True),
        (f"Geometry state: {state_status}", status_color, True),
        ("Visual evidence: pending model review", COLORS["yellow"], False),
        ("", COLORS["white"], False),
        (f"View: {record['view']}", COLORS["white"], True),
        ("Projection: ORTHOGRAPHIC", COLORS["white"], False),
        (
            "Blue = left_finger / CAD +X",
            COLORS["blue"],
            True,
        ),
        (
            "Orange = right_finger / CAD -X",
            COLORS["orange"],
            True,
        ),
        ("Cyan = project Bottle500", COLORS["cyan"], True),
        ("Red volume = runtime gripper-bar AABB", COLORS["red"], False),
        ("EE = official helper; not grasp center", COLORS["magenta"], False),
        ("GF/RF = corrected/rejected grasp frame", COLORS["green"], False),
        ("BC = bottle axis center; LR = contact projection", COLORS["cyan"], False),
        ("", COLORS["white"], False),
        (
            f"Bottle axis-center x: {record['bottle_center_x_m'] * 1000:.3f} mm",
            COLORS["white"],
            False,
        ),
        (
            f"left/right q: {record['left_finger_q_m'] * 1000:.3f} / "
            f"{record['right_finger_q_m'] * 1000:.3f} mm",
            COLORS["white"],
            False,
        ),
    ]
    if state == "corrected_cad":
        lines.extend(
            [
                (
                    "Pad-contact frame x: "
                    f"{report['grasp_frame']['origin_reference_m'][0] * 1000:.3f} mm",
                    COLORS["green"],
                    True,
                ),
                (
                    "Bottle center offset from pad frame: "
                    f"{report['grasp_frame']['bottle_axis_center_from_grasp_m'][0] * 1000:.3f} mm",
                    COLORS["green"],
                    False,
                ),
                (
                    "Max-min hard margin: "
                    f"{report['station_selection']['selected_minimum_margin_m'] * 1000:.3f} mm",
                    COLORS["green"],
                    True,
                ),
            ]
        )
    else:
        lines.extend(
            [
                (
                    "Old hard margin: "
                    f"{report['station_selection']['rejected_station']['hard_clearance_m'] * 1000:.3f} mm",
                    COLORS["red"],
                    True,
                ),
                (
                    "Run13: no bilateral physical finger contact",
                    COLORS["red"],
                    True,
                ),
            ]
        )
    lines.extend(
        [
            ("", COLORS["white"], False),
            ("Axes in gripper_link frame:", COLORS["white"], True),
            ("+X red = vertical approach / world -Z", COLORS["red"], False),
            ("+Y green = toward left_finger", COLORS["green"], False),
            ("+Z blue = directed bottle axis AB", COLORS["blue"], False),
            ("", COLORS["white"], False),
            ("EE helper is NOT the grasp center.", COLORS["magenta"], True),
            ("Task 8: NOT_RUN", COLORS["yellow"], True),
        ]
    )
    y = 28
    for text, color, bold in lines:
        if not text:
            y += 13
            continue
        font = _font(22 if bold else 20, bold=bold)
        draw.multiline_text(
            (x0 + 24, y),
            text,
            font=font,
            fill=color,
            spacing=5,
        )
        y += 35 if bold else 31


def main() -> int:
    args = _parse_args()
    metadata_path = args.render_metadata.resolve(strict=True)
    report_path = args.clearance_report.resolve(strict=True)
    output_root = args.output_root.resolve()
    annotated_root = output_root / "screenshots_annotated"
    annotated_root.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = []
    for capture in metadata["captures"]:
        raw_path = Path(capture["raw_absolute_path"]).resolve(strict=True)
        image = Image.open(raw_path).convert("RGBA")
        width, height = image.size
        canvas = Image.new(
            "RGBA",
            (width + PANEL_WIDTH, height),
            (18, 22, 31, 255),
        )
        canvas.alpha_composite(image, (0, 0))
        draw = ImageDraw.Draw(canvas)
        state = capture["state"]
        view = capture["view"]
        if view == "world_side":
            _marker(
                draw,
                _point(capture, "official_ee_helper"),
                color=COLORS["magenta"],
                label="EE",
                offset=(-55, -42),
            )
            _marker(
                draw,
                _point(capture, "bottle_axis_center"),
                color=COLORS["cyan"],
                label="BC",
                offset=(18, 30),
            )
            _marker(
                draw,
                _point(capture, "grasp_frame_origin"),
                color=(
                    COLORS["green"]
                    if state == "corrected_cad"
                    else COLORS["red"]
                ),
                label=(
                    "GF"
                    if state == "corrected_cad"
                    else "RF"
                ),
                offset=(18, -45),
            )
        axis_a = _point(capture, "bottle_axis_a")
        axis_b = _point(capture, "bottle_axis_b")
        draw.line((*axis_a, *axis_b), fill=COLORS["blue"], width=5)
        draw.text(
            axis_a,
            "A",
            font=_font(25, bold=True),
            fill=COLORS["blue"],
            stroke_width=2,
            stroke_fill=(15, 18, 24, 255),
        )
        draw.text(
            axis_b,
            "B",
            font=_font(25, bold=True),
            fill=COLORS["blue"],
            stroke_width=2,
            stroke_fill=(15, 18, 24, 255),
        )
        if (
            state == "corrected_cad"
            and view == "true_world_top"
            and "left_contact" in capture[
                "projected_points_px"
            ]
        ):
            left_contact = _point(capture, "left_contact")
            right_contact = _point(capture, "right_contact")
            draw.line(
                (*right_contact, *left_contact),
                fill=COLORS["yellow"],
                width=4,
            )
            _marker(
                draw,
                left_contact,
                color=COLORS["blue"],
                label="L",
                offset=(-45, -42),
            )
            _marker(
                draw,
                right_contact,
                color=COLORS["orange"],
                label="R",
                offset=(18, 18),
            )
        elif state == "corrected_cad" and "left_contact" in capture[
            "projected_points_px"
        ]:
            left_contact = _point(capture, "left_contact")
            right_contact = _point(capture, "right_contact")
            projected_contact = (
                (left_contact[0] + right_contact[0]) / 2.0,
                (left_contact[1] + right_contact[1]) / 2.0,
            )
            _marker(
                draw,
                projected_contact,
                color=COLORS["yellow"],
                label="LR",
                offset=(-48, 24),
            )
        camera = capture["camera"]
        axis_origin = [0.0, 0.0, 0.0]
        for name, end, color in (
            ("+X", [0.055, 0.0, 0.0], COLORS["red"]),
            ("+Y", [0.0, 0.055, 0.0], COLORS["green"]),
            ("+Z", [0.0, 0.0, 0.055], COLORS["blue"]),
        ):
            p0 = _project(axis_origin, camera, width=width, height=height)
            p1 = _project(end, camera, width=width, height=height)
            draw.line((*p0, *p1), fill=color, width=5)
            draw.text(
                p1,
                name,
                font=_font(22, bold=True),
                fill=color,
                stroke_width=2,
                stroke_fill=(15, 18, 24, 255),
            )
        draw.rectangle(
            (0, 0, width - 1, height - 1),
            outline=(
                COLORS["green"]
                if state == "corrected_cad"
                else COLORS["red"]
            ),
            width=8,
        )
        _draw_panel(
            draw,
            x0=width,
            height=height,
            record=capture,
            report=report,
        )
        annotated_path = (
            annotated_root / f"{state}_{view}_annotated.png"
        )
        canvas.save(annotated_path)
        records.append(
            {
                **capture,
                "annotated_absolute_path": str(annotated_path),
                "annotated_sha256": _sha256(annotated_path),
                "raw_sha256": _sha256(raw_path),
                "annotated_width_px": canvas.width,
                "annotated_height_px": canvas.height,
                "annotation_status": "PASS",
                "visual_review": "PENDING",
            }
        )
    output = {
        "schema_version": 1,
        "status": "PARTIAL",
        "classification": (
            "ANNOTATED_PENDING_INDIVIDUAL_VISION_MODEL_REVIEW"
        ),
        "render_metadata": {
            "absolute_path": str(metadata_path),
            "sha256": _sha256(metadata_path),
        },
        "clearance_report": {
            "absolute_path": str(report_path),
            "sha256": _sha256(report_path),
        },
        "captures": records,
        "task8": "NOT_RUN",
    }
    output_path = output_root / "annotation_metadata.json"
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PARTIAL",
                "capture_count": len(records),
                "output": str(output_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
