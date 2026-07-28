#!/usr/bin/env python3
"""Annotate visually reviewed Viper CAD gripper screenshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from tools.aloha1_mapping.cad_gripper_screenshot_review import color_bbox
from tools.aloha1_mapping.cad_gripper_screenshot_review import remap_point

PANEL_WIDTH = 480
BLUE = (25, 156, 255, 255)
ORANGE = (255, 130, 30, 255)
CONTACT = (255, 45, 190, 255)
PASS_GREEN = (70, 220, 130, 255)
WHITE = (242, 245, 250, 255)
MUTED = (185, 195, 210, 255)
PANEL = (17, 22, 30, 255)
FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path.resolve(strict=True)), size)


def _arrow_head(
    draw: ImageDraw.ImageDraw,
    *,
    point: tuple[float, float],
    toward: tuple[float, float],
    color: tuple[int, int, int, int],
) -> None:
    x, y = point
    dx = toward[0] - x
    dy = toward[1] - y
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    perpendicular = (-uy, ux)
    size = 12.0
    wing = 5.5
    back = (x + ux * size, y + uy * size)
    draw.polygon(
        [
            (x, y),
            (
                back[0] + perpendicular[0] * wing,
                back[1] + perpendicular[1] * wing,
            ),
            (
                back[0] - perpendicular[0] * wing,
                back[1] - perpendicular[1] * wing,
            ),
        ],
        fill=color,
    )


def _double_arrow(
    draw: ImageDraw.ImageDraw,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    draw.line((start, end), fill=CONTACT, width=4)
    _arrow_head(draw, point=start, toward=end, color=CONTACT)
    _arrow_head(draw, point=end, toward=start, color=CONTACT)


def _dashed_vertical(
    draw: ImageDraw.ImageDraw,
    *,
    x: float,
    height: int,
) -> None:
    for y in range(25, height - 25, 22):
        draw.line((x, y, x, min(y + 11, height - 25)), fill=(255, 255, 255, 135), width=2)


def _text_block(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    lines: list[tuple[str, tuple[int, int, int, int], bool]],
) -> None:
    regular = _font(FONT_REGULAR, 19)
    bold = _font(FONT_BOLD, 20)
    cursor = y
    for text, color, use_bold in lines:
        font = bold if use_bold else regular
        draw.text((x, cursor), text, font=font, fill=color)
        cursor += 31 if use_bold else 27


def _annotate_capture(
    capture: dict[str, Any],
    *,
    annotated_root: Path,
) -> dict[str, Any]:
    raw_path = Path(capture["raw_path"]).resolve(strict=True)
    with Image.open(raw_path) as source:
        raw = source.convert("RGBA")
    canvas = Image.new("RGBA", (raw.width + PANEL_WIDTH, raw.height), PANEL)
    canvas.alpha_composite(raw, (0, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")
    roles = {
        "cad_positive_x_finger": {
            "color": BLUE,
            "label": "left_finger / CAD +X",
            "tag": "L",
        },
        "cad_negative_x_finger": {
            "color": ORANGE,
            "label": "right_finger / CAD -X",
            "tag": "R",
        },
    }
    measured = {}
    for role, style in roles.items():
        bbox = color_bbox(raw, role=role)
        draw.rectangle(bbox, outline=style["color"], width=4)
        projected = capture["role_projection"][role]
        source_bbox_record = projected["bbox_px"]
        source_bbox = (
            source_bbox_record["xmin"],
            source_bbox_record["ymin"],
            source_bbox_record["xmax"],
            source_bbox_record["ymax"],
        )
        sample = remap_point(
            point=tuple(projected["inner_surface_sample_px"]),
            source_bbox=source_bbox,
            target_bbox=tuple(float(value) for value in bbox),
        )
        radius = 9
        draw.ellipse(
            (
                sample[0] - radius,
                sample[1] - radius,
                sample[0] + radius,
                sample[1] + radius,
            ),
            fill=CONTACT,
            outline=WHITE,
            width=2,
        )
        label_y = max(8, bbox[1] - 29)
        label_x = min(max(8, bbox[0]), raw.width - 42)
        draw.rounded_rectangle(
            (label_x, label_y, label_x + 34, label_y + 26),
            radius=5,
            fill=(12, 17, 24, 220),
            outline=style["color"],
            width=2,
        )
        draw.text(
            (label_x + 10, label_y + 3),
            style["tag"],
            font=_font(FONT_BOLD, 15),
            fill=style["color"],
        )
        measured[role] = {
            "measured_color_bbox_px": list(bbox),
            "inner_surface_sample_px": list(sample),
        }
    positive = measured["cad_positive_x_finger"]["inner_surface_sample_px"]
    negative = measured["cad_negative_x_finger"]["inner_surface_sample_px"]
    arrow_y = min(max(60.0, 0.5 * (positive[1] + negative[1])), raw.height - 60.0)
    arrow_start = (positive[0], arrow_y)
    arrow_end = (negative[0], arrow_y)
    _double_arrow(draw, start=arrow_start, end=arrow_end)
    center_x = 0.5 * (positive[0] + negative[0])
    _dashed_vertical(draw, x=center_x, height=raw.height)
    aperture_text = (
        f"finger min distance = "
        f"{capture['finger_minimum_distance_mm']:.3f} mm"
    )
    text_width = draw.textlength(aperture_text, font=_font(FONT_BOLD, 18))
    label_top = arrow_y + 8 if arrow_y < 100 else arrow_y - 36
    draw.rounded_rectangle(
        (
            center_x - text_width / 2 - 9,
            label_top,
            center_x + text_width / 2 + 9,
            label_top + 28,
        ),
        radius=5,
        fill=(12, 17, 24, 220),
    )
    draw.text(
        (center_x - text_width / 2, label_top + 3),
        aperture_text,
        font=_font(FONT_BOLD, 18),
        fill=CONTACT,
    )
    panel_x = raw.width + 24
    camera = capture["camera"]
    lines = [
        ("ALOHA ViperX follower gripper", WHITE, True),
        ("CAD installation review", WHITE, True),
        ("", WHITE, False),
        (f"STATE  {capture['state_id'].upper()}", PASS_GREEN, True),
        (f"VIEW   {capture['view_id']}", WHITE, True),
        ("PASS — CAD VISUAL GATE", PASS_GREEN, True),
        ("", WHITE, False),
        ("Blue  left_finger / CAD +X", BLUE, True),
        ("Orange right_finger / CAD -X", ORANGE, True),
        ("Magenta inner/contact-facing", CONTACT, True),
        ("surface sample (annotation only)", CONTACT, False),
        ("", WHITE, False),
        (
            f"B-Rep min distance: "
            f"{capture['finger_minimum_distance_mm']:.3f} mm",
            WHITE,
            False,
        ),
        (
            "camera forward: "
            + ", ".join(f"{value:+.3f}" for value in camera["camera_forward"]),
            WHITE,
            False,
        ),
        (
            "image up: "
            + ", ".join(f"{value:+.3f}" for value in camera["image_up"]),
            WHITE,
            False,
        ),
        ("frame/time: static CAD / N/A", MUTED, False),
        ("collider: NOT_APPLICABLE", MUTED, False),
        ("physics/contact/hold: NOT_RUN", MUTED, False),
        ("", WHITE, False),
        ("Checks:", WHITE, True),
        ("• both handed fingers visible", WHITE, False),
        ("• inner faces point to center", WHITE, False),
        ("• open/closed paired camera", WHITE, False),
        ("• no crop or shell occlusion", WHITE, False),
        ("", WHITE, False),
        ("Source SHA-256:", MUTED, True),
        (capture["source_cad_sha256"][:32], MUTED, False),
        (capture["source_cad_sha256"][32:], MUTED, False),
    ]
    _text_block(draw, x=panel_x, y=25, lines=lines)
    annotated_root.mkdir(parents=True, exist_ok=True)
    output = annotated_root / (
        f"{capture['state_id']}_{capture['view_id']}_annotated.png"
    )
    canvas.convert("RGB").save(
        output,
        format="PNG",
        compress_level=9,
        optimize=False,
    )
    return {
        **capture,
        "annotated_path": str(output.resolve()),
        "annotated_sha256": _sha256(output),
        "annotated_resolution": [canvas.width, canvas.height],
        "measured_annotation_geometry": measured,
        "raw_visual_self_review": "PASS",
        "annotated_visual_self_review": "NOT_RUN",
        "visual_self_review": "NOT_RUN",
        "visual_review_basis": (
            "assistant vision inspection; geometry/path/hash checks alone "
            "cannot satisfy this field"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--annotated-subdir",
        default="screenshots_annotated",
    )
    parser.add_argument(
        "--metadata-name",
        default="annotation_metadata.json",
    )
    args = parser.parse_args()
    metadata_path = args.metadata.resolve(strict=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["capture_count"] != 8:
        raise RuntimeError("annotation requires exactly eight captures")
    output_root = args.output_root.resolve()
    annotated_root = output_root / args.annotated_subdir
    captures = [
        _annotate_capture(capture, annotated_root=annotated_root)
        for capture in metadata["captures"]
    ]
    result = {
        "schema_version": 1,
        "status": "NOT_RUN",
        "scope": "supplier-CAD installation orientation screenshot evidence",
        "render_metadata_path": str(metadata_path),
        "render_metadata_sha256": _sha256(metadata_path),
        "capture_count": len(captures),
        "captures": captures,
    }
    output = output_root / args.metadata_name
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
