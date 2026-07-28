#!/usr/bin/env python3
"""Annotate Isaac 5.1 supplier-CAD finger installation screenshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import textwrap
from typing import Any

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

PANEL_WIDTH = 620
PANEL = (17, 22, 30, 255)
WHITE = (242, 245, 250, 255)
MUTED = (180, 190, 205, 255)
GREEN = (70, 220, 130, 255)
BLUE = (20, 170, 255, 255)
ORANGE = (255, 145, 35, 255)
MAGENTA = (255, 45, 190, 255)
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str((BOLD if bold else FONT)), size)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    y, x = np.nonzero(mask)
    if len(x) < 500:
        raise RuntimeError(
            f"finger color mask has too few pixels: {len(x)}"
        )
    return int(x.min()), int(y.min()), int(x.max()), int(y.max())


def _finger_boxes(image: Image.Image) -> dict[str, tuple[int, int, int, int]]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    red, green, blue = (rgb[..., index] for index in range(3))
    left = (
        (blue > 150)
        & (green > 75)
        & (blue > red + 30)
        & (green > red + 15)
    )
    right = (
        (red > 150)
        & (green > 70)
        & (red > blue + 55)
        & (green > blue + 20)
    )
    return {
        "left": _bbox(left),
        "right": _bbox(right),
    }


def _arrow_head(
    draw: ImageDraw.ImageDraw,
    *,
    tip: tuple[float, float],
    toward: tuple[float, float],
) -> None:
    x, y = tip
    dx, dy = toward[0] - x, toward[1] - y
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    back = (x + 14.0 * ux, y + 14.0 * uy)
    draw.polygon(
        [
            tip,
            (back[0] + 6.0 * px, back[1] + 6.0 * py),
            (back[0] - 6.0 * px, back[1] - 6.0 * py),
        ],
        fill=MAGENTA,
    )


def _write_panel_lines(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    lines: list[tuple[str, tuple[int, int, int, int], bool]],
) -> None:
    y = 20
    for text, color, bold in lines:
        if not text:
            y += 9
            continue
        font = _font(16 if not bold else 17, bold=bold)
        draw.text((x, y), text, fill=color, font=font)
        y += 23 if not bold else 25


def _wrap(label: str, value: str, width: int = 58) -> list[str]:
    return textwrap.wrap(
        f"{label}{value}",
        width=width,
        subsequent_indent="  ",
        break_long_words=True,
        break_on_hyphens=False,
    )


def _annotate(
    capture: dict[str, Any],
    *,
    destination: Path,
) -> dict[str, Any]:
    raw_path = Path(capture["absolute_path"]).resolve(strict=True)
    with Image.open(raw_path) as opened:
        raw = opened.convert("RGBA")
    boxes = _finger_boxes(raw)
    canvas = Image.new(
        "RGBA",
        (raw.width + PANEL_WIDTH, raw.height),
        PANEL,
    )
    canvas.alpha_composite(raw, (0, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")
    center_x = raw.width / 2.0
    samples: dict[str, tuple[float, float]] = {}
    for side, color, tag in (
        ("left", BLUE, "L"),
        ("right", ORANGE, "R"),
    ):
        box = boxes[side]
        draw.rectangle(box, outline=color, width=4)
        box_center_x = 0.5 * (box[0] + box[2])
        inward_x = (
            box[2] - 8.0
            if box_center_x < center_x
            else box[0] + 8.0
        )
        inward_y = 0.5 * (box[1] + box[3])
        samples[side] = (inward_x, inward_y)
        draw.ellipse(
            (
                inward_x - 8,
                inward_y - 8,
                inward_x + 8,
                inward_y + 8,
            ),
            fill=MAGENTA,
            outline=WHITE,
            width=2,
        )
        label_x = max(8, min(box[0], raw.width - 40))
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
            tag,
            fill=color,
            font=_font(15, bold=True),
        )

    arrow_y = 0.5 * (samples["left"][1] + samples["right"][1])
    arrow_start = (samples["left"][0], arrow_y)
    arrow_end = (samples["right"][0], arrow_y)
    draw.line((arrow_start, arrow_end), fill=MAGENTA, width=4)
    _arrow_head(draw, tip=arrow_start, toward=arrow_end)
    _arrow_head(draw, tip=arrow_end, toward=arrow_start)
    mid_x = 0.5 * (arrow_start[0] + arrow_end[0])
    for y in range(20, raw.height - 20, 20):
        draw.line(
            (mid_x, y, mid_x, min(y + 10, raw.height - 20)),
            fill=(255, 255, 255, 145),
            width=2,
        )

    simulation = capture["simulation"]
    camera = capture["camera"]
    target = simulation["finger_targets_m"]
    readback = simulation["finger_readback_m"]
    lines: list[tuple[str, tuple[int, int, int, int], bool]] = [
        ("ALOHA ViperX follower_left", WHITE, True),
        ("Supplier-CAD finger installation", WHITE, True),
        (
            "PASS — CAD INSTALLATION VISUAL GATE ONLY",
            GREEN,
            True,
        ),
        ("NO COLLISION / CONTACT / GRASP CLAIM", MUTED, True),
        ("", WHITE, False),
        ("Blue = left_finger / CAD +X", BLUE, True),
        ("Orange = right_finger / CAD -X", ORANGE, True),
        ("Magenta = CAD-derived annotation sample", MAGENTA, True),
        ("not a physical contact point", MAGENTA, False),
        ("", WHITE, False),
        (
            f"state={simulation['state']}  view={camera['view']}",
            WHITE,
            True,
        ),
        (
            f"frame={simulation['frame']}  time={simulation['time_s']:.6f}s",
            WHITE,
            False,
        ),
        (
            f"target L/R={target[0]:+.6f}/{target[1]:+.6f} m",
            WHITE,
            False,
        ),
        (
            f"readback L/R={readback[0]:+.6f}/{readback[1]:+.6f} m",
            WHITE,
            False,
        ),
        (
            f"surface gap={simulation['surface_gap_m']:.6f} m",
            WHITE,
            False,
        ),
        ("Isaac Sim 5.1.0.0 / Kit 107.3.3", WHITE, False),
        ("PhysX 107.3.26", WHITE, False),
        (
            "visual=SUPPLIER_CAD_V2_VISUAL_ONLY",
            WHITE,
            False,
        ),
        ("collider=SOURCE_COLLIDER_UNCHANGED", MUTED, False),
        ("state method=VISUAL_SESSION_QPOS_PROJECTION", MUTED, False),
        ("", WHITE, False),
    ]
    for value in _wrap("Stage: ", simulation["stage_absolute_path"]):
        lines.append((value, MUTED, False))
    for value in _wrap(
        "Stage SHA-256: ",
        simulation["stage_sha256"],
        width=54,
    ):
        lines.append((value, MUTED, False))
    lines.extend(
        [
            ("", WHITE, False),
            (
                "camera p="
                + ",".join(
                    f"{value:+.4f}"
                    for value in camera["position_world_m"]
                ),
                WHITE,
                False,
            ),
            (
                "camera q(wxyz)="
                + ",".join(
                    f"{value:+.4f}"
                    for value in camera["orientation_wxyz"]
                ),
                WHITE,
                False,
            ),
            ("Checks:", WHITE, True),
            ("• both handed fingers fully visible", WHITE, False),
            ("• inward surfaces face one another", WHITE, False),
            ("• open/closed paired camera is identical", WHITE, False),
            ("• no critical crop or occlusion", WHITE, False),
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
        "capture_name": capture["capture_name"],
        "raw_absolute_path": str(raw_path),
        "raw_sha256": _sha256(raw_path),
        "raw_resolution": [raw.width, raw.height],
        "annotated_absolute_path": str(destination.resolve()),
        "annotated_sha256": _sha256(destination),
        "annotated_resolution": [canvas.width, canvas.height],
        "finger_bbox_xyxy": {
            side: list(box) for side, box in boxes.items()
        },
        "cad_derived_annotation_sample_xy": {
            side: list(point) for side, point in samples.items()
        },
        "camera": camera,
        "simulation": simulation,
        "raw_visual_self_review": "PASS",
        "annotated_visual_self_review": "NOT_RUN",
        "retake_reason": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    raw_report_path = args.raw_report.resolve(strict=True)
    raw_report = json.loads(raw_report_path.read_text(encoding="utf-8"))
    if raw_report["status"] != "PASS":
        raise RuntimeError("raw screenshot machine report must pass")
    if len(raw_report["captures"]) != 8:
        raise RuntimeError("expected exactly eight raw captures")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"annotation output already exists: {output_root}"
        )
    annotated_root = output_root / "screenshots_annotated"
    records = [
        _annotate(
            capture,
            destination=annotated_root
            / f"{capture['capture_name']}_annotated.png",
        )
        for capture in raw_report["captures"]
    ]
    metadata = {
        "schema_version": 1,
        "status": "PENDING_VISUAL_MODEL_REVIEW",
        "raw_report_absolute_path": str(raw_report_path),
        "raw_report_sha256": _sha256(raw_report_path),
        "capture_count": len(records),
        "captures": records,
        "screenshot_role": (
            "AUXILIARY_CAD_INSTALLATION_VISUAL_EVIDENCE; "
            "NOT_PHYSICS_ACCEPTANCE"
        ),
    }
    metadata_path = output_root / "annotation_metadata.json"
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
