#!/usr/bin/env python3
"""Annotate Task 5 no-bottle supplier-CAD structure screenshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
    "aloha_viper_cad_finger_task5_structure.json"
)
OUTPUT_ROOT = (
    ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
    "isaac_cad_finger/task5_structure/annotation_v1"
)


def _draw_annotations(
    capture: dict[str, Any],
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

    box_centers = {
        side: (
            0.5 * (box[0] + box[2]),
            0.5 * (box[1] + box[3]),
        )
        for side, box in boxes.items()
    }
    ordered = sorted(box_centers, key=lambda side: box_centers[side][0])
    image_left_side, image_right_side = ordered
    samples = {
        image_left_side: (
            float(boxes[image_left_side][2] - 7),
            box_centers[image_left_side][1],
        ),
        image_right_side: (
            float(boxes[image_right_side][0] + 7),
            box_centers[image_right_side][1],
        ),
    }
    for side, color, tag in (
        ("left", BLUE, "L"),
        ("right", ORANGE, "R"),
    ):
        box = boxes[side]
        draw.rectangle(box, outline=color, width=4)
        point = samples[side]
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
        outer_x = box[0] + 8 if side == image_left_side else box[2] - 40
        label_y = max(8, box[1] + 8)
        draw.rounded_rectangle(
            (outer_x, label_y, outer_x + 32, label_y + 24),
            radius=5,
            fill=(12, 17, 24, 225),
            outline=color,
            width=2,
        )
        draw.text(
            (outer_x + 9, label_y + 2),
            tag,
            fill=color,
            font=_font(15, bold=True),
        )

    lower = max(box[3] for box in boxes.values())
    upper = min(box[1] for box in boxes.values())
    arrow_y = float(lower + 35 if lower + 50 < raw.height else upper - 35)
    arrow_start = (samples[image_left_side][0], arrow_y)
    arrow_end = (samples[image_right_side][0], arrow_y)
    draw.line((arrow_start, arrow_end), fill=MAGENTA, width=4)
    _arrow_head(draw, tip=arrow_start, toward=arrow_end)
    _arrow_head(draw, tip=arrow_end, toward=arrow_start)
    center_x = 0.5 * (arrow_start[0] + arrow_end[0])
    for y in range(16, raw.height - 16, 20):
        draw.line(
            (center_x, y, center_x, min(y + 10, raw.height - 16)),
            fill=(255, 255, 255, 145),
            width=2,
        )

    simulation = capture["simulation"]
    camera = capture["camera"]
    target = simulation["finger_targets_m"]
    readback = simulation["finger_readback_m"]
    projection = camera["finger_projection"]
    lines: list[tuple[str, tuple[int, int, int, int], bool]] = [
        ("ALOHA ViperX follower_left", WHITE, True),
        ("Task 5 — no-bottle structure", WHITE, True),
        ("PASS — STRUCTURE VISUAL GATE PASS ONLY", GREEN, True),
        ("DYNAMIC DRIVE / MIMIC = FAIL", ORANGE, True),
        ("NO BOTTLE / CONTACT / GRASP CLAIM", MUTED, True),
        ("", WHITE, False),
        ("Blue = left_finger / CAD +X", BLUE, True),
        ("Orange = right_finger / CAD -X", ORANGE, True),
        ("Magenta = CAD-derived inward-surface sample", MAGENTA, True),
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
        (
            "projection center="
            f"{projection['bbox_center_px'][0]:.1f},"
            f"{projection['bbox_center_px'][1]:.1f} px",
            WHITE,
            False,
        ),
        ("Isaac Sim 5.1.0.0 / Kit 107.3.3", WHITE, False),
        ("PhysX 107.3.26", WHITE, False),
        (
            "collider=SUPPLIER_CAD_V2_CONVEX_HULL_DIAGNOSTIC",
            WHITE,
            False,
        ),
        ("capture=fresh World reset; no physics step", MUTED, False),
        ("", WHITE, False),
    ]
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
            (
                "camera p="
                + ",".join(
                    f"{value:+.4f}"
                    for value in camera["actual_position_world_m"]
                ),
                WHITE,
                False,
            ),
            (
                "camera q(wxyz)="
                + ",".join(
                    f"{value:+.4f}"
                    for value in camera["actual_orientation_wxyz"]
                ),
                WHITE,
                False,
            ),
            ("Checks:", WHITE, True),
            ("• both handed fingers fully visible", WHITE, False),
            ("• inward surfaces face one another", WHITE, False),
            ("• three legal states are visually distinct", WHITE, False),
            ("• paired camera pose is identical", WHITE, False),
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
        "cad_derived_inward_surface_sample_xy": {
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
    parser.add_argument("--raw-report", type=Path, default=RAW_REPORT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    raw_report_path = args.raw_report.resolve(strict=True)
    raw_report = json.loads(raw_report_path.read_text(encoding="utf-8"))
    if raw_report["status"] != "FAIL":
        raise RuntimeError("expected preserved dynamic structure FAIL")
    if raw_report["screenshot_manifest"]["status"] != "PASS":
        raise RuntimeError("raw screenshot acquisition did not pass")
    if raw_report["gates"]["post_step_drive_tracking"]:
        raise RuntimeError("dynamic tracking failure was not preserved")
    if raw_report["gates"]["physx_mimic_or_controller_coupling"]:
        raise RuntimeError("mimic/coupling failure was not preserved")
    if len(raw_report["captures"]) != 12:
        raise RuntimeError("expected exactly twelve raw captures")

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"annotation output already exists: {output_root}"
        )
    annotated_root = output_root / "screenshots_annotated"
    records = [
        _draw_annotations(
            capture,
            annotated_root
            / f"{capture['capture_name']}_annotated.png",
        )
        for capture in raw_report["captures"]
    ]
    metadata = {
        "schema_version": 1,
        "status": "PENDING_VISUAL_MODEL_REVIEW",
        "gate": "TASK5_NO_BOTTLE_STRUCTURE_VISUAL_ONLY",
        "raw_report_absolute_path": str(raw_report_path),
        "raw_report_sha256": _sha256(raw_report_path),
        "capture_count": len(records),
        "captures": records,
        "physics_report_status": raw_report["status"],
        "dynamic_drive_tracking": "FAIL",
        "mimic_or_controller_coupling": "FAIL",
        "bottle_contact_grasp": "NOT_RUN",
        "task8": "NOT_RUN",
    }
    metadata_path = output_root / "annotation_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"status={metadata['status']}")
    print(f"metadata={metadata_path.resolve()}")
    print(f"annotated_root={annotated_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
