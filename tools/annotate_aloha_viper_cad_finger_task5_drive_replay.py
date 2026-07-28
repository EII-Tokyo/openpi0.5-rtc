#!/usr/bin/env python3
"""Annotate the auxiliary Task 5 runtime-readback replay screenshots."""

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


def _draw_annotations(
    capture: dict[str, Any],
    destination: Path,
    *,
    base_drift_m: float,
    arm_drift: float,
    dynamic_drive_gate: str,
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

    centers = {
        side: (
            0.5 * (box[0] + box[2]),
            0.5 * (box[1] + box[3]),
        )
        for side, box in boxes.items()
    }
    image_left_side, image_right_side = sorted(
        centers, key=lambda side: centers[side][0]
    )
    inward_samples = {
        image_left_side: (
            float(boxes[image_left_side][2] - 7),
            centers[image_left_side][1],
        ),
        image_right_side: (
            float(boxes[image_right_side][0] + 7),
            centers[image_right_side][1],
        ),
    }
    for side, color, short_label in (
        ("left", BLUE, "L"),
        ("right", ORANGE, "R"),
    ):
        box = boxes[side]
        draw.rectangle(box, outline=color, width=4)
        point = inward_samples[side]
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
        label_x = box[0] + 8 if side == image_left_side else box[2] - 40
        label_y = max(8, box[1] + 8)
        draw.rounded_rectangle(
            (label_x, label_y, label_x + 32, label_y + 24),
            radius=5,
            fill=(12, 17, 24, 225),
            outline=color,
            width=2,
        )
        draw.text(
            (label_x + 9, label_y + 2),
            short_label,
            fill=color,
            font=_font(15, bold=True),
        )

    lower = max(box[3] for box in boxes.values())
    upper = min(box[1] for box in boxes.values())
    arrow_y = float(lower + 35 if lower + 50 < raw.height else upper - 35)
    arrow_start = (inward_samples[image_left_side][0], arrow_y)
    arrow_end = (inward_samples[image_right_side][0], arrow_y)
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
    numeric_gate_label = (
        "DYNAMIC NUMERIC GATE = PASS"
        if dynamic_drive_gate == "PASS_NUMERIC_ONLY"
        else "DYNAMIC DRIVE GATE = FAIL"
    )
    numeric_gate_color = (
        GREEN if dynamic_drive_gate == "PASS_NUMERIC_ONLY" else ORANGE
    )
    camera_position = camera.get(
        "actual_position_world_m",
        camera["position_world_m"],
    )
    camera_orientation = camera.get(
        "actual_orientation_wxyz",
        camera["orientation_wxyz"],
    )
    lines: list[tuple[str, tuple[int, int, int, int], bool]] = [
        ("ALOHA ViperX follower_left", WHITE, True),
        ("Task 5 — isolated dynamic diagnostic", WHITE, True),
        (numeric_gate_label, numeric_gate_color, True),
        ("RUNTIME READBACK REPLAY — AUXILIARY", GREEN, True),
        ("NO BOTTLE / CONTACT / GRASP CLAIM", MUTED, True),
        ("", WHITE, False),
        ("Blue = left_finger / CAD +X", BLUE, True),
        ("Orange = right_finger / CAD -X", ORANGE, True),
        ("Magenta = CAD-derived inward-surface sample", MAGENTA, True),
        ("not a physical contact point", MAGENTA, False),
        ("", WHITE, False),
        (
            f"source trajectory={simulation['source_trajectory']}",
            WHITE,
            True,
        ),
        (
            (
                "runtime frame="
                f"{simulation['source_runtime_frame']}  "
                f"time={simulation['source_runtime_time_s']:.6f}s"
            ),
            WHITE,
            False,
        ),
        (
            (
                "command L/R="
                f"{simulation['command_left_m']:+.6f}/"
                f"{simulation['command_right_m']:+.6f} m"
            ),
            WHITE,
            False,
        ),
        (
            (
                "readback L/R="
                f"{simulation['readback_left_m']:+.6f}/"
                f"{simulation['readback_right_m']:+.6f} m"
            ),
            WHITE,
            False,
        ),
        (f"base translation drift={base_drift_m:.9f} m", WHITE, False),
        (f"max arm DOF drift={arm_drift:.9f}", WHITE, False),
        ("Isaac Sim 5.1.0.0 / Kit 107.3.3", WHITE, False),
        ("PhysX 107.3.26", WHITE, False),
        ("view=base_oblique  resolution=1280x900", WHITE, False),
        ("capture=fresh reset; qpos replay; no physics step", MUTED, False),
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
                    for value in camera_position
                ),
                WHITE,
                False,
            ),
            (
                "camera q(wxyz)="
                + ",".join(
                    f"{value:+.4f}"
                    for value in camera_orientation
                ),
                WHITE,
                False,
            ),
            ("Visual checks:", WHITE, True),
            ("• both handed fingers fully visible", WHITE, False),
            ("• inward surfaces face one another", WHITE, False),
            ("• frame endpoints are visually distinct", WHITE, False),
            ("• annotations do not cover key geometry", WHITE, False),
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
            side: list(point)
            for side, point in inward_samples.items()
        },
        "camera": camera,
        "simulation": simulation,
        "raw_visual_self_review": "NOT_RUN",
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
    if raw_report["status"] != "PARTIAL":
        raise RuntimeError("expected auxiliary raw replay status PARTIAL")
    dynamic_drive_gate = raw_report["scope"]["dynamic_drive_gate"]
    if dynamic_drive_gate not in {"FAIL", "PASS_NUMERIC_ONLY"}:
        raise RuntimeError("unexpected dynamic drive gate")
    if raw_report["screenshot_manifest"]["status"] != "PASS":
        raise RuntimeError("raw screenshot acquisition failed")
    expected_capture_count = (
        3 if dynamic_drive_gate == "PASS_NUMERIC_ONLY" else 2
    )
    if len(raw_report["captures"]) != expected_capture_count:
        raise RuntimeError(
            f"expected exactly {expected_capture_count} raw replay captures"
        )

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"annotation output already exists: {output_root}"
        )
    annotated_root = output_root / "screenshots_annotated"
    if "trajectory_summary" in raw_report:
        summary = raw_report["trajectory_summary"]
    else:
        numeric_path = Path(
            raw_report["captures"][0]["simulation"][
                "source_numeric_report"
            ]
        ).resolve(strict=True)
        numeric = json.loads(numeric_path.read_text(encoding="utf-8"))
        summary = next(
            item
            for item in numeric["trajectories"]
            if item["name"] == "symmetric_close"
        )
    records = [
        _draw_annotations(
            capture,
            annotated_root
            / f"{capture['capture_name']}_annotated.png",
            base_drift_m=float(summary["base_translation_drift_m"]),
            arm_drift=float(summary["maximum_arm_dof_drift"]),
            dynamic_drive_gate=dynamic_drive_gate,
        )
        for capture in raw_report["captures"]
    ]
    metadata = {
        "schema_version": 1,
        "status": "PENDING_VISUAL_MODEL_REVIEW",
        "gate": "RUNTIME_READBACK_REPLAY_AUXILIARY",
        "raw_report_absolute_path": str(raw_report_path),
        "raw_report_sha256": _sha256(raw_report_path),
        "capture_count": len(records),
        "captures": records,
        "dynamic_drive_gate": dynamic_drive_gate,
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
