#!/usr/bin/env python3
"""Annotate individually reviewed Task 7 pose screenshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[1]
RAW_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task7_pose_screenshots_raw.json"
)
DECISIONS = (
    ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
    "task7_robot_scope/pose_evidence_attempt5/"
    "raw_visual_review_decisions.json"
)
OUTPUT_ROOT = (
    ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
    "task7_robot_scope/pose_evidence_attempt5/screenshots_annotated_v2"
)
OUTPUT_METADATA = (
    ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
    "task7_robot_scope/pose_evidence_attempt5/annotation_metadata_v2.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(
        f"/usr/share/fonts/truetype/dejavu/{name}",
        size,
    )


def _box(
    draw: ImageDraw.ImageDraw,
    projection: dict[str, Any],
    color: tuple[int, int, int],
    label: str,
) -> None:
    minimum = projection["bbox_min_px"]
    maximum = projection["bbox_max_px"]
    xy = (
        int(minimum[0]),
        int(minimum[1]),
        int(maximum[0]),
        int(maximum[1]),
    )
    draw.rectangle(xy, outline=color, width=4)
    draw.rounded_rectangle(
        (xy[0], max(0, xy[1] - 32), xy[0] + 32, xy[1]),
        radius=5,
        fill=color,
    )
    draw.text(
        (xy[0] + 8, max(1, xy[1] - 29)),
        label,
        font=_font(20, bold=True),
        fill=(255, 255, 255),
    )


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    draw.line((start, end), fill=color, width=4)
    direction = 1 if end[0] >= start[0] else -1
    draw.polygon(
        [
            end,
            (end[0] - direction * 14, end[1] - 8),
            (end[0] - direction * 14, end[1] + 8),
        ],
        fill=color,
    )


def _write_panel(
    canvas: Image.Image,
    record: dict[str, Any],
) -> None:
    draw = ImageDraw.Draw(canvas)
    simulation = record["simulation"]
    camera = record["camera"]
    x = 1312
    y = 26
    draw.text(
        (x, y),
        "TASK 7 POSE EVIDENCE",
        font=_font(28, bold=True),
        fill=(120, 240, 150),
    )
    y += 48
    lines = [
        "VISUAL SELF-REVIEW: PASS",
        "Robot: follower_left",
        f"View: {camera['view']}",
        f"Phase: {simulation['phase']}",
        (
            "Frame/time: "
            f"{simulation['source_runtime_frame']} / "
            f"{simulation['source_runtime_time_s']:.6f} s"
        ),
        "Isaac Sim 5.1.0.0 / Kit 107.3.3",
        "PhysX 107.3.26",
        "",
        "Blue = left_finger / CAD +X",
        "Orange = right_finger / CAD -X",
        "L/R boxes: projected CAD finger meshes",
        "Green box: complete follower_left visual",
        "Cyan arrows: finger closing direction",
        "",
        (
            "left readback = "
            f"{simulation['readback_left_m']:+.9f} m"
        ),
        (
            "right readback = "
            f"{simulation['readback_right_m']:+.9f} m"
        ),
        "Trajectory: certified symmetric_close",
        f"Stage SHA: {simulation['stage_sha256'][:16]}...",
        "",
        "PASS means pose/direction visual gate only.",
        "It is not collision/contact/hold acceptance.",
        "Runtime numeric report is authoritative.",
        "",
        "follower_right: NOT_RUN",
        "Approved Stage contains follower_left only.",
        "No duplicated or synthetic right arm.",
        "",
        "Task 8: NOT_RUN",
    ]
    for line in lines:
        color = (235, 235, 240)
        if line.startswith("VISUAL"):
            color = (120, 240, 150)
        elif line.startswith("follower_right"):
            color = (255, 205, 90)
        draw.text((x, y), line, font=_font(18), fill=color)
        y += 29


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-report", type=Path, default=RAW_REPORT)
    parser.add_argument("--decisions", type=Path, default=DECISIONS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--metadata", type=Path, default=OUTPUT_METADATA)
    args = parser.parse_args()

    raw = json.loads(
        args.raw_report.resolve(strict=True).read_text(encoding="utf-8")
    )
    decisions = json.loads(
        args.decisions.resolve(strict=True).read_text(encoding="utf-8")
    )
    if raw["capture_status"] != "PASS":
        raise RuntimeError("raw screenshot acquisition is not PASS")
    if decisions["status"] != "PASS":
        raise RuntimeError("raw visual review is not PASS")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"annotation output exists: {output_root}")
    output_root.mkdir(parents=True)

    records = []
    for raw_record in raw["captures"]:
        name = raw_record["capture_name"]
        decision = decisions["records"][name]
        if decision["status"] != "PASS":
            raise RuntimeError(f"raw visual review failed: {name}")
        path = Path(raw_record["absolute_path"]).resolve(strict=True)
        if _sha256(path) != raw_record["file_sha256"]:
            raise RuntimeError(f"raw image hash drift: {path}")
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        canvas = Image.new("RGB", (1920, 900), (23, 26, 34))
        canvas.paste(image, (0, 0))
        draw = ImageDraw.Draw(canvas)
        projections = raw_record["camera"]["projections"]
        if raw_record["camera"]["view"] == "full_arm_oblique":
            robot_box = projections["robot"]
            draw.rectangle(
                (
                    int(robot_box["bbox_min_px"][0]),
                    int(robot_box["bbox_min_px"][1]),
                    int(robot_box["bbox_max_px"][0]),
                    int(robot_box["bbox_max_px"][1]),
                ),
                outline=(80, 220, 130),
                width=3,
            )
        _box(draw, projections["left_finger"], (50, 125, 255), "L")
        _box(draw, projections["right_finger"], (238, 145, 32), "R")
        left_center = projections["left_finger"]["bbox_center_px"]
        right_center = projections["right_finger"]["bbox_center_px"]
        midpoint_x = int((left_center[0] + right_center[0]) / 2.0)
        geometry_y_min = int(
            min(
                projections["left_finger"]["bbox_min_px"][1],
                projections["right_finger"]["bbox_min_px"][1],
            )
        )
        geometry_y_max = int(
            max(
                projections["left_finger"]["bbox_max_px"][1],
                projections["right_finger"]["bbox_max_px"][1],
            )
        )
        arrow_y = max(38, geometry_y_min - 38)
        _arrow(
            draw,
            (int(left_center[0]), arrow_y),
            (midpoint_x - 8, arrow_y),
            (50, 220, 230),
        )
        _arrow(
            draw,
            (int(right_center[0]), arrow_y),
            (midpoint_x + 8, arrow_y),
            (50, 220, 230),
        )
        if geometry_y_min > 72:
            draw.line(
                ((midpoint_x, 60), (midpoint_x, geometry_y_min - 12)),
                fill=(180, 180, 185),
                width=2,
            )
        if geometry_y_max < 828:
            draw.line(
                (
                    (midpoint_x, geometry_y_max + 12),
                    (midpoint_x, 840),
                ),
                fill=(180, 180, 185),
                width=2,
            )
        _write_panel(canvas, raw_record)
        destination = output_root / name.replace(
            "_raw", ""
        )
        destination = destination.with_name(
            f"{name}_annotated.png"
        )
        canvas.save(destination)
        records.append(
            {
                "capture_name": name,
                "annotated_absolute_path": str(destination),
                "annotated_sha256": _sha256(destination),
                "annotated_resolution": [1920, 900],
                "raw_absolute_path": str(path),
                "raw_sha256": raw_record["file_sha256"],
                "camera": raw_record["camera"],
                "simulation": raw_record["simulation"],
                "raw_visual_review": decision,
                "annotated_visual_review": "PENDING",
            }
        )
    metadata = {
        "schema_version": 1,
        "status": "PENDING_ANNOTATED_VISUAL_MODEL_REVIEW",
        "records": records,
        "raw_report": str(args.raw_report.resolve()),
        "raw_report_sha256": _sha256(args.raw_report.resolve()),
        "decisions": str(args.decisions.resolve()),
        "decisions_sha256": _sha256(args.decisions.resolve()),
    }
    args.metadata.resolve().write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"status={metadata['status']}")
    print(f"capture_count={len(records)}")
    print(f"metadata={args.metadata.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
