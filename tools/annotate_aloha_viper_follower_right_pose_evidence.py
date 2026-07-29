#!/usr/bin/env python3
"""Annotate visually reviewed follower_right robot-local screenshots."""

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
    "aloha_viper_follower_right_pose_screenshots_raw.json"
)
ARTIFACT_ROOT = (
    ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
    "follower_right_pose_evidence/attempt4_final"
)
DECISIONS = ARTIFACT_ROOT / "raw_visual_review_decisions.json"
OUTPUT_ROOT = ARTIFACT_ROOT / "screenshots_annotated_v2"
OUTPUT_METADATA = ARTIFACT_ROOT / "annotation_metadata_v2.json"


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
    *,
    outer_side: str,
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
    label_top = max(0, xy[1])
    if outer_side == "left":
        label_left = max(0, xy[0] - 35)
    elif outer_side == "right":
        label_left = min(1248, xy[2] + 4)
    else:
        raise ValueError(f"unsupported label side: {outer_side}")
    draw.rounded_rectangle(
        (label_left, label_top, label_left + 31, label_top + 30),
        radius=5,
        fill=color,
    )
    draw.text(
        (label_left + 8, label_top + 3),
        label,
        font=_font(19, bold=True),
        fill=(255, 255, 255),
    )


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    color = (50, 220, 230)
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


def _panel(canvas: Image.Image, record: dict[str, Any]) -> None:
    draw = ImageDraw.Draw(canvas)
    simulation = record["simulation"]
    camera = record["camera"]
    x = 1312
    y = 24
    draw.text(
        (x, y),
        "FOLLOWER_RIGHT POSE EVIDENCE",
        font=_font(25, bold=True),
        fill=(120, 240, 150),
    )
    y += 42
    numeric = simulation["numeric_status"]
    target = simulation["target"]
    readback = simulation["readback"]
    if isinstance(target, list) and len(target) == 9:
        target_text = "9-DOF home vector (machine report)"
        readback_text = "9-DOF home vector (machine report)"
    elif isinstance(target, list):
        target_text = "[" + ", ".join(
            f"{float(value):+.6f}" for value in target
        ) + "]"
        readback_text = "[" + ", ".join(
            f"{float(value):+.6f}" for value in readback
        ) + "]"
    else:
        target_text = f"{float(target):+.9f}"
        readback_text = f"{float(readback):+.9f}"
    position = camera["position_world_m"]
    orientation = camera["orientation_wxyz"]
    stage = Path(simulation["stage_absolute_path"])
    stage_relative = stage.relative_to(ROOT)
    lines = [
        "VISUAL INSTALLATION/POSE GATE: PASS",
        f"Numeric runtime gate: {numeric}",
        "Robot: follower_right",
        "Scope: ROBOT_LOCAL_DIAGNOSTIC",
        "This is not a workcell placement.",
        f"View: {camera['view']}",
        f"Phase: {simulation['phase']}",
        (
            f"Frame/time: {simulation['frame']} / "
            f"{simulation['time_s']:.3f} s"
        ),
        "Isaac Sim 5.1.0.0 / Kit 107.3.3",
        "PhysX 107.3.26",
        "",
        "Blue = left_finger / CAD +X",
        "Orange = right_finger / CAD -X",
        "Supplier embedded handed v2 pair",
        "L/R boxes = projected supplier meshes",
        "Cyan arrows = inward closing direction",
        "",
        f"Joint: {simulation['joint_name']}",
        f"Index: {simulation['joint_index']}",
        f"Target: {target_text}",
        f"Readback: {readback_text}",
    ]
    if simulation.get("mimic_residual_m") is not None:
        lines.extend(
            [
                f"Aperture: {simulation['aperture_m']:.9f} m",
                (
                    "Mimic residual: "
                    f"{simulation['mimic_residual_m']:.9f} m"
                ),
                "Mimic accuracy: "
                + (
                    "PASS"
                    if simulation["mimic_residual_m"] <= 0.001
                    else "FAIL"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "Camera position [m]: "
             "[" + ", ".join(f"{float(v):+.3f}" for v in position) + "]",
            "Camera orientation wxyz: "
             "[" + ", ".join(
                f"{float(v):+.3f}" for v in orientation
            ) + "]",
            "",
            "Stage absolute path:",
            str(ROOT) + "/",
            str(stage_relative.parent) + "/",
            stage_relative.name,
            f"Stage SHA: {simulation['stage_sha256'][:16]}...",
            "PASS above means visual pose gate only.",
            "Numeric report remains authoritative.",
            "Task 8: NOT_RUN",
        ]
    )
    for line in lines:
        color = (235, 235, 240)
        if line.startswith("VISUAL"):
            color = (120, 240, 150)
        elif "FAIL" in line:
            color = (255, 120, 100)
        elif line.startswith("This is not"):
            color = (255, 205, 90)
        draw.text((x, y), line, font=_font(15), fill=color)
        y += 22


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-report", type=Path, default=RAW_REPORT)
    parser.add_argument("--decisions", type=Path, default=DECISIONS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--metadata", type=Path, default=OUTPUT_METADATA)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_path = args.raw_report.resolve(strict=True)
    decisions_path = args.decisions.resolve(strict=True)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    if raw["capture_status"] != "PASS":
        raise RuntimeError("raw screenshot acquisition is not PASS")
    if decisions["status"] != "PASS":
        raise RuntimeError("raw visual-model review is not PASS")
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
            robot = projections["robot"]
            draw.rectangle(
                (
                    int(robot["bbox_min_px"][0]),
                    int(robot["bbox_min_px"][1]),
                    int(robot["bbox_max_px"][0]),
                    int(robot["bbox_max_px"][1]),
                ),
                outline=(80, 220, 130),
                width=3,
            )
        left = projections["left_finger"]["bbox_center_px"]
        right = projections["right_finger"]["bbox_center_px"]
        midpoint_x = int((left[0] + right[0]) / 2.0)
        _box(
            draw,
            projections["left_finger"],
            (50, 125, 255),
            "L",
            outer_side="left" if left[0] < midpoint_x else "right",
        )
        _box(
            draw,
            projections["right_finger"],
            (238, 145, 32),
            "R",
            outer_side="left" if right[0] < midpoint_x else "right",
        )
        geometry_y = int(
            min(
                projections["left_finger"]["bbox_min_px"][1],
                projections["right_finger"]["bbox_min_px"][1],
            )
        )
        arrow_y = max(38, geometry_y - 38)
        _arrow(
            draw,
            (int(left[0]), arrow_y),
            (midpoint_x - 8, arrow_y),
        )
        _arrow(
            draw,
            (int(right[0]), arrow_y),
            (midpoint_x + 8, arrow_y),
        )
        _panel(canvas, raw_record)
        destination = output_root / f"{name}_annotated.png"
        canvas.save(destination)
        records.append(
            {
                "capture_name": name,
                "phase": raw_record["simulation"]["phase"],
                "raw_absolute_path": str(path),
                "raw_sha256": raw_record["file_sha256"],
                "annotated_absolute_path": str(destination),
                "annotated_sha256": _sha256(destination),
                "annotated_resolution": [1920, 900],
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
        "raw_report": str(raw_path),
        "raw_report_sha256": _sha256(raw_path),
        "decisions": str(decisions_path),
        "decisions_sha256": _sha256(decisions_path),
    }
    metadata_path = args.metadata.resolve()
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"status={metadata['status']}")
    print(f"capture_count={len(records)}")
    print(f"metadata={metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
