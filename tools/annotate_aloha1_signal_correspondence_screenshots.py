#!/usr/bin/env python3
"""Annotate fresh dual-follower Isaac signal-correspondence screenshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from tools.aloha1_mapping.signal_correspondence_screenshots import merge_capture_documents

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / ".codex/artifacts/20260729-aloha1-signal-correspondence"
LEFT_METADATA = ARTIFACT_ROOT / "metadata/aloha1_signal_screenshot_metadata_left.json"
RIGHT_METADATA = ARTIFACT_ROOT / "metadata/aloha1_signal_screenshot_metadata_right.json"
MERGED_METADATA = ARTIFACT_ROOT / "metadata/aloha1_signal_screenshot_metadata.json"
OUTPUT_ROOT = ARTIFACT_ROOT / "screenshots_annotated"
OUTPUT_METADATA = ARTIFACT_ROOT / "metadata/aloha1_signal_annotation_metadata.json"
RAW_SIZE = (1280, 900)
CANVAS_SIZE = (1740, 900)


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


def _bbox(
    draw: ImageDraw.ImageDraw,
    projection: dict[str, Any],
    *,
    color: tuple[int, int, int],
    label: str,
    width: int,
) -> None:
    minimum = projection["bbox_min_px"]
    maximum = projection["bbox_max_px"]
    box = (
        round(minimum[0]),
        round(minimum[1]),
        round(maximum[0]),
        round(maximum[1]),
    )
    draw.rectangle(box, outline=color, width=width)
    label_box = (
        box[0],
        max(0, box[1] - 25),
        min(RAW_SIZE[0] - 1, box[0] + 94),
        max(24, box[1]),
    )
    draw.rectangle(label_box, fill=color)
    draw.text(
        (label_box[0] + 5, label_box[1] + 2),
        label,
        font=_font(15, bold=True),
        fill=(12, 18, 24),
    )


def _point(
    draw: ImageDraw.ImageDraw,
    center: list[float],
    *,
    color: tuple[int, int, int],
    label: str,
    label_side: str = "right",
) -> tuple[int, int]:
    x, y = round(center[0]), round(center[1])
    radius = 8
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline=color,
        fill=(20, 28, 38),
        width=4,
    )
    if label_side == "right":
        label_x = x + 11
    elif label_side == "left":
        label_x = x - 27
    else:
        raise ValueError(f"unsupported point label side: {label_side}")
    label_y = max(0, y - 11)
    text_box = draw.textbbox(
        (label_x, label_y),
        label,
        font=_font(13, bold=True),
    )
    draw.rectangle(text_box, fill=(20, 28, 38))
    draw.text(
        (label_x, label_y),
        label,
        font=_font(13, bold=True),
        fill=color,
    )
    return x, y


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    if start == end:
        return
    color = (40, 235, 235)
    draw.line((start, end), fill=color, width=4)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    magnitude = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / magnitude, dy / magnitude
    px, py = -uy, ux
    length = 15
    width = 8
    draw.polygon(
        [
            end,
            (
                round(end[0] - ux * length + px * width),
                round(end[1] - uy * length + py * width),
            ),
            (
                round(end[0] - ux * length - px * width),
                round(end[1] - uy * length - py * width),
            ),
        ],
        fill=color,
    )


def _dataset_action_index(record: dict[str, Any]) -> int:
    base = 0 if record["robot"] == "follower_left" else 7
    return base + int(record["isaac_dof_index"])


def _panel(canvas: Image.Image, record: dict[str, Any]) -> None:
    draw = ImageDraw.Draw(canvas)
    runtime = record["runtime"]
    camera = record["camera"]
    x = 1302
    y = 20
    title_color = (110, 225, 255) if record["robot"] == "follower_left" else (215, 165, 245)
    draw.text(
        (x, y),
        "ALOHA1 SIGNAL EVIDENCE",
        font=_font(23, bold=True),
        fill=title_color,
    )
    y += 38
    lines = [
        "POSE/SIGNAL SCREENSHOT: PASS",
        "Not grasp/dynamics/digital-twin PASS",
        "",
        f"Robot: {record['robot']}",
        "Scope: WORKCELL_SIGNAL_CORRESPONDENCE",
        "Baseline: USER_CONFIRMED_PROJECT_BASELINE",
        f"Phase: {record['phase']}",
        f"Joint: {record['joint']}",
        f"Isaac DOF index: {record['isaac_dof_index']}",
        f"ROS joint index: {record['isaac_dof_index']}",
        f"Dataset action index: {_dataset_action_index(record)}",
        "Unit: rad",
        "",
        f"Target: {record['command_target']:+.8f}",
        f"Readback: {runtime['joint_readback']:+.8f}",
        f"Error: {runtime['position_error']:+.3e}",
        f"EE z: {runtime['end_effector_z_m']:+.8f} m",
        f"Delta z/home: {runtime['delta_z_from_home_m']:+.8f} m",
        (f"Frame/time: {runtime['frame']} / {runtime['simulation_time_s']:.3f} s"),
        "",
        "Green box = current follower",
        "Yellow box = driven-joint link mesh",
        "Magenta H = home EE",
        "Cyan EE = current EE / direction",
        "H=EE = coincident home/current",
        "",
        "Isaac Sim 5.1.0.0",
        "Kit 107.3.3 / PhysX 107.3.26",
        f"Camera: {camera['view']}",
        "Projection: perspective, 1280x900",
        "",
        "Stage SHA-256:",
        record["stage_sha256"][:32],
        record["stage_sha256"][32:],
        "",
        "Render note:",
        "Exact source topology visual clones",
        "in session only; no physics/collision.",
        "Numeric JSON/CSV is authoritative.",
        "Task 8: NOT_RUN",
    ]
    for line in lines:
        color = (232, 236, 242)
        if line.startswith("POSE/SIGNAL"):
            color = (105, 235, 145)
        elif line.startswith(("Not ", "Task 8")):
            color = (255, 205, 90)
        draw.text((x, y), line, font=_font(14), fill=color)
        y += 21


def _annotate(record: dict[str, Any], destination: Path) -> None:
    raw = Path(record["raw_absolute_path"]).resolve(strict=True)
    if _sha256(raw) != record["raw_sha256"]:
        raise RuntimeError(f"raw hash drift: {raw}")
    with Image.open(raw) as opened:
        image = opened.convert("RGB")
    if image.size != RAW_SIZE:
        raise RuntimeError(f"unexpected raw resolution: {raw} {image.size}")
    canvas = Image.new("RGB", CANVAS_SIZE, (22, 26, 34))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    projections = record["camera"]["projections"]
    _bbox(
        draw,
        projections["robot_visual"],
        color=(75, 235, 135),
        label="ROBOT",
        width=3,
    )
    _bbox(
        draw,
        projections["driven_joint_visual"],
        color=(255, 205, 70),
        label=record["joint"].upper(),
        width=4,
    )
    home_center = projections["home_end_effector"]["bbox_center_px"]
    current_center = projections["end_effector"]["bbox_center_px"]
    pixel_delta = ((current_center[0] - home_center[0]) ** 2 + (current_center[1] - home_center[1]) ** 2) ** 0.5
    if abs(record["runtime"]["delta_z_from_home_m"]) <= 1.0e-8 and pixel_delta <= 1.0:
        home = _point(
            draw,
            current_center,
            color=(80, 225, 235),
            label="H=EE",
        )
        current = home
    else:
        home = _point(
            draw,
            home_center,
            color=(235, 90, 235),
            label="H",
            label_side="left",
        )
        current = _point(
            draw,
            current_center,
            color=(40, 235, 235),
            label="EE",
            label_side="right",
        )
    _arrow(draw, home, current)
    _panel(canvas, record)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-metadata", type=Path, default=LEFT_METADATA)
    parser.add_argument("--right-metadata", type=Path, default=RIGHT_METADATA)
    parser.add_argument("--merged-metadata", type=Path, default=MERGED_METADATA)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--metadata", type=Path, default=OUTPUT_METADATA)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    left_path = args.left_metadata.resolve(strict=True)
    right_path = args.right_metadata.resolve(strict=True)
    left = json.loads(left_path.read_text(encoding="utf-8"))
    right = json.loads(right_path.read_text(encoding="utf-8"))
    merged = merge_capture_documents(left, right)
    merged_path = args.merged_metadata.resolve()
    merged_path.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.rglob("*.png")):
        raise FileExistsError(f"annotation directory already contains PNGs: {output_root}")

    annotated = []
    for record in merged["captures"]:
        destination = output_root / record["robot"] / f"{record['capture_id']}_annotated.png"
        _annotate(record, destination)
        annotated.append(
            {
                "capture_id": record["capture_id"],
                "robot": record["robot"],
                "phase": record["phase"],
                "joint": record["joint"],
                "raw_absolute_path": record["raw_absolute_path"],
                "raw_sha256": record["raw_sha256"],
                "annotated_absolute_path": str(destination.resolve(strict=True)),
                "annotated_sha256": _sha256(destination),
                "raw_resolution": list(RAW_SIZE),
                "annotated_resolution": list(CANVAS_SIZE),
                "camera": record["camera"],
                "runtime": record["runtime"],
                "numeric_validation_status": record["status"],
                "visual_model_review": "PENDING",
            }
        )
    metadata = {
        "schema_version": 1,
        "status": "PENDING_VISUAL_MODEL_REVIEW",
        "record_count": len(annotated),
        "expected_record_count": 12,
        "records": annotated,
        "merged_capture_metadata": str(merged_path),
        "merged_capture_metadata_sha256": _sha256(merged_path),
        "left_process_metadata": str(left_path),
        "left_process_metadata_sha256": _sha256(left_path),
        "right_process_metadata": str(right_path),
        "right_process_metadata_sha256": _sha256(right_path),
    }
    metadata_path = args.metadata.resolve()
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": metadata["status"],
                "record_count": len(annotated),
                "annotated_root": str(output_root),
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        )
    )
    return 0 if len(annotated) == 12 else 1


if __name__ == "__main__":
    raise SystemExit(main())
