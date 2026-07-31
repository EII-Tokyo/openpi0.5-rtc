#!/usr/bin/env python3
"""Annotate follower-finger Bottle500 collider-overlay screenshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "reports/aloha1_mapping/aloha1_follower_finger_collision_runtime_run04.json"
)
DEFAULT_REVIEW_JSON = (
    ROOT
    / "reports/aloha1_mapping/aloha1_follower_finger_collision_screenshot_review.json"
)
DEFAULT_REVIEW_MD = (
    ROOT
    / "reports/aloha1_mapping/aloha1_follower_finger_collision_screenshot_review.md"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--review-json", type=Path, default=DEFAULT_REVIEW_JSON)
    parser.add_argument("--review-md", type=Path, default=DEFAULT_REVIEW_MD)
    parser.add_argument("--vision-reviewed", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def _rotation_wxyz(quaternion: list[float]) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _project_world(record: dict[str, Any], point: list[float]) -> tuple[int, int] | None:
    position = np.asarray(record["camera_position_world_m"], dtype=np.float64)
    rotation = _rotation_wxyz(record["camera_orientation_wxyz"])
    camera = rotation.T @ (np.asarray(point, dtype=np.float64) - position)
    depth = -float(camera[2])
    if depth <= 1.0e-9:
        return None
    intrinsics = np.asarray(record["camera_intrinsics_pixels"], dtype=np.float64)
    u = intrinsics[0, 0] * float(camera[0]) / depth + intrinsics[0, 2]
    v = intrinsics[1, 2] - intrinsics[1, 1] * float(camera[1]) / depth
    return round(u), round(v)


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    fill: tuple[int, int, int],
    width: int = 4,
) -> None:
    draw.line([start, end], fill=fill, width=width)
    direction = np.asarray(start, dtype=np.float64) - np.asarray(end, dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if length <= 0.0:
        return
    unit = direction / length
    normal = np.asarray([-unit[1], unit[0]], dtype=np.float64)
    tip = np.asarray(end, dtype=np.float64)
    for sign in (-1.0, 1.0):
        wing = tip + 14.0 * unit + sign * 7.0 * normal
        draw.line([tuple(tip.astype(int)), tuple(wing.astype(int))], fill=fill, width=width)


def _draw_marker(
    draw: ImageDraw.ImageDraw,
    point: tuple[int, int] | None,
    *,
    label: str,
    color: tuple[int, int, int],
) -> None:
    if point is None:
        return
    x, y = point
    draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(18, 18, 18), outline=color, width=3)
    draw.text((x + 10, y - 13), label, font=_font(16, bold=True), fill=color)


def _green_overlay_pixels(image: Image.Image) -> int:
    pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
    red, green, blue = (pixels[:, :, index] for index in range(3))
    mask = (green > 75) & (green - red > 8) & (green - blue > -5)
    return int(np.count_nonzero(mask))


def _joint_record(report: dict[str, Any], frame: int) -> dict[str, Any]:
    matches = [
        record for record in report["joint_trace"] if int(record["frame"]) == frame
    ]
    if len(matches) != 1:
        raise RuntimeError(f"joint trace frame is not unique: {frame}")
    return matches[0]


def _annotate(
    raw: Image.Image,
    *,
    capture: dict[str, Any],
    report: dict[str, Any],
) -> Image.Image:
    panel_width = 590
    canvas = Image.new("RGB", (raw.width + panel_width, raw.height), (22, 25, 30))
    canvas.paste(raw.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    title = _font(20, bold=True)
    body = _font(12)
    small = _font(13)

    left_position = capture["left_finger_physx_transform_xyzw"][:3]
    right_position = capture["right_finger_physx_transform_xyzw"][:3]
    _draw_marker(
        draw,
        _project_world(capture, left_position),
        label="L",
        color=(50, 130, 255),
    )
    _draw_marker(
        draw,
        _project_world(capture, right_position),
        label="R",
        color=(255, 135, 25),
    )
    _draw_marker(
        draw,
        _project_world(capture, capture["bottle_position_world_m"]),
        label="B",
        color=(235, 245, 255),
    )
    for side, color in (("left", (255, 230, 35)), ("right", (255, 70, 220))):
        contact = capture["contact_evidence"][side]
        if contact is None:
            continue
        point = _project_world(capture, contact["position_world_m"])
        endpoint = _project_world(
            capture,
            (
                np.asarray(contact["position_world_m"], dtype=np.float64)
                + 0.030 * np.asarray(contact["normal_world"], dtype=np.float64)
            ).tolist(),
        )
        if point is not None:
            _draw_marker(draw, point, label=f"{side[0].upper()}C", color=color)
        if point is not None and endpoint is not None:
            _draw_arrow(draw, point, endpoint, fill=color)

    draw.rectangle([8, 8, 350, 39], fill=(15, 18, 22))
    draw.text(
        (16, 12),
        "GREEN = COLLISION AREA DISPLAY",
        font=_font(16, bold=True),
        fill=(80, 255, 95),
    )
    panel_x = raw.width + 20
    draw.text((panel_x, 16), "ALOHA finger collision evidence", font=title, fill=(245, 245, 245))
    frame = int(capture["physics_frame"])
    joint = _joint_record(report, frame)
    contact_left = capture["contact_evidence"]["left"]
    contact_right = capture["contact_evidence"]["right"]
    lines = [
        f"MACHINE: {report['status']}",
        f"class: {report['classification']}",
        f"phase: {capture['phase']}",
        f"view: {capture['view']}",
        f"frame/time: {frame} / {capture['time_s']:.3f} s",
        "",
        "Isaac Sim 5.1.0.0",
        "Kit 107.3.3 / PhysX 107.3.26",
        f"Stage SHA: {report['stage']['sha256_after'][:20]}...",
        "displayColliders = 2",
        "",
        "Blue = left_finger",
        "Orange = right_finger",
        "Green = collision-area overlay",
        "B = Bottle500 body origin",
        "Yellow = left contact/normal",
        "Magenta = right contact/normal",
        "",
        f"left contact: {contact_left is not None}",
        f"right contact: {contact_right is not None}",
        (
            f"left impulse: {contact_left['impulse_ns']:.7f} Ns"
            if contact_left is not None
            else "left impulse: n/a"
        ),
        (
            f"right impulse: {contact_right['impulse_ns']:.7f} Ns"
            if contact_right is not None
            else "right impulse: n/a"
        ),
        (
            f"left sep: {contact_left['separation_m'] * 1000.0:.4f} mm"
            if contact_left is not None
            else "left sep: n/a"
        ),
        (
            f"right sep: {contact_right['separation_m'] * 1000.0:.4f} mm"
            if contact_right is not None
            else "right sep: n/a"
        ),
        (
            "max penetration depth: "
            f"{abs(report['contacts']['maximum_penetration_m']) * 1000.0:.4f} mm"
        ),
        "",
        f"L target/readback: {joint['target'][7]:.6f} / {joint['readback'][7]:.6f}",
        f"R target/readback: {joint['target'][8]:.6f} / {joint['readback'][8]:.6f}",
        "",
        "Each finger link has exactly one",
        "enabled supplier-CAD collider.",
        "approximation = convexHull",
        "",
        "Green pixels combine Isaac's physics",
        "debug display and session-only exact",
        "authored-collider render evidence.",
        "They are not a cooked-hull readback.",
        "",
        "PASS = collision pipeline only.",
        "NOT a static grasp or five-trial PASS.",
        "Task 8 = NOT_RUN",
    ]
    y = 52
    for line in lines:
        color = (100, 245, 125) if line.startswith("MACHINE") else (225, 229, 235)
        draw.text((panel_x, y), line, font=body, fill=color)
        y += 16
    draw.text(
        (15, raw.height - 25),
        "Markers are runtime-projected; arrows use PhysX contact-report normals.",
        font=small,
        fill=(245, 245, 245),
    )
    return canvas


def _render_markdown(review: dict[str, Any]) -> str:
    lines = [
        "# ALOHA follower finger collision screenshot review",
        "",
        f"- Status: `{review['status']}`",
        f"- Input runtime report: `{review['input_report']}`",
        f"- Final raw root: `{review['raw_root_absolute_path']}`",
        f"- Final annotated root: `{review['annotated_root_absolute_path']}`",
        "",
        "Every accepted screenshot has Isaac Sim's collision-area display enabled",
        "with `/persistent/physics/visualizationDisplayColliders = 2`.",
        "",
        "| phase | view | raw | annotated | visual review |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        (
            f"| {record['phase']} | {record['view']} | "
            f"`{record['raw_absolute_path']}` | "
            f"`{record['annotated_absolute_path']}` | "
            f"{record['visual_model_review']} |"
        )
        for record in review["records"]
    )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- The screenshots support the machine collision/contact report; they do not replace it.",
            "- `PASS` here means the collision regions are visible and the evidence image is reviewable.",
            "- It does not mean Bottle500 static grasp or the five-position acceptance campaign passed.",
            "- Task 8 remains `NOT_RUN`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    input_path = args.input.resolve(strict=True)
    report = json.loads(input_path.read_text(encoding="utf-8"))
    if report["status"] != "PASS":
        raise RuntimeError("runtime collision report is not PASS")
    captures = [
        record
        for record in report["capture_manifest"]
        if record["mode"] == "physics_collider_overlay"
    ]
    if len(captures) != 8:
        raise RuntimeError(f"expected 8 collision-overlay captures, got {len(captures)}")
    records: list[dict[str, Any]] = []
    for capture in captures:
        raw_path = Path(capture["absolute_path"]).resolve(strict=True)
        annotated_path = (
            raw_path.parents[2]
            / "screenshots_annotated"
            / capture["phase"]
            / raw_path.name.replace("_raw.png", "_annotated.png")
        )
        with Image.open(raw_path) as source:
            source.load()
            raw = source.convert("RGB")
        annotated = _annotate(raw, capture=capture, report=report)
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(annotated_path)
        green_pixels = _green_overlay_pixels(raw)
        automated = (
            "PASS"
            if raw.size == (960, 720)
            and green_pixels >= 1_000
            and capture["display_colliders_readback"] == 2
            and capture["same_physics_frame"]
            and capture["same_camera_pose"]
            else "FAIL"
        )
        records.append(
            {
                "phase": capture["phase"],
                "view": capture["view"],
                "physics_frame": capture["physics_frame"],
                "time_s": capture["time_s"],
                "raw_absolute_path": str(raw_path),
                "raw_sha256": _sha256(raw_path),
                "raw_dimensions_px": list(raw.size),
                "annotated_absolute_path": str(annotated_path.resolve()),
                "annotated_sha256": _sha256(annotated_path),
                "annotated_dimensions_px": list(annotated.size),
                "camera_position_world_m": capture["camera_position_world_m"],
                "camera_orientation_wxyz": capture["camera_orientation_wxyz"],
                "camera_intrinsics_pixels": capture["camera_intrinsics_pixels"],
                "detection_target": (
                    "both supplier-CAD fingers, Bottle500, relative placement, "
                    "green collision regions, and projected contact/normal where present"
                ),
                "green_overlay_pixel_count": green_pixels,
                "automated_precheck": automated,
                "visual_model_review": (
                    "PASS"
                    if args.vision_reviewed
                    else "PENDING_VISION_MODEL_REVIEW"
                ),
                "retake_reason": None,
            }
        )
    automated_pass = all(item["automated_precheck"] == "PASS" for item in records)
    vision_pass = args.vision_reviewed and all(
        item["visual_model_review"] == "PASS" for item in records
    )
    review = {
        "schema_version": 1,
        "status": "PASS" if automated_pass and vision_pass else "PARTIAL",
        "input_report": str(input_path),
        "input_report_sha256": _sha256(input_path),
        "runtime_collision_status": report["status"],
        "runtime_collision_classification": report["classification"],
        "final_capture_count": len(records),
        "raw_root_absolute_path": str(
            Path(records[0]["raw_absolute_path"]).parents[2]
        ),
        "annotated_root_absolute_path": str(
            Path(records[0]["annotated_absolute_path"]).parents[2]
        ),
        "collision_display_contract": {
            "setting": "/persistent/physics/visualizationDisplayColliders",
            "value": 2,
            "every_final_raw_capture_has_collision_display": True,
            "green_pixel_semantics": (
                "ISAAC_PHYSICS_DEBUG_DISPLAY_PLUS_SESSION_AUTHORED_COLLIDER_CLONE_"
                "NOT_SEPARABLE_PIXELWISE_NOT_COOKED_HULL_READBACK"
            ),
        },
        "retake_history": [
            {
                "attempt": "follower_finger_runtime_run01",
                "status": "REJECTED_OCCLUDED_TRUE_TOP_AND_ONE_SIDED_CONTACT_VIEW",
            },
            {
                "attempt": "follower_finger_runtime_run02",
                "status": "REJECTED_FINGER_CROPPING",
            },
            {
                "attempt": "follower_finger_runtime_run03",
                "status": "ACCEPTABLE_FRAMING_BUT_SUPERSEDED_TO_ADD_INTRINSICS_AND_CONTACT_METADATA",
            },
            {
                "attempt": "follower_finger_runtime_run04",
                "status": (
                    "PASS"
                    if automated_pass and vision_pass
                    else "PENDING_OR_FAILED_VISUAL_MODEL_REVIEW"
                ),
            },
        ],
        "records": records,
        "boundaries": {
            "collision_pipeline_only": True,
            "static_grasp_pass": "NOT_EVALUATED",
            "five_position_acceptance": "NOT_RUN",
            "source_or_final_asset_modified": False,
            "task8": "NOT_RUN",
        },
    }
    _write_json(args.review_json.resolve(), review)
    _write_text(args.review_md.resolve(), _render_markdown(review))
    print(
        json.dumps(
            {
                "status": review["status"],
                "records": len(records),
                "review_json": str(args.review_json.resolve()),
                "review_markdown": str(args.review_md.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0 if automated_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
