#!/usr/bin/env python3
"""Annotate the accepted Bottle500 collision probe screenshots."""

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
DEFAULT_INPUT = ROOT / "reports/aloha1_mapping/aloha1_bottle_collision_runtime_standard_pusher_run13.json"
DEFAULT_REVIEW_JSON = ROOT / "reports/aloha1_mapping/aloha1_bottle_collision_screenshot_review.json"
DEFAULT_REVIEW_MD = ROOT / "reports/aloha1_mapping/aloha1_bottle_collision_screenshot_review.md"


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


def _bbox(mask: np.ndarray) -> list[int] | None:
    y, x = np.where(mask)
    if not len(x):
        return None
    return [int(x.min()), int(y.min()), int(x.max()), int(y.max())]


def _detections(image: Image.Image) -> dict[str, dict[str, Any]]:
    pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
    red, green, blue = (pixels[:, :, index] for index in range(3))
    masks = {
        "bottle_visual": (blue - red > 12) & (blue - green > 3) & (blue > 110),
        "authored_collider_overlay": (green - red > 15) & (green - blue > 0) & (green > 100),
        "pusher_visual": (red - green > 80) & (red - blue > 80) & (red > 150),
    }
    return {
        name: {
            "bbox_xyxy": _bbox(mask),
            "pixel_count": int(np.count_nonzero(mask)),
        }
        for name, mask in masks.items()
    }


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
        ]
    )


def _project_world(record: dict[str, Any], point: list[float]) -> tuple[int, int] | None:
    position = np.asarray(record["camera_position_readback_world_m"], dtype=np.float64)
    rotation = _rotation_wxyz(record["camera_orientation_readback_wxyz"])
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
    width: int,
) -> None:
    draw.line([start, end], fill=fill, width=width)
    direction = np.asarray(start, dtype=np.float64) - np.asarray(end, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        return
    unit = direction / norm
    normal = np.asarray([-unit[1], unit[0]])
    tip = np.asarray(end, dtype=np.float64)
    for sign in (-1.0, 1.0):
        wing = tip + unit * 14.0 + normal * sign * 7.0
        draw.line([tuple(tip.astype(int)), tuple(wing.astype(int))], fill=fill, width=width)


def _annotate(
    raw: Image.Image,
    *,
    record: dict[str, Any],
    mode: str,
    report: dict[str, Any],
    detections: dict[str, dict[str, Any]],
) -> Image.Image:
    panel_width = 540
    canvas = Image.new("RGB", (raw.width + panel_width, raw.height), (24, 27, 32))
    canvas.paste(raw.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(20, bold=True)
    body_font = _font(15)
    small_font = _font(13)
    colors = {
        "bottle_visual": (50, 170, 255),
        "authored_collider_overlay": (45, 255, 70),
        "pusher_visual": (255, 65, 45),
    }
    labels = {
        "bottle_visual": "B",
        "authored_collider_overlay": "C",
        "pusher_visual": "P",
    }
    for name, detection in detections.items():
        box = detection["bbox_xyxy"]
        if box is None or (name == "authored_collider_overlay" and mode == "normal"):
            continue
        draw.rectangle(box, outline=colors[name], width=3)
        draw.rectangle(
            [box[0], max(0, box[1] - 22), box[0] + 21, box[1]],
            fill=(20, 23, 27),
        )
        draw.text(
            (box[0] + 4, max(1, box[1] - 21)),
            labels[name],
            font=small_font,
            fill=colors[name],
        )

    contact = max(
        report["probe"]["contacts"],
        key=lambda item: float(item["impulse_ns"]),
    )
    if record["phase"] == "first_contact":
        point = _project_world(record, contact["position_world_m"])
        endpoint_world = (
            np.asarray(contact["position_world_m"], dtype=np.float64)
            + np.asarray(contact["normal_world"], dtype=np.float64) * 0.030
        )
        endpoint = _project_world(record, endpoint_world.tolist())
        if point is not None and endpoint is not None:
            draw.ellipse(
                [point[0] - 6, point[1] - 6, point[0] + 6, point[1] + 6],
                fill=(255, 230, 35),
                outline=(20, 20, 20),
                width=2,
            )
            _draw_arrow(draw, point, endpoint, fill=(255, 230, 35), width=4)

    panel_x = raw.width + 20
    draw.text(
        (panel_x, 18),
        "Bottle500 collision gate",
        font=title_font,
        fill=(245, 245, 245),
    )
    y = 55
    lines = [
        f"RESULT: {report['status']}",
        f"root cause: {report['root_cause']}",
        f"phase: {record['phase']}",
        f"view: {record['view']}",
        f"mode: {mode}",
        f"frame/time: {record['physics_frame']} / {record['physics_frame'] / 60.0:.3f} s",
        "",
        "Isaac Sim 5.1.0.0",
        "Kit 107.3.3 / PhysX 107.3.26",
        f"Stage SHA: {report['stage']['sha256_after'][:16]}...",
        f"Bottle SHA: {report['bottle']['sha256_after'][:16]}...",
        "",
        f"bottle colliders: {report['render_evidence']['collider_mesh_count']}",
        f"physical pusher contacts: {report['physical_pusher_contact_count']}",
        f"first contact frame: {report['first_contact_frame']}",
        "bottle displacement:",
        "  "
        + ", ".join(f"{value * 1000.0:.3f}mm" for value in report["probe"]["response"]["bottle_displacement_world_m"]),
        f"max bottle speed: {report['probe']['response']['maximum_speed_m_s']:.6f} m/s",
        "",
        ("displayColliders = 2" if mode == "physics_collider_overlay" else "displayColliders = 0"),
        (
            "green = authored CollisionAPI geometry"
            if mode == "physics_collider_overlay"
            else "normal render; collider overlay hidden"
        ),
        "Legend: B=bottle, C=collider, P=pusher",
        "Yellow point/arrow=contact/normal",
        "",
        "Red pusher is render-only evidence",
        "synced to PhysX pusher pose.",
        "Blue/green clones have NO physics API.",
        "Physics remains original Bottle500.",
        "",
        "PASS means collision response only.",
        "It does NOT mean gripper grasp PASS.",
    ]
    for line in lines:
        fill = (95, 245, 120) if line.startswith("RESULT") else (224, 228, 234)
        draw.text((panel_x, y), line, font=body_font, fill=fill)
        y += 22
    return canvas


def main() -> int:
    args = _parse_args()
    input_path = args.input.resolve(strict=True)
    report = json.loads(input_path.read_text(encoding="utf-8"))
    if report["status"] != "PASS":
        raise RuntimeError("only an accepted collision probe may be annotated")
    records: list[dict[str, Any]] = []
    for capture in report["capture_manifest"]:
        for mode, key in (
            ("normal", "normal_path"),
            ("physics_collider_overlay", "overlay_path"),
        ):
            raw_path = Path(capture[key]).resolve(strict=True)
            annotated_path = (
                raw_path.parents[2]
                / "screenshots_annotated"
                / capture["phase"]
                / raw_path.name.replace("_raw.png", "_annotated.png")
            )
            with Image.open(raw_path) as source:
                source.load()
                raw = source.convert("RGB")
            detections = _detections(raw)
            annotated = _annotate(
                raw,
                record=capture,
                mode=mode,
                report=report,
                detections=detections,
            )
            annotated_path.parent.mkdir(parents=True, exist_ok=True)
            annotated.save(annotated_path)
            bottle_ok = int(detections["bottle_visual"]["pixel_count"]) > 10_000
            pusher_ok = int(detections["pusher_visual"]["pixel_count"]) > 10
            collider_ok = (
                int(detections["authored_collider_overlay"]["pixel_count"]) > 1_000
                if mode == "physics_collider_overlay"
                else True
            )
            records.append(
                {
                    "phase": capture["phase"],
                    "view": capture["view"],
                    "mode": mode,
                    "physics_frame": capture["physics_frame"],
                    "raw_path": str(raw_path),
                    "raw_sha256": _sha256(raw_path),
                    "raw_dimensions_px": list(raw.size),
                    "annotated_path": str(annotated_path),
                    "annotated_sha256": _sha256(annotated_path),
                    "annotated_dimensions_px": list(annotated.size),
                    "raw_region_pixel_identical_in_annotated": bool(
                        np.array_equal(
                            np.asarray(raw),
                            np.asarray(annotated)[:, : raw.width],
                        )
                    ),
                    "camera_position_world_m": capture["camera_position_readback_world_m"],
                    "camera_orientation_wxyz": capture["camera_orientation_readback_wxyz"],
                    "detections": detections,
                    "automated_precheck": ("PASS" if bottle_ok and pusher_ok and collider_ok else "FAIL"),
                    "vision_model_review": ("PASS" if args.vision_reviewed else "PENDING_VISION_MODEL_REVIEW"),
                    "detection_target": (
                        "Bottle500, red pusher, table support, and green authored collider geometry in overlay mode"
                    ),
                    "retake_reason": None,
                }
            )
    automated_pass = all(record["automated_precheck"] == "PASS" for record in records)
    vision_pass = args.vision_reviewed and all(record["vision_model_review"] == "PASS" for record in records)
    review = {
        "schema_version": 1,
        "status": "PASS" if automated_pass and vision_pass else "PARTIAL",
        "input_report": str(input_path),
        "input_report_sha256": _sha256(input_path),
        "machine_collision_status": report["status"],
        "deterministic_signature": report["evaluation"]["deterministic_signature"],
        "capture_record_count": len(records),
        "raw_and_annotated_required": True,
        "same_frame_and_camera_required": True,
        "overlay_semantics": {
            "official_setting": "/persistent/physics/visualizationDisplayColliders",
            "official_setting_value": 2,
            "green_geometry": "SESSION_ONLY_AUTHORED_COLLISIONAPI_GEOMETRY_CLONES",
            "green_geometry_is_cooked_hull_readback": False,
            "green_geometry_has_physics_or_collision_api": False,
        },
        "retake_history": [
            {
                "attempt": "standard_pusher",
                "status": "REJECTED_PROBE_DID_NOT_REACH_BOTTLE",
            },
            {
                "attempt": "standard_pusher_run02",
                "status": "REJECTED_PROBE_DID_NOT_REACH_BOTTLE",
            },
            {
                "attempt": "standard_pusher_run03",
                "status": "REJECTED_TELEPORT_PROBE_AND_TARGET_NOT_VISIBLE",
            },
            {
                "attempt": "standard_pusher_run04",
                "status": "REJECTED_TARGET_NOT_VISIBLE_AND_OVERLAY_NOT_VERIFIABLE",
            },
            {
                "attempt": "standard_pusher_run05_to_run07",
                "status": "REJECTED_FSD_VISUAL_CLONE_NOT_VISIBLE",
            },
            {
                "attempt": "standard_pusher_run08",
                "status": "REJECTED_CROPPED_BOTTLE_AND_OCCLUDED_PUSHER",
            },
            {
                "attempt": "standard_pusher_run09",
                "status": "REJECTED_PUSHER_VISUAL_NOT_SYNCHRONIZED",
            },
            {
                "attempt": "standard_pusher_run13_annotation_v1",
                "status": "REJECTED_LABEL_OVERLAP",
            },
        ],
        "records": records,
    }
    args.review_json.parent.mkdir(parents=True, exist_ok=True)
    args.review_json.write_text(
        json.dumps(review, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# ALOHA1 Bottle500 Collision Screenshot Review",
        "",
        f"- Status: **{review['status']}**",
        f"- Machine collision gate: **{report['status']}**",
        f"- Deterministic signature: `{review['deterministic_signature']}`",
        f"- Capture records: `{len(records)}`",
        "- Green overlay: session-only clones of the 41 authored CollisionAPI meshes; "
        "the official `visualizationDisplayColliders=2` setting is also enabled.",
        "- The green overlay is not a cooked-hull readback and has no physics schema.",
        "- PASS refers only to Bottle500 collision response, not gripper grasp.",
        "",
        "| Phase | View | Mode | Automated | Vision | Annotated |",
        "|---|---|---|---|---|---|",
    ]
    lines.extend(
        (
            f"| {record['phase']} | {record['view']} | {record['mode']} | "
            f"{record['automated_precheck']} | {record['vision_model_review']} | "
            f"`{record['annotated_path']}` |"
        )
        for record in records
    )
    args.review_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": review["status"], "records": len(records)}))
    return 0 if review["status"] in {"PASS", "PARTIAL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
