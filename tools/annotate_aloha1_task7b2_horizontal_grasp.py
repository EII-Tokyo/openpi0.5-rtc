#!/usr/bin/env python3
"""Annotate machine-derived Task 7B.2 horizontal-grasp screenshots."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

PHASE_FRAMES = {
    "release_dynamic": 1,
    "support_settle": 121,
    "open_pregrasp": 131,
    "vertical_descent": 150,
    "bilateral_contact": 165,
    "support_clear": 167,
    "hold_end": 287,
}
VIEWS = ("true_top", "side")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array(values: Any, *, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"expected finite array {shape}, got {array.shape}")
    return array


def _camera_normalized(
    camera_world_matrix: Sequence[Sequence[float]],
    world_points: Sequence[Sequence[float]],
) -> np.ndarray:
    matrix = _array(camera_world_matrix, shape=(4, 4))
    points = np.asarray(world_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("world points must be finite Nx3")
    homogeneous = np.column_stack([points, np.ones(len(points))])
    camera = (np.linalg.inv(matrix) @ homogeneous.T).T[:, :3]
    if np.any(camera[:, 2] >= -1.0e-9):
        raise ValueError("world point is behind the USD camera")
    return np.column_stack([camera[:, 0] / -camera[:, 2], camera[:, 1] / -camera[:, 2]])


def derive_projection_model(
    *,
    camera_world_matrix: Sequence[Sequence[float]],
    projection_world_points: Mapping[str, Sequence[float]],
    projection_pixels_xy: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    """Fit pinhole scalars from Isaac Camera projection readback."""
    labels = [label for label in projection_world_points if label in projection_pixels_xy]
    if len(labels) < 3:
        raise ValueError("at least three runtime projection pairs are required")
    world = [projection_world_points[label] for label in labels]
    pixels = np.asarray(
        [projection_pixels_xy[label] for label in labels],
        dtype=np.float64,
    )
    normalized = _camera_normalized(camera_world_matrix, world)
    u_design = np.column_stack([normalized[:, 0], np.ones(len(labels))])
    v_design = np.column_stack([normalized[:, 1], np.ones(len(labels))])
    u_scale, u_center = np.linalg.lstsq(u_design, pixels[:, 0], rcond=None)[0]
    v_scale, v_center = np.linalg.lstsq(v_design, pixels[:, 1], rcond=None)[0]
    fitted = np.column_stack(
        [
            u_scale * normalized[:, 0] + u_center,
            v_scale * normalized[:, 1] + v_center,
        ]
    )
    rms = float(np.sqrt(np.mean(np.square(fitted - pixels))))
    if not math.isfinite(rms) or rms > 1.0e-3:
        raise ValueError(f"runtime projection reconstruction error {rms} px")
    return {
        "u_scale": float(u_scale),
        "u_center": float(u_center),
        "v_scale": float(v_scale),
        "v_center": float(v_center),
        "rms_error_px": rms,
        "method": "FIT_FROM_ISAAC_CAMERA_RUNTIME_PROJECTION_READBACK",
    }


def project_world_points(
    *,
    camera_world_matrix: Sequence[Sequence[float]],
    model: Mapping[str, float],
    world_points: Sequence[Sequence[float]],
) -> list[list[float]]:
    normalized = _camera_normalized(camera_world_matrix, world_points)
    pixels = np.column_stack(
        [
            float(model["u_scale"]) * normalized[:, 0] + float(model["u_center"]),
            float(model["v_scale"]) * normalized[:, 1] + float(model["v_center"]),
        ]
    )
    return pixels.tolist()


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
    return ImageFont.truetype(str(path), size=size) if path.is_file() else ImageFont.load_default()


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: Sequence[float],
    end: Sequence[float],
    *,
    fill: tuple[int, int, int],
    width: int = 4,
) -> None:
    p0 = np.asarray(start, dtype=np.float64)
    p1 = np.asarray(end, dtype=np.float64)
    draw.line([tuple(p0), tuple(p1)], fill=fill, width=width)
    delta = p1 - p0
    length = float(np.linalg.norm(delta))
    if length <= 1.0e-9:
        return
    direction = delta / length
    normal = np.asarray([-direction[1], direction[0]])
    tip = p1
    wing = max(8.0, min(16.0, length * 0.12))
    for sign in (-1.0, 1.0):
        endpoint = tip - wing * direction + sign * 0.5 * wing * normal
        draw.line([tuple(tip), tuple(endpoint)], fill=fill, width=width)


def _best_contact(
    trial: Mapping[str, Any],
    *,
    side: str,
    frame: int,
) -> Mapping[str, Any] | None:
    candidates = [item for item in trial["contacts"][f"{side}_physical"] if int(item["frame"]) == frame]
    return max(candidates, key=lambda item: float(item["impulse_ns"])) if candidates else None


def annotate_screenshots(
    *,
    runtime_report_path: Path,
    frame_manifest_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    runtime_report = json.loads(runtime_report_path.read_text(encoding="utf-8"))
    trial = runtime_report["trials"][0]
    manifest = json.loads(frame_manifest_path.read_text(encoding="utf-8"))
    if manifest["runtime_trial_signature"] != trial["runtime_trial_signature"]:
        raise ValueError("runtime and frame-manifest signatures differ")
    records = {int(item["physics_frame"]): item for item in manifest["records"]}
    telemetry = {int(item["frame"]): item for item in trial["telemetry"]}
    output_root.mkdir(parents=True, exist_ok=True)
    captures = []
    for phase, frame in PHASE_FRAMES.items():
        record = records[frame]
        if str(record["phase"]) != phase:
            raise ValueError(f"frame {frame} is {record['phase']}, expected {phase}")
        state = telemetry[frame]
        for view in VIEWS:
            source = record["views"][view]
            raw_path = Path(source["absolute_path"]).resolve(strict=True)
            if _sha256(raw_path) != source["sha256"]:
                raise ValueError(f"raw screenshot hash mismatch: {raw_path}")
            camera = trial["video_capture"]["views"][view]
            projection = derive_projection_model(
                camera_world_matrix=camera["camera_world_matrix"],
                projection_world_points=source["projection_world_points"],
                projection_pixels_xy=source["projection_pixels_xy"],
            )
            pixels = source["projection_pixels_xy"]
            with Image.open(raw_path) as opened:
                raw = opened.convert("RGB")
            panel_height = 130
            annotated = Image.new("RGB", (raw.width, raw.height + panel_height), (18, 18, 18))
            annotated.paste(raw, (0, 0))
            draw = ImageDraw.Draw(annotated)
            large = _font(18)
            small = _font(14)
            a_pixel = pixels["bottle_a"]
            b_pixel = pixels["bottle_b"]
            left_pixel = pixels["left_finger_collider_origin"]
            right_pixel = pixels["right_finger_collider_origin"]
            _arrow(draw, a_pixel, b_pixel, fill=(0, 255, 255), width=4)
            draw.text(tuple(a_pixel), "A", fill=(0, 255, 255), font=large)
            draw.text(tuple(b_pixel), "B", fill=(0, 255, 255), font=large)
            for label, pixel, color in (
                ("L", left_pixel, (40, 130, 255)),
                ("R", right_pixel, (255, 135, 30)),
            ):
                x, y = (float(value) for value in pixel)
                draw.ellipse(
                    (x - 10, y - 10, x + 10, y + 10),
                    fill=color,
                )
                draw.text(
                    (x, y),
                    label,
                    fill=(255, 255, 255),
                    font=small,
                    anchor="mm",
                )

            contact_records = {}
            for side, color in (
                ("left", (40, 130, 255)),
                ("right", (255, 135, 30)),
            ):
                contact = _best_contact(trial, side=side, frame=frame)
                if contact is None:
                    contact_records[side] = None
                    continue
                point = np.asarray(contact["position_world_m"], dtype=np.float64)
                normal = np.asarray(contact["normal_world"], dtype=np.float64)
                endpoint = point + 0.025 * normal
                contact_pixel, normal_pixel = project_world_points(
                    camera_world_matrix=camera["camera_world_matrix"],
                    model=projection,
                    world_points=[point, endpoint],
                )
                _arrow(
                    draw,
                    contact_pixel,
                    normal_pixel,
                    fill=color,
                    width=3,
                )
                draw.text(
                    tuple(contact_pixel),
                    f"{side[0].upper()}c",
                    fill=color,
                    font=small,
                    anchor="ms",
                )
                contact_records[side] = {
                    "position_world_m": point.tolist(),
                    "normal_world": normal.tolist(),
                    "contact_pixel_xy": contact_pixel,
                    "normal_endpoint_pixel_xy": normal_pixel,
                    "impulse_ns": float(contact["impulse_ns"]),
                    "separation_m": float(contact["separation_m"]),
                }

            bottle = state["bottle"]
            base_y = raw.height + 8
            draw.text(
                (12, base_y),
                (
                    f"PHYSICAL FAIL: {trial['failure_mode']} | "
                    f"{view} | {phase} | frame {frame} | "
                    f"t={float(state['time_s']):.3f}s"
                ),
                fill=(255, 90, 90),
                font=large,
            )
            draw.text(
                (12, base_y + 30),
                ("cyan A->B = CAD bottle axis; blue/orange L/R = collider prim origins, NOT contact regions"),
                fill=(235, 235, 235),
                font=small,
            )
            draw.text(
                (12, base_y + 54),
                ("contact arrows appear only for a physical contact-report sample at this exact frame"),
                fill=(235, 235, 235),
                font=small,
            )
            draw.text(
                (12, base_y + 78),
                (
                    f"bottle_z={float(bottle['position_world_m'][2]):+.5f} m | "
                    f"support_clearance="
                    f"{float(bottle['bottom_clearance_m']):+.5f} m | "
                    f"signature={trial['runtime_trial_signature'][:12]}"
                ),
                fill=(235, 235, 235),
                font=small,
            )
            draw.text(
                (12, base_y + 102),
                "visual evidence scope only; Task 8 NOT_RUN",
                fill=(255, 210, 90),
                font=small,
            )
            annotated_path = output_root / view / f"{phase}_{frame:06d}_annotated.png"
            annotated_path.parent.mkdir(parents=True, exist_ok=True)
            annotated.save(annotated_path)
            matrix = np.asarray(camera["camera_world_matrix"], dtype=np.float64)
            forward = matrix[:3, :3] @ np.asarray([0.0, 0.0, -1.0])
            captures.append(
                {
                    "view_name": view,
                    "phase": phase,
                    "frame": frame,
                    "time_s": float(state["time_s"]),
                    "raw_absolute_path": str(raw_path),
                    "raw_sha256": _sha256(raw_path),
                    "annotated_absolute_path": str(annotated_path.resolve()),
                    "annotated_sha256": _sha256(annotated_path),
                    "resolution": [raw.width, raw.height],
                    "annotated_resolution": list(annotated.size),
                    "camera_world_matrix": camera["camera_world_matrix"],
                    "camera_forward_world": forward.tolist(),
                    "projection_model": projection,
                    "bottle_a_world": bottle["a_world_m"],
                    "bottle_b_world": bottle["b_world_m"],
                    "bottle_axis_world": bottle["axis_world"],
                    "left_finger_origin_world": source["projection_world_points"]["left_finger_collider_origin"],
                    "right_finger_origin_world": source["projection_world_points"]["right_finger_collider_origin"],
                    "contact_records": contact_records,
                    "bottle_z_m": float(bottle["position_world_m"][2]),
                    "support_clearance_m": float(bottle["bottom_clearance_m"]),
                    "machine_status": trial["physical_trial_status"],
                    "machine_failure_mode": trial["failure_mode"],
                    "vision_review_status": "PENDING_VISUAL_MODEL_REVIEW",
                    "retake_reason": None,
                }
            )
    candidate = {
        "schema_version": 1,
        "status": "PENDING_VISUAL_MODEL_REVIEW",
        "runtime_trial_signature": trial["runtime_trial_signature"],
        "physical_trial_status": trial["physical_trial_status"],
        "machine_conclusion": runtime_report["conclusion"],
        "stage_path": trial["stage"]["absolute_path"],
        "stage_sha256": trial["stage"]["sha256_after"],
        "runtime_report_absolute_path": str(runtime_report_path.resolve()),
        "runtime_report_sha256": _sha256(runtime_report_path),
        "frame_manifest_absolute_path": str(frame_manifest_path.resolve()),
        "frame_manifest_sha256": _sha256(frame_manifest_path),
        "capture_count": len(captures),
        "captures": captures,
    }
    output = output_root / "screenshot_candidate_manifest.json"
    output.write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return candidate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--frame-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    candidate = annotate_screenshots(
        runtime_report_path=args.runtime_report.resolve(strict=True),
        frame_manifest_path=args.frame_manifest.resolve(strict=True),
        output_root=args.output_root.resolve(),
    )
    print(
        json.dumps(
            {
                "status": candidate["status"],
                "capture_count": candidate["capture_count"],
                "output": str((args.output_root / "screenshot_candidate_manifest.json").resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
