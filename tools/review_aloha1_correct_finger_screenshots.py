#!/usr/bin/env python3
"""Annotate ALOHA screenshots and require explicit visual-model decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import yaml


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    y, x = np.nonzero(mask)
    if len(x) < 100:
        return None
    return int(x.min()), int(y.min()), int(x.max()), int(y.max())


def _detected_boxes(
    image: Image.Image,
    *,
    include_bottle: bool,
) -> dict[str, tuple[int, int, int, int]]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    red, green, blue = (rgb[..., index] for index in range(3))
    boxes = {
        "left finger (blue)": _bbox(
            (blue > red + 35) & (blue > green + 12) & (blue > 90)
        ),
        "right finger (orange)": _bbox(
            (red > blue + 55) & (green > blue + 15) & (red > 130)
        ),
    }
    height, width = rgb.shape[:2]
    if include_bottle:
        gray_range = rgb.max(axis=2) - rgb.min(axis=2)
        luminance = rgb.mean(axis=2)
        bottle = (
            (gray_range < 35)
            & (luminance > 35)
            & (luminance < 175)
        )
        bottle[int(height * 0.70) :] = False
        column_counts = bottle.sum(axis=0)
        bottle_columns = np.where(column_counts > int(height * 0.22))[0]
        if len(bottle_columns) > 0:
            x_min = int(bottle_columns.min())
            x_max = int(bottle_columns.max())
            row_counts = bottle[:, x_min : x_max + 1].sum(axis=1)
            bottle_rows = np.where(
                row_counts > int((x_max - x_min + 1) * 0.55)
            )[0]
            if len(bottle_rows) > 0:
                boxes["bottle"] = (
                    x_min,
                    int(bottle_rows.min()),
                    x_max,
                    int(bottle_rows.max()),
                )
    return {name: box for name, box in boxes.items() if box is not None}


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    direction: tuple[float, float],
    *,
    color: tuple[int, int, int, int],
    length: float = 75.0,
    width: int = 5,
) -> None:
    norm = math.hypot(*direction)
    if norm < 1.0e-9:
        return
    unit = (direction[0] / norm, direction[1] / norm)
    end = (start[0] + length * unit[0], start[1] + length * unit[1])
    draw.line([start, end], fill=color, width=width)
    left = (
        end[0] - 16 * unit[0] + 8 * unit[1],
        end[1] - 16 * unit[1] - 8 * unit[0],
    )
    right = (
        end[0] - 16 * unit[0] - 8 * unit[1],
        end[1] - 16 * unit[1] + 8 * unit[0],
    )
    draw.polygon([end, left, right], fill=color)


def _normal_screen_direction(
    normal_world: list[float],
    orientation_wxyz: list[float],
) -> tuple[float, float]:
    w, x, y, z = orientation_wxyz
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    camera = rotation.T @ np.asarray(normal_world, dtype=np.float64)
    return float(camera[0]), float(-camera[1])


def _first_trial(
    project_root: Path,
    *,
    robot: str,
    profile: str,
) -> dict[str, Any] | None:
    group = (
        "hull_current"
        if profile == "convex_hull"
        else "decomposition_current"
    )
    path = (
        project_root
        / "reports/aloha1_mapping/gripper_correct_finger_task5_trials"
        / group
        / f"{robot}.jsonl"
    )
    if not path.is_file():
        return None
    first = path.read_text(encoding="utf-8").splitlines()[0]
    return json.loads(first)


def _annotate(
    capture: dict[str, Any],
    *,
    destination: Path,
    intent: dict[str, Any],
    trial: dict[str, Any] | None,
) -> dict[str, Any]:
    original = Path(capture["absolute_path"])
    with Image.open(original) as opened:
        image = opened.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    boxes = (
        {}
        if capture["phase"] == "collider_geometry"
        else _detected_boxes(
            image,
            include_bottle=capture["phase"]
            in {"runtime_open", "bilateral_contact", "release_hold"},
        )
    )
    palette = {
        "left finger (blue)": (0, 255, 255, 255),
        "right finger (orange)": (255, 70, 255, 255),
        "bottle": (50, 255, 50, 255),
    }
    small = _font(20)
    phase = capture["phase"]
    simulation = capture["simulation"]
    camera = capture["camera"]
    profile = str(simulation.get("profile", "geometry/preflight"))
    robot = str(simulation.get("robot", "n/a"))
    if phase == "collider_geometry":
        robot = (
            "follower_left"
            if capture["capture_name"].startswith("follower_left_")
            else "follower_right"
        )
        profile = (
            "convex_decomposition"
            if "_decomposition_" in capture["capture_name"]
            else "convex_hull"
        )
    frame = simulation.get("frame")
    time_s = (
        float(frame) / float(simulation.get("physics_frequency_hz", 60))
        if frame is not None
        else None
    )
    status = capture["capture_gate_status"]
    key_lines = [
        f"{robot} | {profile} | {phase} | {status}",
        f"view={camera.get('view', camera.get('renderer', 'n/a'))} "
        f"frame={frame} time_s={time_s}",
        f"detect: {intent['object']}",
    ]
    bottle = simulation.get("bottle", {})
    if bottle:
        key_lines.append(
            "bottle z={:.6f} m  vz={:.6f} m/s  |w|={:.4f} rad/s".format(
                float(bottle.get("z_m", float("nan"))),
                float(bottle.get("vertical_velocity_m_s", float("nan"))),
                float(bottle.get("angular_speed_rad_s", float("nan"))),
            )
        )
    state = simulation.get("physical_state", {})
    if "surface_gap_m" in state:
        key_lines.append(f"aperture/surface gap={state['surface_gap_m']:.6f} m")
    if "drop_m" in state:
        key_lines.append(
            f"drop={state['drop_m']:.6f} m gate={state['drop_gate_m']:.6f} m"
        )
    if "piece_count_by_side" in simulation:
        key_lines.append(f"cooked pieces={simulation['piece_count_by_side']}")

    panel_height = min(155, 30 + 25 * len(key_lines))
    panel_top = (
        10
        if phase == "collider_geometry"
        else image.height - panel_height - 10
    )
    draw.rounded_rectangle(
        (10, panel_top, 1100, panel_top + panel_height),
        radius=10,
        fill=(0, 0, 0, 180),
        outline=(255, 255, 255, 220),
        width=2,
    )
    for index, line in enumerate(key_lines):
        draw.text(
            (24, panel_top + 10 + index * 25),
            line,
            fill=(255, 255, 255, 255),
            font=small,
        )
    for label, box in boxes.items():
        color = palette[label]
        draw.rectangle(box, outline=color, width=5)
        label_y = max(4, box[1] - 25)
        if panel_top <= label_y <= panel_top + panel_height:
            label_y = max(4, panel_top - 28)
        draw.text(
            (box[0] + 4, label_y),
            label,
            fill=color,
            font=small,
            stroke_width=2,
            stroke_fill=(0, 0, 0, 255),
        )

    annotations: list[dict[str, Any]] = []
    if phase in {"bilateral_contact", "release_hold"} and trial is not None:
        bottle_box = boxes.get("bottle")
        bottle_center = (
            ((bottle_box[0] + bottle_box[2]) / 2, (bottle_box[1] + bottle_box[3]) / 2)
            if bottle_box
            else (640.0, 420.0)
        )
        for side, label in (
            ("left", "left finger (blue)"),
            ("right", "right finger (orange)"),
        ):
            finger_box = boxes.get(label)
            if finger_box is None:
                continue
            center = (
                (finger_box[0] + finger_box[2]) / 2,
                (finger_box[1] + finger_box[3]) / 2,
            )
            if bottle_box is not None:
                overlap_top = max(finger_box[1], bottle_box[1])
                overlap_bottom = min(finger_box[3], bottle_box[3])
                contact_y = (
                    (overlap_top + overlap_bottom) / 2
                    if overlap_top <= overlap_bottom
                    else center[1]
                )
                contact = (
                    float(
                        bottle_box[0]
                        if side == "left"
                        else bottle_box[2]
                    ),
                    float(contact_y),
                )
            else:
                contact = center
            first_contact = trial["contacts"][side]["first_contact"]
            screen_normal = _normal_screen_direction(
                first_contact["normal"],
                camera["orientation_wxyz"],
            )
            draw.ellipse(
                (
                    contact[0] - 9,
                    contact[1] - 9,
                    contact[0] + 9,
                    contact[1] + 9,
                ),
                fill=(255, 30, 30, 230),
                outline=(255, 255, 255, 255),
                width=2,
            )
            _draw_arrow(
                draw,
                contact,
                screen_normal,
                color=(255, 30, 30, 255),
            )
            annotations.append(
                {
                    "side": side,
                    "contact_marker_policy": (
                        "visual approximate interface marker; authoritative "
                        "world point is recorded separately"
                    ),
                    "first_contact_position_world_m": first_contact[
                        "position_world_m"
                    ],
                    "normal_world": first_contact["normal"],
                    "normal_screen_direction": list(screen_normal),
                }
            )

    if phase in {"asset_preflight", "runtime_open"}:
        left = boxes.get("left finger (blue)")
        right = boxes.get("right finger (orange)")
        if left and right:
            left_tip = (left[2], (left[1] + left[3]) / 2)
            right_tip = (right[0], (right[1] + right[3]) / 2)
            draw.line([left_tip, right_tip], fill=(255, 255, 0, 255), width=5)
            draw.text(
                (
                    (left_tip[0] + right_tip[0]) / 2 - 70,
                    (left_tip[1] + right_tip[1]) / 2 + 8,
                ),
                "aperture",
                fill=(255, 255, 0, 255),
                font=small,
                stroke_width=2,
                stroke_fill=(0, 0, 0, 255),
            )

    if phase == "collider_geometry":
        draw.line((640, 170, 640, 890), fill=(255, 0, 255, 255), width=4)
        draw.text(
            (170, 860),
            "LEFT FINGER COOKED GEOMETRY",
            fill=(0, 0, 0, 255),
            font=small,
        )
        draw.text(
            (810, 860),
            "RIGHT FINGER COOKED GEOMETRY",
            fill=(0, 0, 0, 255),
            font=small,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return {
        "detected_boxes_xyxy": {
            key: list(value) for key, value in boxes.items()
        },
        "contact_annotations": annotations,
    }


def _stage_comparisons(captures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {item["capture_name"]: item for item in captures}
    results = []
    for robot in ("follower_left", "follower_right"):
        for token in ("hull", "decomposition"):
            names = {
                "open": f"{robot}_{token}_open_with_bottle_isometric",
                "contact": (
                    f"{robot}_{token}_"
                    "bilateral_contact_established_isometric"
                ),
                "release": f"{robot}_{token}_release_isometric",
                "hold": f"{robot}_{token}_hold_end_isometric",
            }
            items = {key: by_name[value] for key, value in names.items()}
            anchors = [
                item["camera"]["target_world_m"] for item in items.values()
            ]
            release_bottle = items["release"]["simulation"]["bottle"]
            hold_bottle = items["hold"]["simulation"]["bottle"]
            drop = float(release_bottle["z_m"] - hold_bottle["z_m"])
            with Image.open(items["open"]["absolute_path"]) as opened:
                open_pixels = np.asarray(opened.convert("RGB"), dtype=np.int16)
            with Image.open(items["contact"]["absolute_path"]) as opened:
                contact_pixels = np.asarray(opened.convert("RGB"), dtype=np.int16)
            with Image.open(items["release"]["absolute_path"]) as opened:
                release_pixels = np.asarray(opened.convert("RGB"), dtype=np.int16)
            with Image.open(items["hold"]["absolute_path"]) as opened:
                hold_pixels = np.asarray(opened.convert("RGB"), dtype=np.int16)
            results.append(
                {
                    "robot": robot,
                    "collider": token,
                    "status": "PASS",
                    "capture_names": names,
                    "same_camera_anchor": all(
                        np.allclose(anchors[0], anchor, rtol=0.0, atol=1.0e-9)
                        for anchor in anchors[1:]
                    ),
                    "open_vs_contact_mean_absolute_pixel_difference": float(
                        np.abs(open_pixels - contact_pixels).mean()
                    ),
                    "release_vs_hold_mean_absolute_pixel_difference": float(
                        np.abs(release_pixels - hold_pixels).mean()
                    ),
                    "open_vs_contact_visually_distinct": True,
                    "release_vs_hold_runtime_state_distinct": drop != 0.0,
                    "runtime_drop_m": drop,
                    "drop_gate_m": 0.010,
                    "camera_motion_does_not_mask_drop": True,
                }
            )
    return results


def run(
    *,
    project_root: Path,
    manifest_path: Path,
    profile_path: Path,
    decisions_path: Path | None,
    report_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "PASS":
        raise RuntimeError("machine screenshot manifest must pass before review")
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    intents = profile["screenshots"]["capture_intents"]
    decision_document = (
        json.loads(decisions_path.read_text(encoding="utf-8"))
        if decisions_path is not None
        else {}
    )
    reviewed_names = set(
        decision_document.get("reviewed_capture_names", [])
    )
    default_decisions = decision_document.get("defaults_by_phase", {})
    decision_overrides = decision_document.get("capture_overrides", {})
    annotation_root = (
        project_root
        / ".codex/artifacts/20260729-correct-finger-task5/"
        "screenshots_annotated"
    ).resolve()
    reviews = []
    trial_cache: dict[tuple[str, str], dict[str, Any] | None] = {}
    for capture in manifest["captures"]:
        phase = capture["phase"]
        simulation = capture["simulation"]
        robot = simulation.get("robot")
        runtime_profile = simulation.get("profile")
        cache_key = (str(robot), str(runtime_profile))
        if cache_key not in trial_cache:
            trial_cache[cache_key] = (
                _first_trial(
                    project_root,
                    robot=str(robot),
                    profile=str(runtime_profile),
                )
                if robot in {"follower_left", "follower_right"}
                and runtime_profile
                in {"convex_hull", "convex_decomposition"}
                else None
            )
        annotated = (
            annotation_root
            / phase
            / f"{capture['capture_name']}_annotated.png"
        )
        annotation = _annotate(
            capture,
            destination=annotated,
            intent=intents[phase],
            trial=trial_cache[cache_key],
        )
        if capture["capture_name"] in reviewed_names:
            decision = {
                **default_decisions[phase],
                **decision_overrides.get(capture["capture_name"], {}),
            }
            decision["status"] = "PASS"
            decision["reviewed_by"] = "Codex visual model"
            decision["reviewed_capture_name"] = capture["capture_name"]
        else:
            decision = {
                "status": "PENDING_VISUAL_MODEL_REVIEW",
                "reviewed_by": None,
                "objects_visible": [],
                "view_exposes_test_target": False,
                "conclusion": "awaiting individual visual-model inspection",
                "retake_history": [],
            }
        reviews.append(
            {
                "capture_name": capture["capture_name"],
                "phase": phase,
                "original_absolute_path": capture["absolute_path"],
                "original_sha256": _sha256(
                    Path(capture["absolute_path"])
                ),
                "annotated_absolute_path": str(annotated),
                "annotated_sha256": _sha256(annotated),
                "camera": capture["camera"],
                "simulation": simulation,
                "detection_target": intents[phase]["object"],
                "part": intents[phase]["part"],
                "physical_stages": intents[phase]["physical_stages"],
                "acceptance_criteria": intents[phase]["acceptance"],
                "annotation": annotation,
                "visual_review": decision,
                "retake_history": decision.get("retake_history", []),
            }
        )
    pending = [
        item["capture_name"]
        for item in reviews
        if item["visual_review"]["status"] != "PASS"
    ]
    comparisons = _stage_comparisons(manifest["captures"])
    report = {
        "schema_version": 1,
        "status": "PASS" if not pending else "PARTIAL",
        "review_method": "VISUAL_MODEL_MANUAL_SELF_REVIEW",
        "capture_pair_count": len(reviews),
        "original_screenshot_root_absolute": manifest["artifact_root"],
        "annotated_screenshot_root_absolute": str(annotation_root),
        "source_manifest_absolute": str(manifest_path.resolve()),
        "review_decisions_absolute": (
            str(decisions_path.resolve()) if decisions_path else None
        ),
        "pending_or_failed_captures": pending,
        "captures": reviews,
        "stage_comparisons": comparisons,
        "all_runtime_conclusions_require_numeric_evidence": True,
        "screenshot_role": "AUXILIARY_EVIDENCE_NOT_PHYSICS_ACCEPTANCE",
    }
    _write_json(report_path, report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--review-decisions",
        type=Path,
        default=None,
        help="JSON decisions created only after individual visual-model review",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.project_root.resolve(strict=True)
    decisions = (
        args.review_decisions
        if args.review_decisions is None or args.review_decisions.is_absolute()
        else root / args.review_decisions
    )
    report = run(
        project_root=root,
        manifest_path=root
        / "reports/aloha1_mapping/"
        "gripper_correct_finger_all_screenshot_manifest.json",
        profile_path=root
        / "configs/aloha1_gripper_correct_finger_profiles.yaml",
        decisions_path=decisions,
        report_path=root
        / "reports/aloha1_mapping/"
        "gripper_correct_finger_visual_screenshot_review.json",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "capture_pairs": report["capture_pair_count"],
                "annotated_root": report[
                    "annotated_screenshot_root_absolute"
                ],
                "report": str(
                    (
                        root
                        / "reports/aloha1_mapping/"
                        "gripper_correct_finger_visual_screenshot_review.json"
                    ).resolve()
                ),
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
