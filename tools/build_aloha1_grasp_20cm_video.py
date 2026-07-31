#!/usr/bin/env python3
"""Build and validate full-arm ALOHA 20 cm grasp video evidence."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

VIEWS = ("overview", "gripper_closeup")
REQUIRED_FULL_ARM_LINKS = (
    "base",
    "shoulder",
    "elbow",
    "forearm",
    "wrist",
    "gripper",
)
REQUIRED_PHASES = (
    "RELEASE_DYNAMIC",
    "SETTLE",
    "OPEN_PREGRASP",
    "VERTICAL_DESCENT",
    "BILATERAL_CONTACT",
    "CLOSE_PRELOAD",
    "VERTICAL_LIFT",
    "HEIGHT_REACHED",
    "HOLD",
)
FPS = 60
ANNOTATED_PANEL_HEIGHT = 140


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one exact evidence file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_frame_manifest(
    manifest: Mapping[str, Any],
    *,
    required_phases: Sequence[str] = REQUIRED_PHASES,
) -> dict[str, Any]:
    """Fail closed on missing, desynchronized, or incomplete view frames."""

    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("frame manifest has no frames")
    signature = str(manifest.get("runtime_signature", ""))
    if not signature:
        raise ValueError("frame manifest has no runtime_signature")
    required_links = list(manifest.get("required_full_arm_links", []))
    if required_links != list(REQUIRED_FULL_ARM_LINKS):
        raise ValueError("required full-arm link contract mismatch")
    frame_numbers = [int(record["physics_frame"]) for record in frames]
    expected = list(range(frame_numbers[0], frame_numbers[-1] + 1))
    if frame_numbers != expected:
        missing = sorted(set(expected) - set(frame_numbers))
        raise ValueError(f"non-contiguous physics frames: {missing}")
    phase_ranges: dict[str, list[int]] = {}
    for record in frames:
        frame = int(record["physics_frame"])
        time_s = float(record["time_s"])
        phase = str(record["phase"])
        views = record.get("views")
        if not isinstance(views, Mapping):
            raise ValueError(f"frame {frame} has no views")
        for view in VIEWS:
            if view not in views:
                raise ValueError(f"frame {frame} missing {view}")
            view_record = views[view]
            if int(view_record.get("physics_frame", -1)) != frame:
                raise ValueError(f"frame {frame}/{view} physics_frame mismatch")
            if float(view_record.get("time_s", float("nan"))) != time_s:
                raise ValueError(f"frame {frame}/{view} time_s mismatch")
            if str(view_record.get("runtime_signature")) != signature:
                raise ValueError(
                    f"frame {frame}/{view} runtime_signature mismatch"
                )
            image_path = Path(str(view_record["absolute_path"]))
            if not image_path.is_file():
                raise ValueError(f"frame {frame}/{view} image missing")
            if sha256_file(image_path) != str(view_record["sha256"]):
                raise ValueError(f"frame {frame}/{view} hash mismatch")
            if list(view_record.get("resolution", [])) != [960, 540]:
                raise ValueError(f"frame {frame}/{view} resolution mismatch")
        framing = views["overview"].get("framing_evidence")
        if not isinstance(framing, Mapping):
            raise ValueError(f"frame {frame} missing framing evidence")
        observed_links = set(
            framing.get("required_full_arm_links_in_frame", [])
        )
        missing_links = [
            link for link in required_links if link not in observed_links
        ]
        if missing_links:
            raise ValueError(
                f"frame {frame} overview missing links: {missing_links}"
            )
        if (
            framing.get("occlusion_status")
            != "PENDING_VISUAL_MODEL_REVIEW"
        ):
            raise ValueError(
                f"frame {frame} invalid occlusion review boundary"
            )
        phase_ranges.setdefault(phase, [frame, frame])[1] = frame
    missing_phases = sorted(set(required_phases) - set(phase_ranges))
    if missing_phases:
        raise ValueError(f"required phases missing: {missing_phases}")
    return {
        "first_physics_frame": frame_numbers[0],
        "last_physics_frame": frame_numbers[-1],
        "missing_physics_frames": [],
        "frame_count": len(frames),
        "views": list(VIEWS),
        "required_full_arm_links": required_links,
        "phase_frame_ranges": phase_ranges,
        "runtime_signature": signature,
        "evidence_scope": (
            "COMPLETE_SUCCESS_TRAJECTORY"
            if tuple(required_phases) == REQUIRED_PHASES
            else "TERMINAL_FAILURE_PHASE_PREFIX"
        ),
    }


def required_phases_for_report(
    *,
    report: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> list[str]:
    """Select a fail-closed phase contract for success or failure evidence."""

    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("frame manifest has no frames")
    observed = {
        str(record.get("phase", ""))
        for record in frames
        if isinstance(record, Mapping)
    }
    status = str(report.get("status", "NOT_REPORTED"))
    missing_success = [
        phase for phase in REQUIRED_PHASES if phase not in observed
    ]
    if status == "PASS":
        if missing_success:
            raise ValueError(
                f"PASS report lacks required phases: {missing_success}"
            )
        return list(REQUIRED_PHASES)
    observed_required = [
        phase for phase in REQUIRED_PHASES if phase in observed
    ]
    expected_prefix = list(REQUIRED_PHASES[: len(observed_required)])
    if observed_required != expected_prefix:
        raise ValueError(
            "terminal failure has non-contiguous phase prefix: "
            f"observed={observed_required}, expected={expected_prefix}"
        )
    if not observed_required:
        raise ValueError("terminal failure contains no formal grasp phase")
    return observed_required


def compose_synchronized_frames(
    *,
    source_records: Sequence[Mapping[str, Any]],
    runtime_signature: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Compose a 2/3 full-arm overview with a 1/3 gripper close-up."""

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for record in source_records:
        frame = int(record["physics_frame"])
        time_s = float(record["time_s"])
        views = record["views"]
        for view in VIEWS:
            view_record = views[view]
            if int(view_record["physics_frame"]) != frame:
                raise ValueError(f"frame {frame}/{view} physics_frame mismatch")
            if float(view_record["time_s"]) != time_s:
                raise ValueError(f"frame {frame}/{view} time_s mismatch")
            if str(view_record["runtime_signature"]) != runtime_signature:
                raise ValueError(
                    f"frame {frame}/{view} runtime_signature mismatch"
                )
        with Image.open(views["overview"]["absolute_path"]) as source:
            overview = source.convert("RGB")
        with Image.open(views["gripper_closeup"]["absolute_path"]) as source:
            closeup = source.convert("RGB")
        if overview.size != (960, 540) or closeup.size != (960, 540):
            raise ValueError(f"frame {frame} source resolution mismatch")
        closeup_inset = ImageOps.contain(
            closeup,
            (480, 540),
            method=Image.Resampling.LANCZOS,
        )
        composite = Image.new("RGB", (1440, 540), (0, 0, 0))
        composite.paste(overview, (0, 0))
        inset_y = (540 - closeup_inset.height) // 2
        composite.paste(closeup_inset, (960, inset_y))
        destination = (output_dir / f"{frame:06d}.png").resolve()
        composite.save(destination)
        records.append(
            {
                "physics_frame": frame,
                "time_s": time_s,
                "phase": str(record["phase"]),
                "absolute_path": str(destination),
                "sha256": sha256_file(destination),
                "resolution": [1440, 540],
                "runtime_signature": runtime_signature,
                "source_views": list(VIEWS),
                "layout": (
                    "FULL_ARM_TWO_THIRDS_WITH_UNCROPPED_GRIPPER_INSET"
                ),
                "gripper_inset_resize": (
                    "ASPECT_PRESERVING_CONTAIN_NO_SOURCE_CROP"
                ),
            }
        )
    return records


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
    if font_path.is_file():
        return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def _format_vector(value: Any) -> str:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return "not-reported"
    try:
        return "[" + ", ".join(f"{float(item):+.3f}" for item in value) + "]"
    except (TypeError, ValueError):
        return "not-reported"


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: Sequence[float],
    end: Sequence[float],
    *,
    fill: tuple[int, int, int],
    width: int = 4,
) -> None:
    p0 = tuple(float(value) for value in start)
    p1 = tuple(float(value) for value in end)
    draw.line([p0, p1], fill=fill, width=width)
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length <= 1e-9:
        return
    ux, uy = dx / length, dy / length
    wing = min(16.0, max(8.0, length * 0.12))
    for sign in (-1.0, 1.0):
        endpoint = (
            p1[0] - wing * ux - sign * 0.5 * wing * uy,
            p1[1] - wing * uy + sign * 0.5 * wing * ux,
        )
        draw.line([p1, endpoint], fill=fill, width=width)


def annotate_composite_frames(
    *,
    composite_records: Sequence[Mapping[str, Any]],
    telemetry: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Add a non-overlapping evidence panel below each composite frame."""

    telemetry_by_frame = {
        int(record["physics_frame"]): record for record in telemetry
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    title_font = _font(18)
    text_font = _font(15)
    results: list[dict[str, Any]] = []
    for record in composite_records:
        frame = int(record["physics_frame"])
        sample = telemetry_by_frame.get(frame)
        if sample is None:
            raise ValueError(f"telemetry missing physics frame {frame}")
        if str(sample.get("phase")) != str(record["phase"]):
            raise ValueError(f"telemetry phase mismatch at frame {frame}")
        with Image.open(str(record["absolute_path"])) as source:
            raw = source.convert("RGB")
        annotated = Image.new(
            "RGB",
            (raw.width, raw.height + ANNOTATED_PANEL_HEIGHT),
            (18, 18, 20),
        )
        annotated.paste(raw, (0, 0))
        draw = ImageDraw.Draw(annotated)
        panel_y = raw.height
        machine_status = str(report.get("status", "NOT_REPORTED"))
        status_color = (
            (105, 245, 135)
            if machine_status == "PASS"
            else (255, 115, 105)
        )
        draw.text(
            (12, panel_y + 8),
            (
                f"MACHINE {machine_status}: "
                f"{report.get('reason', 'not-reported')}"
            ),
            fill=status_color,
            font=title_font,
        )
        draw.text(
            (12, panel_y + 36),
            (
                f"frame={frame} t={float(record['time_s']):.3f}s "
                f"phase={record['phase']} | overview=FULL ARM "
                "| inset=GRIPPER+BOTTLE"
            ),
            fill=(245, 245, 245),
            font=text_font,
        )
        draw.text(
            (12, panel_y + 60),
            (
                f"clearance={float(sample.get('clearance_m', float('nan'))):+.4f}m "
                f"(target +0.2000m) | hold_drop="
                f"{float(sample.get('hold_drop_m', 0.0)):+.4f}m | "
                f"vz={float(sample.get('bottle_vertical_velocity_m_s', float('nan'))):+.4f}m/s"
            ),
            fill=(220, 220, 220),
            font=text_font,
        )
        draw.text(
            (12, panel_y + 84),
            (
                "contact geometric L/R="
                f"{bool(sample.get('left_geometric_contact'))}/"
                f"{bool(sample.get('right_geometric_contact'))} "
                "| solver-active L/R="
                f"{bool(sample.get('left_solver_active_contact'))}/"
                f"{bool(sample.get('right_solver_active_contact'))}"
            ),
            fill=(220, 220, 220),
            font=text_font,
        )
        draw.text(
            (12, panel_y + 108),
            (
                f"omega={_format_vector(sample.get('bottle_angular_velocity_rad_s'))} "
                f"| IK={sample.get('ik', {}).get('status', 'not-reported')} "
                f"| sig={str(report.get('deterministic_signature', ''))[:16]}"
            ),
            fill=(220, 220, 220),
            font=text_font,
        )
        destination = (output_dir / f"{frame:06d}.png").resolve()
        annotated.save(destination)
        results.append(
            {
                "physics_frame": frame,
                "time_s": float(record["time_s"]),
                "phase": str(record["phase"]),
                "machine_status": machine_status,
                "absolute_path": str(destination),
                "sha256": sha256_file(destination),
                "resolution": [annotated.width, annotated.height],
                "annotation_region": [
                    0,
                    raw.height,
                    raw.width,
                    ANNOTATED_PANEL_HEIGHT,
                ],
                "source_pixels_unobstructed": True,
            }
        )
    return results


def annotate_collision_evidence(
    *,
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Annotate normal/contact and raw PhysX collider-overlay pairs."""

    collision = manifest.get("collision_evidence")
    if not isinstance(collision, Mapping):
        return {"status": "NOT_REPORTED", "records": []}
    if collision.get("enabled") is False:
        return {
            "status": "NOT_RUN_PRIMARY_CLEAN_VIDEO",
            "enabled": False,
            "purpose": collision.get("purpose"),
            "records": [],
        }
    render_evidence = collision.get("render_evidence", {})
    authored_clone = bool(
        isinstance(render_evidence, Mapping)
        and render_evidence.get("authored_geometry_clone")
    )
    frame_records = {
        int(record["physics_frame"]): record
        for record in manifest["frames"]
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_records: list[dict[str, Any]] = []
    for capture in collision.get("records", []):
        frame = int(capture["physics_frame"])
        view = str(capture["view"])
        frame_view = frame_records[frame]["views"][view]
        pixels = frame_view.get("projection_pixels_xy", {})
        for mode, source_key in (
            ("normal_contact", "normal_absolute_path"),
            ("physics_collider_overlay", "collider_overlay_absolute_path"),
        ):
            source_path = Path(str(capture[source_key])).resolve(strict=True)
            with Image.open(source_path) as opened:
                raw = opened.convert("RGB")
            panel_height = 130
            annotated = Image.new(
                "RGB",
                (raw.width, raw.height + panel_height),
                (18, 18, 20),
            )
            annotated.paste(raw, (0, 0))
            draw = ImageDraw.Draw(annotated)
            if mode == "normal_contact":
                if "bottle_a" in pixels and "bottle_b" in pixels:
                    _arrow(
                        draw,
                        pixels["bottle_a"],
                        pixels["bottle_b"],
                        fill=(0, 245, 245),
                    )
                    draw.text(
                        tuple(pixels["bottle_a"]),
                        "A",
                        fill=(0, 245, 245),
                        font=_font(18),
                    )
                    draw.text(
                        tuple(pixels["bottle_b"]),
                        "B",
                        fill=(0, 245, 245),
                        font=_font(18),
                    )
                for side, color, short in (
                    ("left", (50, 135, 255), "L"),
                    ("right", (255, 145, 35), "R"),
                ):
                    origin = pixels.get(
                        f"{side}_finger_collider_origin"
                    )
                    if origin is not None:
                        x, y = (float(value) for value in origin)
                        draw.ellipse(
                            (x - 9, y - 9, x + 9, y + 9),
                            fill=color,
                        )
                        draw.text(
                            (x, y),
                            short,
                            fill=(255, 255, 255),
                            font=_font(13),
                            anchor="mm",
                        )
                    contact = pixels.get(f"{side}_contact")
                    endpoint = pixels.get(f"{side}_normal_endpoint")
                    if contact is not None and endpoint is not None:
                        _arrow(
                            draw,
                            contact,
                            endpoint,
                            fill=color,
                            width=3,
                        )
            panel_y = raw.height
            draw.text(
                (12, panel_y + 8),
                (
                    f"MACHINE {report['status']}: {report['reason']} | "
                    f"{capture['phase_label']} | {view} | "
                    f"frame={frame} t={float(capture['time_s']):.3f}s"
                ),
                fill=(
                    (105, 245, 135)
                    if report["status"] == "PASS"
                    else (255, 115, 105)
                ),
                font=_font(18),
            )
            draw.text(
                (12, panel_y + 38),
                (
                    "cyan A->B=runtime Bottle500 axis; blue/orange L/R="
                    "finger collider origins (not pad centers)"
                ),
                fill=(235, 235, 235),
                font=_font(14),
            )
            draw.text(
                (12, panel_y + 62),
                (
                    "blue/orange contact arrows=PhysX report normal "
                    "at this physics frame"
                    if mode == "normal_contact"
                    else (
                        (
                            "green transparent geometry=session-only "
                            "authored collider clone + Isaac debug "
                            "display; not cooked-hull readback"
                        )
                        if authored_clone
                        else (
                            "Isaac physics collider display readback; "
                            "no authored clone and no inferred arrow"
                        )
                    )
                ),
                fill=(235, 235, 235),
                font=_font(14),
            )
            draw.text(
                (12, panel_y + 86),
                (
                    f"mode={mode} | Isaac Sim 5.1.0.0 | "
                    "DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING"
                ),
                fill=(255, 210, 90),
                font=_font(14),
            )
            draw.text(
                (12, panel_y + 108),
                "visual evidence only; machine contacts/pose/drop are authoritative; Task 8 NOT_RUN",
                fill=(220, 220, 220),
                font=_font(14),
            )
            destination = (
                output_dir
                / str(capture["phase_label"])
                / f"{view}_{mode}_annotated.png"
            ).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            annotated.save(destination)
            annotated_records.append(
                {
                    "phase_label": str(capture["phase_label"]),
                    "runtime_phase": str(capture["runtime_phase"]),
                    "physics_frame": frame,
                    "time_s": float(capture["time_s"]),
                    "view": view,
                    "mode": mode,
                    "raw_absolute_path": str(source_path),
                    "raw_sha256": sha256_file(source_path),
                    "annotated_absolute_path": str(destination),
                    "annotated_sha256": sha256_file(destination),
                    "raw_resolution": [raw.width, raw.height],
                    "annotated_resolution": [
                        annotated.width,
                        annotated.height,
                    ],
                    "visual_model_review": "NOT_RUN",
                    "retake_reason": None,
                }
            )
    required = list(collision.get("required_phase_labels", []))
    captured = list(collision.get("captured_phase_labels", []))
    return {
        "status": (
            "AWAITING_VISUAL_MODEL_REVIEW"
            if sorted(required) == sorted(captured)
            else "FAIL_MISSING_REQUIRED_PHASE"
        ),
        "setting_path": collision.get("setting_path"),
        "setting_before": collision.get("setting_before"),
        "setting_after": collision.get("setting_after"),
        "render_evidence": render_evidence,
        "required_phase_labels": required,
        "captured_phase_labels": captured,
        "records": annotated_records,
    }


def build_review_contact_sheets(
    *,
    frame_records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    frames_per_sheet: int = 20,
) -> list[dict[str, Any]]:
    """Tile every composite frame exactly once for visual-model review."""

    if frames_per_sheet != 20:
        raise ValueError("review contract requires 20 frames per sheet")
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for sheet_index, start in enumerate(
        range(0, len(frame_records), frames_per_sheet),
        start=1,
    ):
        group = frame_records[start : start + frames_per_sheet]
        sheet = Image.new("RGB", (1440, 675), (12, 12, 12))
        frame_numbers: list[int] = []
        for slot, record in enumerate(group):
            frame = int(record["physics_frame"])
            frame_numbers.append(frame)
            with Image.open(str(record["absolute_path"])) as opened:
                source = opened.convert("RGB")
            thumbnail = ImageOps.fit(
                source,
                (360, 135),
                method=Image.Resampling.LANCZOS,
            )
            tile_draw = ImageDraw.Draw(thumbnail)
            tile_draw.rectangle((0, 0, 92, 22), fill=(0, 0, 0))
            tile_draw.text(
                (5, 3),
                f"frame {frame}",
                fill=(255, 255, 255),
                font=_font(13),
            )
            x = (slot % 4) * 360
            y = (slot // 4) * 135
            sheet.paste(thumbnail, (x, y))
        destination = (
            output_dir
            / (
                f"sheet_{sheet_index:02d}_frames_"
                f"{frame_numbers[0]:06d}_{frame_numbers[-1]:06d}.png"
            )
        ).resolve()
        sheet.save(destination)
        records.append(
            {
                "sheet_index": sheet_index,
                "absolute_path": str(destination),
                "sha256": sha256_file(destination),
                "resolution": [1440, 675],
                "frame_numbers": frame_numbers,
                "frame_count": len(frame_numbers),
                "visual_model_review": "NOT_RUN",
            }
        )
    flattened = [
        frame
        for record in records
        for frame in record["frame_numbers"]
    ]
    expected = [int(record["physics_frame"]) for record in frame_records]
    if flattened != expected:
        raise RuntimeError("contact sheets do not cover frames exactly once")
    return records


def encode_frame_sequence(
    *,
    frames_dir: Path,
    first_frame: int,
    frame_count: int,
    destination: Path,
    log_path: Path,
) -> dict[str, Any]:
    """Encode one exact contiguous PNG sequence and verify the result."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info",
        "-y",
        "-framerate",
        str(FPS),
        "-start_number",
        str(first_frame),
        "-i",
        str(frames_dir / "%06d.png"),
        "-frames:v",
        str(frame_count),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
    )
    log_path.write_text(
        completed.stdout + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({completed.returncode}); see {log_path}"
        )
    probe_command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        (
            "stream=width,height,pix_fmt,r_frame_rate,"
            "nb_read_frames,duration"
        ),
        "-of",
        "json",
        str(destination),
    ]
    probe_result = subprocess.run(
        probe_command,
        check=True,
        text=True,
        capture_output=True,
    )
    stream = json.loads(probe_result.stdout)["streams"][0]
    numerator, denominator = (
        int(value) for value in stream["r_frame_rate"].split("/", maxsplit=1)
    )
    observed_count = int(stream["nb_read_frames"])
    probe = {
        "resolution": [int(stream["width"]), int(stream["height"])],
        "fps": numerator / denominator,
        "r_frame_rate": str(stream["r_frame_rate"]),
        "frame_count": observed_count,
        "duration_s": float(stream["duration"]),
        "pixel_format": str(stream["pix_fmt"]),
    }
    if observed_count != frame_count:
        raise RuntimeError(
            f"encoded frame count {observed_count} != {frame_count}"
        )
    if probe["fps"] != FPS or probe["pixel_format"] != "yuv420p":
        raise RuntimeError(f"encoded stream contract failed: {probe}")
    return {
        "absolute_path": str(destination.resolve()),
        "sha256": sha256_file(destination),
        "command": shlex.join(command),
        "log_absolute_path": str(log_path.resolve()),
        "probe": probe,
    }


def validate_encoded_video_pair(
    *,
    expected_frame_count: int,
    raw_probe: Mapping[str, Any],
    annotated_probe: Mapping[str, Any],
) -> None:
    """Require synchronized 60 fps raw and annotated evidence."""

    for kind, probe in (
        ("raw", raw_probe),
        ("annotated", annotated_probe),
    ):
        if int(probe["frame_count"]) != expected_frame_count:
            raise RuntimeError(f"{kind} video frame count mismatch")
        if float(probe["fps"]) != FPS:
            raise RuntimeError(f"{kind} video is not {FPS} fps")
        if str(probe["pixel_format"]) != "yuv420p":
            raise RuntimeError(f"{kind} video is not yuv420p")


def build_candidate_manifest(
    *,
    report: Mapping[str, Any],
    frame_validation: Mapping[str, Any],
    encoded_videos: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind machine evidence to unpromoted video candidates."""

    signature = str(report.get("deterministic_signature", ""))
    if signature != str(frame_validation.get("runtime_signature", "")):
        raise ValueError("runtime report and frame signature differ")
    videos: list[dict[str, Any]] = []
    for record in encoded_videos:
        path = Path(str(record["absolute_path"]))
        if not path.is_file():
            raise ValueError(f"encoded video missing: {path}")
        if sha256_file(path) != str(record["sha256"]):
            raise ValueError(f"encoded video hash mismatch: {path}")
        if int(record["frame_count"]) != int(
            frame_validation["frame_count"]
        ):
            raise ValueError("encoded video frame count mismatch")
        if int(record["fps"]) != 60:
            raise ValueError("encoded video must be 60 fps")
        videos.append(dict(record))
    return {
        "schema_version": 1,
        "machine_status": str(report["status"]),
        "machine_reason": str(report["reason"]),
        "runtime_signature": signature,
        "classification": (
            "DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING"
        ),
        "frame_validation": dict(frame_validation),
        "videos": videos,
        "promotion_status": (
            "AWAITING_VISUAL_MODEL_REVIEW"
            if str(report["status"]) == "PASS"
            else "MACHINE_FAIL_EVIDENCE_ONLY"
        ),
        "visual_model_review": "NOT_RUN",
        "user_confirmation": "NOT_RUN",
        "task8": "NOT_RUN",
    }


def _annotation_telemetry(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for record in records:
        observation = record.get("observation", {})
        contact = record.get("contact_semantics", {})
        bottle = record.get("bottle", {})
        linear = bottle.get(
            "linear_velocity_world_m_s",
            [float("nan")] * 3,
        )
        flattened.append(
            {
                "physics_frame": int(record["frame"]),
                "time_s": float(record["time_s"]),
                "phase": str(record["phase"]),
                "clearance_m": float(
                    observation.get("clearance_m", float("nan"))
                ),
                "left_geometric_contact": bool(
                    contact.get("left_geometric_contact", False)
                ),
                "right_geometric_contact": bool(
                    contact.get("right_geometric_contact", False)
                ),
                "left_solver_active_contact": bool(
                    contact.get("left_solver_active_contact", False)
                ),
                "right_solver_active_contact": bool(
                    contact.get("right_solver_active_contact", False)
                ),
                "hold_drop_m": float(
                    observation.get("hold_drop_m", 0.0)
                ),
                "bottle_vertical_velocity_m_s": float(linear[2]),
                "bottle_angular_velocity_rad_s": bottle.get(
                    "angular_velocity_world_rad_s",
                    [float("nan")] * 3,
                ),
                "ik": {"status": "PASS"},
            }
        )
    return flattened


def build_video_evidence(
    *,
    report_path: Path,
    telemetry_path: Path,
    frame_manifest_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Build raw/annotated candidates from one terminal physics run."""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(frame_manifest_path.read_text(encoding="utf-8"))
    telemetry = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required_phases = required_phases_for_report(
        report=report,
        manifest=manifest,
    )
    validation = validate_frame_manifest(
        manifest,
        required_phases=required_phases,
    )
    if len(telemetry) != int(validation["frame_count"]):
        raise ValueError("telemetry and video frame count differ")
    if str(report["deterministic_signature"]) != str(
        validation["runtime_signature"]
    ):
        raise ValueError("runtime report and video signature differ")
    output_root.mkdir(parents=True, exist_ok=True)
    composite = compose_synchronized_frames(
        source_records=manifest["frames"],
        runtime_signature=str(report["deterministic_signature"]),
        output_dir=output_root / "composite_raw_frames",
    )
    annotated = annotate_composite_frames(
        composite_records=composite,
        telemetry=_annotation_telemetry(telemetry),
        report=report,
        output_dir=output_root / "composite_annotated_frames",
    )
    review_contact_sheets = build_review_contact_sheets(
        frame_records=composite,
        output_dir=output_root / "review_contact_sheets",
    )
    first_frame = int(validation["first_physics_frame"])
    frame_count = int(validation["frame_count"])
    raw = encode_frame_sequence(
        frames_dir=output_root / "composite_raw_frames",
        first_frame=first_frame,
        frame_count=frame_count,
        destination=output_root / "aloha1_grasp_20cm_raw_candidate.mp4",
        log_path=output_root / "ffmpeg_raw.log",
    )
    marked = encode_frame_sequence(
        frames_dir=output_root / "composite_annotated_frames",
        first_frame=first_frame,
        frame_count=frame_count,
        destination=(
            output_root / "aloha1_grasp_20cm_annotated_candidate.mp4"
        ),
        log_path=output_root / "ffmpeg_annotated.log",
    )
    validate_encoded_video_pair(
        expected_frame_count=frame_count,
        raw_probe=raw["probe"],
        annotated_probe=marked["probe"],
    )
    videos = [
        {
            "kind": kind,
            "absolute_path": record["absolute_path"],
            "sha256": record["sha256"],
            "frame_count": frame_count,
            "fps": FPS,
            "probe": record["probe"],
            "encoder_command": record["command"],
            "encoder_log_absolute_path": record["log_absolute_path"],
        }
        for kind, record in (("raw", raw), ("annotated", marked))
    ]
    collision_evidence = annotate_collision_evidence(
        manifest=manifest,
        report=report,
        output_dir=output_root / "collision_annotated",
    )
    candidate = build_candidate_manifest(
        report=report,
        frame_validation=validation,
        encoded_videos=videos,
    )
    candidate.update(
        {
            "source_report": {
                "absolute_path": str(report_path.resolve()),
                "sha256": sha256_file(report_path),
            },
            "source_telemetry": {
                "absolute_path": str(telemetry_path.resolve()),
                "sha256": sha256_file(telemetry_path),
            },
            "source_frame_manifest": {
                "absolute_path": str(frame_manifest_path.resolve()),
                "sha256": sha256_file(frame_manifest_path),
            },
            "annotated_frame_count": len(annotated),
            "review_contact_sheets": review_contact_sheets,
            "collision_evidence": collision_evidence,
            "primary_evidence_view": "full_arm_composite",
            "layout": (
                "FULL_ARM_TWO_THIRDS_WITH_UNCROPPED_GRIPPER_INSET"
            ),
        }
    )
    candidate_path = output_root / "candidate_manifest.json"
    candidate_path.write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return candidate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--telemetry", required=True, type=Path)
    parser.add_argument("--frame-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    candidate = build_video_evidence(
        report_path=args.report.resolve(strict=True),
        telemetry_path=args.telemetry.resolve(strict=True),
        frame_manifest_path=args.frame_manifest.resolve(strict=True),
        output_root=args.output_root.resolve(),
    )
    print(
        json.dumps(
            {
                "promotion_status": candidate["promotion_status"],
                "runtime_signature": candidate["runtime_signature"],
                "candidate_manifest": str(
                    (args.output_root / "candidate_manifest.json").resolve()
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
