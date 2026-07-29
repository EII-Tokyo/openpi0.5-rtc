#!/usr/bin/env python3
"""Build continuous two-view videos for the horizontal Bottle500 trial."""

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

VIEWS = ("overview", "gripper_closeup")
REQUIRED_PHASES = (
    "release_dynamic",
    "support_settle",
    "open_pregrasp",
    "vertical_descent",
    "bilateral_contact",
    "closing_preload",
    "vertical_lift",
    "support_clear",
    "hold_end",
)
FPS = 60


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _records(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = manifest.get("records", manifest.get("frames"))
    if not isinstance(records, list) or not records:
        raise ValueError("frame manifest has no records")
    return records


def validate_frame_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a complete, synchronized, two-view physics-frame stream."""
    records = _records(manifest)
    frame_numbers = [int(record["physics_frame"]) for record in records]
    first = min(frame_numbers)
    last = max(frame_numbers)
    expected = list(range(first, last + 1))
    if frame_numbers != expected:
        missing = sorted(set(expected) - set(frame_numbers))
        raise ValueError(f"non-contiguous physics frames: {missing}")
    phase_ranges: dict[str, list[int]] = {}
    for record in records:
        frame = int(record["physics_frame"])
        phase = str(record["phase"])
        views = record.get("views", {})
        for view in VIEWS:
            if view not in views:
                raise ValueError(f"frame {frame} missing view {view}")
            image = Path(views[view]["absolute_path"])
            if not image.is_file():
                raise ValueError(f"frame {frame}/{view} image missing: {image}")
            if _sha256(image) != views[view]["sha256"]:
                raise ValueError(f"frame {frame}/{view} hash mismatch")
            if list(views[view]["resolution"]) != [960, 540]:
                raise ValueError(f"frame {frame}/{view} resolution mismatch")
        if phase not in phase_ranges:
            phase_ranges[phase] = [frame, frame]
        else:
            phase_ranges[phase][1] = frame
    missing_phases = sorted(set(REQUIRED_PHASES) - set(phase_ranges))
    if missing_phases:
        raise ValueError(f"required phases missing: {missing_phases}")
    return {
        "first_physics_frame": first,
        "last_physics_frame": last,
        "missing_physics_frames": [],
        "frame_count": len(records),
        "views": list(VIEWS),
        "phase_frame_ranges": phase_ranges,
        "runtime_trial_signature": str(manifest["runtime_trial_signature"]),
    }


def select_review_frames(
    *,
    first_frame: int,
    last_frame: int,
    phase_frame_ranges: Mapping[str, Sequence[int]],
    max_interval_frames: int = 30,
) -> list[int]:
    """Return phase boundaries plus uniform samples no farther than 0.5 s."""
    if max_interval_frames <= 0:
        raise ValueError("max_interval_frames must be positive")
    samples = set(range(first_frame, last_frame + 1, max_interval_frames))
    samples.update((first_frame, last_frame))
    for phase in REQUIRED_PHASES:
        if phase not in phase_frame_ranges:
            raise ValueError(f"missing phase range: {phase}")
        start, end = (int(value) for value in phase_frame_ranges[phase])
        samples.update((start, end))
    ordered = sorted(samples)
    filled: list[int] = [ordered[0]]
    for target in ordered[1:]:
        while target - filled[-1] > max_interval_frames:
            filled.append(filled[-1] + max_interval_frames)
        filled.append(target)
    return filled


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
    if font_path.is_file():
        return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def _contact_frames(trial: Mapping[str, Any], side: str) -> set[int]:
    contacts = trial.get("contacts", {}).get(f"{side}_physical", [])
    return {int(item["frame"]) for item in contacts}


def _annotate_frames(
    *,
    source_records: Sequence[Mapping[str, Any]],
    trial: Mapping[str, Any],
    view: str,
    output_dir: Path,
) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    telemetry = {int(item["frame"]): item for item in trial.get("telemetry", [])}
    left_frames = _contact_frames(trial, "left")
    right_frames = _contact_frames(trial, "right")
    signature = str(trial["runtime_trial_signature"])
    physical_status = str(trial["physical_trial_status"])
    font = _font(18)
    small = _font(15)
    annotated_size = (0, 0)
    for record in source_records:
        frame = int(record["physics_frame"])
        raw_path = Path(record["views"][view]["absolute_path"])
        with Image.open(raw_path) as source:
            rgb = source.convert("RGB")
        panel_height = 96
        annotated = Image.new("RGB", (rgb.width, rgb.height + panel_height), (20, 20, 20))
        annotated.paste(rgb, (0, 0))
        draw = ImageDraw.Draw(annotated)
        item = telemetry.get(frame, {})
        bottle = item.get("bottle", {})
        position = bottle.get("position_world_m", [float("nan")] * 3)
        clearance = bottle.get("bottom_clearance_m", float("nan"))
        phase = str(record["phase"])
        status_text = f"PHYSICAL {physical_status}: {trial.get('failure_mode', 'none')}"
        draw.text((12, rgb.height + 8), status_text, fill=(255, 90, 90), font=font)
        draw.text(
            (12, rgb.height + 36),
            (f"{view} | frame {frame:03d} | t={float(record['time_s']):.3f}s | {phase}"),
            fill=(245, 245, 245),
            font=small,
        )
        draw.text(
            (12, rgb.height + 60),
            (
                f"bottle_z={float(position[2]):+.5f} m | "
                f"support_clearance={float(clearance):+.5f} m | "
                f"contact L/R={'Y' if frame in left_frames else 'N'}/"
                f"{'Y' if frame in right_frames else 'N'} | "
                f"sig={signature[:12]}"
            ),
            fill=(220, 220, 220),
            font=small,
        )
        destination = output_dir / f"{frame:06d}.png"
        annotated.save(destination)
        annotated_size = annotated.size
    return annotated_size


def _ffmpeg_version() -> str:
    completed = subprocess.run(
        ["ffmpeg", "-version"],
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.splitlines()[0]


def _encode(
    *,
    frames_dir: Path,
    frame_count: int,
    destination: Path,
    log_path: Path,
) -> tuple[list[str], dict[str, Any]]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info",
        "-y",
        "-framerate",
        str(FPS),
        "-start_number",
        "0",
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
        raise RuntimeError(f"ffmpeg failed ({completed.returncode}); see {log_path}")
    probe_command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_read_frames,duration",
        "-of",
        "json",
        str(destination),
    ]
    probe = subprocess.run(
        probe_command,
        check=True,
        text=True,
        capture_output=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    observed_frames = int(stream["nb_read_frames"])
    if observed_frames != frame_count:
        raise RuntimeError(f"encoded frame count {observed_frames} != {frame_count}")
    return command, {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "r_frame_rate": stream["r_frame_rate"],
        "frame_count": observed_frames,
        "duration_s": float(stream["duration"]),
    }


def build_candidate_videos(
    *,
    report_path: Path,
    frame_manifest_path: Path,
    trial_index: int,
    attempt_id: str,
    output_root: Path,
) -> dict[str, Any]:
    report = _load_json(report_path)
    trial = report["trials"][trial_index]
    manifest = _load_json(frame_manifest_path)
    validated = validate_frame_manifest(manifest)
    if validated["runtime_trial_signature"] != trial["runtime_trial_signature"]:
        raise ValueError("trial and frame-manifest signatures differ")
    source_records = _records(manifest)
    review_frames = select_review_frames(
        first_frame=validated["first_physics_frame"],
        last_frame=validated["last_physics_frame"],
        phase_frame_ranges=validated["phase_frame_ranges"],
        max_interval_frames=30,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    videos = []
    encoder_version = _ffmpeg_version()
    for view in VIEWS:
        annotated_dir = output_root / "annotated_frames" / view
        annotated_resolution = _annotate_frames(
            source_records=source_records,
            trial=trial,
            view=view,
            output_dir=annotated_dir,
        )
        raw_path = (output_root / f"{view}_raw_candidate.mp4").resolve()
        annotated_path = (output_root / f"{view}_annotated_candidate.mp4").resolve()
        raw_frames_dir = Path(source_records[0]["views"][view]["absolute_path"]).parent
        raw_command, raw_probe = _encode(
            frames_dir=raw_frames_dir,
            frame_count=validated["frame_count"],
            destination=raw_path,
            log_path=output_root / f"{view}_raw_ffmpeg.log",
        )
        annotated_command, annotated_probe = _encode(
            frames_dir=annotated_dir,
            frame_count=validated["frame_count"],
            destination=annotated_path,
            log_path=output_root / f"{view}_annotated_ffmpeg.log",
        )
        if raw_probe["r_frame_rate"] != "60/1":
            raise RuntimeError(f"{view} raw video is not 60 fps")
        if annotated_probe["r_frame_rate"] != "60/1":
            raise RuntimeError(f"{view} annotated video is not 60 fps")
        camera = trial["video_capture"]["views"][view]
        videos.append(
            {
                "attempt_id": attempt_id,
                "view_name": view,
                "runtime_trial_signature": trial["runtime_trial_signature"],
                "raw_candidate_absolute_path": str(raw_path),
                "raw_candidate_sha256": _sha256(raw_path),
                "annotated_candidate_absolute_path": str(annotated_path),
                "annotated_candidate_sha256": _sha256(annotated_path),
                "verified_raw_absolute_path": None,
                "verified_raw_sha256": None,
                "verified_annotated_absolute_path": None,
                "verified_annotated_sha256": None,
                "resolution": [raw_probe["width"], raw_probe["height"]],
                "annotated_resolution": list(annotated_resolution),
                "fps": FPS,
                "frame_count": validated["frame_count"],
                "duration_s": raw_probe["duration_s"],
                "first_physics_frame": validated["first_physics_frame"],
                "last_physics_frame": validated["last_physics_frame"],
                "missing_physics_frames": [],
                "camera_world_matrix": camera["camera_world_matrix"],
                "phase_frame_ranges": validated["phase_frame_ranges"],
                "source_frame_manifest_sha256": _sha256(frame_manifest_path),
                "encoder_name": "ffmpeg/libx264",
                "encoder_version": encoder_version,
                "encoder_command": {
                    "raw": shlex.join(raw_command),
                    "annotated": shlex.join(annotated_command),
                },
                "vision_review_status": "PENDING_VISUAL_MODEL_REVIEW",
                "reviewed_sample_frames": review_frames,
                "retake_reason": None,
                "promotion_status": "NOT_REVIEWED",
                "machine_status": trial["physical_trial_status"],
                "machine_failure_mode": trial.get("failure_mode"),
            }
        )
    candidate = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "source_report_absolute_path": str(report_path.resolve()),
        "source_report_sha256": _sha256(report_path),
        "source_frame_manifest_absolute_path": str(frame_manifest_path.resolve()),
        "source_frame_manifest_sha256": _sha256(frame_manifest_path),
        "runtime_trial_signature": trial["runtime_trial_signature"],
        "physical_trial_status": trial["physical_trial_status"],
        "machine_conclusion": report["conclusion"],
        "visual_evidence_scope": ("COMPLETE_DYNAMIC_TRIAL_CAPTURE_NOT_PHYSICAL_PASS"),
        "videos": videos,
    }
    candidate_path = output_root / "candidate_manifest.json"
    candidate_path.write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return candidate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--frame-manifest", type=Path, required=True)
    parser.add_argument("--trial-index", type=int, default=0)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    candidate = build_candidate_videos(
        report_path=args.report.resolve(strict=True),
        frame_manifest_path=args.frame_manifest.resolve(strict=True),
        trial_index=args.trial_index,
        attempt_id=args.attempt_id,
        output_root=args.output_root.resolve(),
    )
    print(
        json.dumps(
            {
                "attempt_id": candidate["attempt_id"],
                "physical_trial_status": candidate["physical_trial_status"],
                "candidate_manifest": str((args.output_root / "candidate_manifest.json").resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
