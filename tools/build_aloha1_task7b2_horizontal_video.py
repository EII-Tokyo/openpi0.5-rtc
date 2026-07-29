#!/usr/bin/env python3
"""Build synchronized source and full-arm composite Bottle500 trial videos."""

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
COMPOSITE_VIEW = "full_arm_composite"
ALL_VIEWS = (*VIEWS, COMPOSITE_VIEW)
LAYOUT = "FULL_ARM_WITH_SYNCHRONIZED_GRIPPER_INSET"
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


def _required_strings(
    manifest: Mapping[str, Any],
    key: str,
) -> list[str]:
    values = manifest.get(key)
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) or not value.strip() for value in values)
    ):
        raise ValueError(f"frame manifest requires non-empty {key}")
    return values


def validate_frame_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a complete, synchronized, two-view physics-frame stream."""
    records = _records(manifest)
    signature = str(manifest["runtime_trial_signature"])
    required_prims = _required_strings(manifest, "required_full_arm_prims")
    required_links = _required_strings(manifest, "required_full_arm_links")
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
            view_record = views[view]
            if int(view_record.get("physics_frame", -1)) != frame:
                raise ValueError(f"frame {frame}/{view} physics_frame mismatch")
            if float(view_record.get("time_s", float("nan"))) != float(record["time_s"]):
                raise ValueError(f"frame {frame}/{view} time_s mismatch")
            if str(view_record.get("runtime_trial_signature")) != signature:
                raise ValueError(f"frame {frame}/{view} runtime_trial_signature mismatch")
            image = Path(view_record["absolute_path"])
            if not image.is_file():
                raise ValueError(f"frame {frame}/{view} image missing: {image}")
            if _sha256(image) != view_record["sha256"]:
                raise ValueError(f"frame {frame}/{view} hash mismatch")
            if list(view_record["resolution"]) != [960, 540]:
                raise ValueError(f"frame {frame}/{view} resolution mismatch")
        framing = views["overview"].get("framing_evidence")
        if not isinstance(framing, Mapping):
            raise ValueError(f"frame {frame} missing overview framing_evidence")
        visible_prims = set(framing.get("visible_prims", []))
        visible_links = set(framing.get("visible_links", []))
        missing_prims = sorted(set(required_prims) - visible_prims)
        missing_links = sorted(set(required_links) - visible_links)
        if missing_prims:
            raise ValueError(f"frame {frame} full-arm framing missing prims: {missing_prims}")
        if missing_links:
            raise ValueError(f"frame {frame} full-arm framing missing links: {missing_links}")
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
        "required_full_arm_prims": required_prims,
        "required_full_arm_links": required_links,
        "phase_frame_ranges": phase_ranges,
        "runtime_trial_signature": signature,
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


def compose_synchronized_frames(
    *,
    source_records: Sequence[Mapping[str, Any]],
    runtime_trial_signature: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Compose full-arm overview and synchronized gripper inset frame pairs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    composite_records: list[dict[str, Any]] = []
    for record in source_records:
        frame = int(record["physics_frame"])
        time_s = float(record["time_s"])
        views = record["views"]
        with Image.open(views["overview"]["absolute_path"]) as image:
            overview = image.convert("RGB")
        with Image.open(views["gripper_closeup"]["absolute_path"]) as image:
            closeup = image.convert("RGB")
        if overview.size != (960, 540) or closeup.size != (960, 540):
            raise ValueError(f"frame {frame} source resolution mismatch")
        inset = ImageOps.fit(
            closeup,
            (480, 540),
            method=Image.Resampling.LANCZOS,
        )
        composite = Image.new("RGB", (1440, 540), (0, 0, 0))
        composite.paste(overview, (0, 0))
        composite.paste(inset, (960, 0))
        destination = (output_dir / f"{frame:06d}.png").resolve()
        composite.save(destination)
        copied = dict(record)
        copied_views = dict(views)
        copied_views[COMPOSITE_VIEW] = {
            "absolute_path": str(destination),
            "sha256": _sha256(destination),
            "resolution": [1440, 540],
            "physics_frame": frame,
            "time_s": time_s,
            "runtime_trial_signature": runtime_trial_signature,
            "layout": LAYOUT,
            "source_views": list(VIEWS),
            "full_arm_width_fraction": 2 / 3,
            "gripper_inset_width_fraction": 1 / 3,
        }
        copied["views"] = copied_views
        composite_records.append(copied)
    return composite_records


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


def validate_encoded_video_pair(
    *,
    view: str,
    expected_frame_count: int,
    raw_probe: Mapping[str, Any],
    annotated_probe: Mapping[str, Any],
) -> None:
    """Fail closed unless raw and annotated encodes are synchronized at 60 fps."""
    if raw_probe["r_frame_rate"] != "60/1":
        raise RuntimeError(f"{view} raw video is not 60 fps")
    if annotated_probe["r_frame_rate"] != "60/1":
        raise RuntimeError(f"{view} annotated video is not 60 fps")
    raw_count = int(raw_probe["frame_count"])
    annotated_count = int(annotated_probe["frame_count"])
    if raw_count != expected_frame_count or annotated_count != expected_frame_count:
        raise RuntimeError(
            f"{view} frame count mismatch: "
            f"raw={raw_count}, annotated={annotated_count}, "
            f"expected={expected_frame_count}"
        )


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
    composite_records = compose_synchronized_frames(
        source_records=source_records,
        runtime_trial_signature=validated["runtime_trial_signature"],
        output_dir=output_root / "composite_frames",
    )
    records_by_view = {
        "overview": source_records,
        "gripper_closeup": source_records,
        COMPOSITE_VIEW: composite_records,
    }
    videos = []
    encoder_version = _ffmpeg_version()
    for view in ALL_VIEWS:
        view_records = records_by_view[view]
        annotated_dir = output_root / "annotated_frames" / view
        annotated_resolution = _annotate_frames(
            source_records=view_records,
            trial=trial,
            view=view,
            output_dir=annotated_dir,
        )
        raw_path = (output_root / f"{view}_raw_candidate.mp4").resolve()
        annotated_path = (output_root / f"{view}_annotated_candidate.mp4").resolve()
        raw_frames_dir = Path(view_records[0]["views"][view]["absolute_path"]).parent
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
        validate_encoded_video_pair(
            view=view,
            expected_frame_count=validated["frame_count"],
            raw_probe=raw_probe,
            annotated_probe=annotated_probe,
        )
        video = {
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
        if view == COMPOSITE_VIEW:
            video.update(
                {
                    "evidence_role": "PRIMARY_FULL_ARM_EVIDENCE",
                    "layout": LAYOUT,
                    "source_views": list(VIEWS),
                    "layout_regions": {
                        "full_arm": {
                            "source_view": "overview",
                            "width_fraction": 2 / 3,
                        },
                        "gripper_inset": {
                            "source_view": "gripper_closeup",
                            "width_fraction": 1 / 3,
                        },
                    },
                    "required_full_arm_prims": validated["required_full_arm_prims"],
                    "required_full_arm_links": validated["required_full_arm_links"],
                    "framing_evidence_input": {
                        "source_frame_manifest_absolute_path": str(frame_manifest_path.resolve()),
                        "source_frame_manifest_sha256": _sha256(frame_manifest_path),
                        "validated_for_every_physics_frame": True,
                    },
                    "source_camera_world_matrices": {
                        source_view: trial["video_capture"]["views"][source_view]["camera_world_matrix"]
                        for source_view in VIEWS
                    },
                }
            )
        else:
            video.update(
                {
                    "evidence_role": "SYNCHRONIZED_SOURCE",
                    "camera_world_matrix": trial["video_capture"]["views"][view]["camera_world_matrix"],
                }
            )
        videos.append(video)
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
        "primary_evidence_view": COMPOSITE_VIEW,
        "layout": LAYOUT,
        "synchronized_source_views": list(VIEWS),
        "required_full_arm_prims": validated["required_full_arm_prims"],
        "required_full_arm_links": validated["required_full_arm_links"],
        "frame_synchronization": [
            {
                "physics_frame": int(record["physics_frame"]),
                "time_s": float(record["time_s"]),
                "runtime_trial_signature": validated["runtime_trial_signature"],
            }
            for record in source_records
        ],
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
