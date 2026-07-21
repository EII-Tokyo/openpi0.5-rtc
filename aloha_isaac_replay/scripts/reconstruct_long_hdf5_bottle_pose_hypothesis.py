from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
from PIL import Image
from PIL import ImageDraw

from aloha_isaac_replay.data.long_video_pose_reconstruction import candidates_to_rows
from aloha_isaac_replay.data.long_video_pose_reconstruction import detect_open_close_lift_candidates


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HDF5 = Path("/home/eii/project/bottles_data/episode_19.hdf5")
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "reports/aloha1_isaac_adaptation/episode19_long_video_pose_reconstruction_20260721"
)
DEFAULT_CAMERAS = ("cam_high", "cam_left_wrist", "cam_low", "cam_right_wrist")
DEFAULT_OFFSETS = (-80, -40, -10, 0, 20, 60, 120)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _trim_jpeg(raw: np.ndarray) -> np.ndarray:
    """Trim a padded embedded JPEG row using the first JPEG EOI marker."""

    eoi = np.flatnonzero((raw[:-1] == 255) & (raw[1:] == 217))
    if len(eoi):
        return raw[: int(eoi[0]) + 2]
    nonzero = np.flatnonzero(raw)
    return raw[: int(nonzero[-1]) + 1] if len(nonzero) else raw


def _decode_frame(h5: h5py.File, *, camera: str, frame: int) -> Image.Image:
    raw = np.asarray(h5[f"observations/images/{camera}"][frame], dtype=np.uint8)
    encoded = _trim_jpeg(raw)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to decode {camera} frame {frame}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)


def _fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.copy()
    image.thumbnail(size)
    canvas = Image.new("RGB", size, (246, 246, 246))
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def _write_contact_sheet(
    h5: h5py.File,
    *,
    qpos: np.ndarray,
    cameras: tuple[str, ...],
    center_frame: int,
    offsets: tuple[int, ...],
    output_path: Path,
    fps: float,
) -> None:
    frames = [min(max(center_frame + offset, 0), len(qpos) - 1) for offset in offsets]
    tile_w, tile_h = 240, 210
    sheet = Image.new("RGB", (tile_w * len(frames), tile_h * len(cameras)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for row, camera in enumerate(cameras):
        for col, frame in enumerate(frames):
            image = _fit_image(_decode_frame(h5, camera=camera, frame=frame), (tile_w, 180))
            x = col * tile_w
            y = row * tile_h
            sheet.paste(image, (x, y + 28))
            draw.text(
                (x + 4, y + 4),
                f"{camera} f={frame} t={frame / fps:.2f}s g={float(qpos[frame, 6]):.3f}",
                fill=(0, 0, 0),
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "rank",
        "pass_signal_gate",
        "score",
        "open_segment_start",
        "open_segment_end",
        "approach_frame",
        "close_frame",
        "grasp_lock_frame",
        "lift_start_frame",
        "lift_confirm_frame",
        "approach_time_s",
        "close_time_s",
        "pre_open_median",
        "close_value",
        "post_close_gripper_median",
        "post_close_left_arm_motion",
        "post_close_left_arm_peak_step_motion",
        "reasons",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            out = {key: row.get(key) for key in fieldnames}
            out["rank"] = rank
            out["reasons"] = ";".join(row.get("reasons") or [])
            writer.writerow(out)


def _write_markdown(
    path: Path,
    *,
    hdf5_path: Path,
    rows: list[dict[str, object]],
    selected: dict[str, object],
    contact_sheets: list[Path],
    fps: float,
) -> None:
    lines = [
        "# Long HDF5 Bottle Pose Hypothesis",
        "",
        f"- HDF5: `{hdf5_path}`",
        f"- fps: `{fps}`",
        "- claim level: `diagnostic_pose_candidate`",
        "- method: detect long-video open -> close -> post-close left-arm motion; validate visually with contact sheets.",
        f"- selected source: `{selected.get('selection_source', 'auto_signal_detector')}`",
        "- not a ground truth pose and not a physics replay PASS by itself.",
        "",
        "## Selected Candidate",
        "",
        f"- approach/open frame: `{selected.get('approach_frame')}`",
        f"- close frame: `{selected.get('close_frame')}`",
        f"- grasp lock frame: `{selected.get('grasp_lock_frame')}`",
        f"- lift start frame: `{selected.get('lift_start_frame')}`",
        f"- lift confirm frame: `{selected.get('lift_confirm_frame')}`",
        f"- recommended validator placement: `hdf5_open_finger_rear_quarter_tabletop`",
        f"- recommended replay window: `{selected.get('approach_frame')}` to `{selected.get('lift_confirm_frame')}`",
        "",
        "The selected window should be used as the first pose-reconstruction seed. The object should still be",
        "created as a dynamic rigid body on the table; any later PASS claim must come from bilateral contact and",
        "lift gates, not from attaching the object to the gripper.",
        "",
        "## Candidate Table",
        "",
        "| rank | pass | score | open seg | approach | close | lock | lift start | lift confirm | post motion | reasons |",
        "| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    str(bool(row.get("pass_signal_gate"))),
                    f"{float(row.get('score') or 0.0):.3f}",
                    f"{row.get('open_segment_start')}-{row.get('open_segment_end')}",
                    str(row.get("approach_frame")),
                    str(row.get("close_frame")),
                    str(row.get("grasp_lock_frame")),
                    str(row.get("lift_start_frame")),
                    str(row.get("lift_confirm_frame")),
                    f"{float(row.get('post_close_left_arm_motion') or 0.0):.3f}",
                    "; ".join(row.get("reasons") or []),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Contact Sheets", ""])
    for sheet in contact_sheets:
        lines.append(f"- `{sheet}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reconstruct_long_hdf5_bottle_pose_hypothesis(
    *,
    hdf5_path: Path,
    output_dir: Path,
    selected_close_frame: int | None,
    manual_approach_frame: int | None = None,
    manual_close_frame: int | None = None,
    manual_grasp_lock_frame: int | None = None,
    manual_lift_start_frame: int | None = None,
    manual_lift_confirm_frame: int | None = None,
    manual_note: str | None = None,
    fps: float,
    cameras: tuple[str, ...],
    offsets: tuple[int, ...],
) -> dict[str, Any]:
    with h5py.File(hdf5_path, "r") as h5:
        qpos = np.asarray(h5["observations/qpos"], dtype=np.float64)
        candidates = detect_open_close_lift_candidates(qpos, fps=fps)
        rows = candidates_to_rows(candidates, fps=fps)
        if not rows and manual_approach_frame is None:
            raise RuntimeError("no open-close-lift candidates detected")
        selected = None
        if manual_approach_frame is not None or manual_close_frame is not None:
            if manual_approach_frame is None or manual_close_frame is None:
                raise ValueError("--manual-approach-frame and --manual-close-frame must be provided together")
            if manual_lift_confirm_frame is None:
                manual_lift_confirm_frame = manual_close_frame
            for frame_name, frame in {
                "manual_approach_frame": manual_approach_frame,
                "manual_close_frame": manual_close_frame,
                "manual_lift_confirm_frame": manual_lift_confirm_frame,
            }.items():
                if not (0 <= int(frame) < len(qpos)):
                    raise ValueError(f"{frame_name} outside HDF5 range: {frame}")
            motion_start = min(int(manual_close_frame), len(qpos) - 1)
            motion_end = min(int(manual_lift_confirm_frame), len(qpos) - 1)
            step_motion = np.linalg.norm(np.diff(qpos[:, :6], axis=0), axis=1)
            post_motion = (
                float(np.sum(step_motion[motion_start:motion_end])) if motion_end > motion_start else 0.0
            )
            post_peak = (
                float(np.max(step_motion[motion_start:motion_end])) if motion_end > motion_start else 0.0
            )
            selected = {
                "open_segment_start": None,
                "open_segment_end": None,
                "approach_frame": int(manual_approach_frame),
                "close_frame": int(manual_close_frame),
                "grasp_lock_frame": None if manual_grasp_lock_frame is None else int(manual_grasp_lock_frame),
                "lift_start_frame": None if manual_lift_start_frame is None else int(manual_lift_start_frame),
                "lift_confirm_frame": int(manual_lift_confirm_frame),
                "pre_open_median": float(qpos[int(manual_approach_frame), 6]),
                "close_value": float(qpos[int(manual_close_frame), 6]),
                "lock_value": None
                if manual_grasp_lock_frame is None
                else float(qpos[int(manual_grasp_lock_frame), 6]),
                "post_close_gripper_median": float(np.median(qpos[motion_start : motion_end + 1, 6])),
                "post_close_left_arm_motion": post_motion,
                "post_close_left_arm_peak_step_motion": post_peak,
                "score": None,
                "reasons": [],
                "approach_time_s": float(manual_approach_frame / fps),
                "close_time_s": float(manual_close_frame / fps),
                "grasp_lock_time_s": None
                if manual_grasp_lock_frame is None
                else float(manual_grasp_lock_frame / fps),
                "lift_start_time_s": None
                if manual_lift_start_frame is None
                else float(manual_lift_start_frame / fps),
                "lift_confirm_time_s": float(manual_lift_confirm_frame / fps),
                "pass_signal_gate": True,
                "selection_source": "manual_user_confirmed_video_window",
                "manual_note": manual_note or "",
            }
            rows.insert(0, selected)
        elif selected_close_frame is not None:
            for row in rows:
                if int(row["close_frame"]) == int(selected_close_frame):
                    selected = row
                    break
            if selected is None:
                raise ValueError(f"selected close frame {selected_close_frame} was not detected")
        else:
            selected = rows[0]

        output_dir.mkdir(parents=True, exist_ok=True)
        contact_sheets: list[Path] = []
        written_close_frames: set[int] = set()
        for row in rows:
            close_frame = int(row["close_frame"])
            if close_frame in written_close_frames:
                continue
            written_close_frames.add(close_frame)
            sheet_path = output_dir / f"candidate_close_{close_frame}_contact_sheet.jpg"
            _write_contact_sheet(
                h5,
                qpos=qpos,
                cameras=cameras,
                center_frame=close_frame,
                offsets=offsets,
                output_path=sheet_path,
                fps=fps,
            )
            contact_sheets.append(sheet_path)

    csv_path = output_dir / "long_video_pose_candidates.csv"
    json_path = output_dir / "long_video_pose_hypothesis.json"
    md_path = output_dir / "long_video_pose_hypothesis.md"
    _write_csv(csv_path, rows)
    payload = {
        "status": "PASS",
        "claim_level": "diagnostic_pose_candidate",
        "pose_source": "long_hdf5_video_open_close_lift_hypothesis",
        "hdf5_path": str(hdf5_path),
        "fps": float(fps),
        "candidate_count": len(rows),
        "selected": selected,
        "recommended_validator_args": {
            "hdf5_gripper_episode": str(hdf5_path),
            "hdf5_gripper_start_frame": int(selected["approach_frame"]),
            "hdf5_gripper_end_frame": int(selected["lift_confirm_frame"]) + 1,
            "object_placement": "hdf5_open_finger_rear_quarter_tabletop",
            "object_shape": "bottle_usd_cylinder_proxy",
            "targets_modified": False,
            "object_attachment": "none",
        },
        "candidates": rows,
        "contact_sheets": [str(path) for path in contact_sheets],
        "outputs": {
            "csv": str(csv_path),
            "json": str(json_path),
            "markdown": str(md_path),
        },
    }
    json_path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(
        md_path,
        hdf5_path=hdf5_path,
        rows=rows,
        selected=selected,
        contact_sheets=contact_sheets,
        fps=fps,
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Infer a diagnostic bottle pose/replay-window hypothesis from a long raw HDF5 episode. "
            "This does not prove physical replay success."
        )
    )
    parser.add_argument("--hdf5", type=Path, default=DEFAULT_HDF5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--camera", action="append", choices=DEFAULT_CAMERAS)
    parser.add_argument("--frame-offset", type=int, action="append")
    parser.add_argument(
        "--selected-close-frame",
        type=int,
        default=None,
        help="Optional close frame selected after visual review of the generated contact sheets.",
    )
    parser.add_argument(
        "--manual-approach-frame",
        type=int,
        default=None,
        help="User-confirmed frame where the gripper is open and approaching the bottle.",
    )
    parser.add_argument(
        "--manual-close-frame",
        type=int,
        default=None,
        help="User-confirmed frame where the gripper has closed on the bottle.",
    )
    parser.add_argument(
        "--manual-grasp-lock-frame",
        type=int,
        default=None,
        help="Optional user-confirmed frame where the grasp appears locked.",
    )
    parser.add_argument(
        "--manual-lift-start-frame",
        type=int,
        default=None,
        help="Optional user-confirmed frame where the arm starts lifting after grasp.",
    )
    parser.add_argument(
        "--manual-lift-confirm-frame",
        type=int,
        default=None,
        help="Optional user-confirmed frame where lift is visually confirmed.",
    )
    parser.add_argument(
        "--manual-note",
        default=None,
        help="Short note describing the visual evidence for a user-confirmed manual window.",
    )
    args = parser.parse_args()

    payload = reconstruct_long_hdf5_bottle_pose_hypothesis(
        hdf5_path=args.hdf5,
        output_dir=args.output_dir,
        selected_close_frame=args.selected_close_frame,
        manual_approach_frame=args.manual_approach_frame,
        manual_close_frame=args.manual_close_frame,
        manual_grasp_lock_frame=args.manual_grasp_lock_frame,
        manual_lift_start_frame=args.manual_lift_start_frame,
        manual_lift_confirm_frame=args.manual_lift_confirm_frame,
        manual_note=args.manual_note,
        fps=float(args.fps),
        cameras=tuple(args.camera or DEFAULT_CAMERAS),
        offsets=tuple(args.frame_offset or DEFAULT_OFFSETS),
    )
    print(json.dumps({"status": payload["status"], "json": payload["outputs"]["json"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
