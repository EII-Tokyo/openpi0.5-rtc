from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any
import warnings

import numpy as np
from PIL import Image


def _add_local_paths() -> None:
    this_file = Path(__file__).resolve()
    pkg_src = this_file.parents[1] / "src"
    if str(pkg_src) not in sys.path:
        sys.path.insert(0, str(pkg_src))


_add_local_paths()

from topreward.model import TOPRewardModel  # noqa: E402
from topreward.progress import estimate_progress  # noqa: E402
from topreward.success_detector import detect_success  # noqa: E402
from topreward.utils.metrics import compute_voc  # noqa: E402


def _to_pil_rgb(frame: np.ndarray) -> Image.Image:
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.ndim != 3:
        raise ValueError(f"Unsupported frame shape: {array.shape}")

    if array.shape[2] == 4:
        array = array[:, :, :3]
    elif array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] != 3:
        raise ValueError(f"Unsupported channel dimension in frame: {array.shape}")

    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            if float(np.max(array)) <= 1.0:
                array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
            else:
                array = np.clip(array, 0, 255).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)

    return Image.fromarray(array, mode="RGB")


def _decode_video_frames_cv2(
    video_path: Path,
    stride: int,
    max_frames: int | None,
) -> tuple[list[Image.Image], float | None, int | None, int]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video via OpenCV: {video_path}")

    fps_raw = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_frames_meta = frame_count_raw if frame_count_raw > 0 else None

    frames: list[Image.Image] = []
    decoded = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break

        if decoded % stride == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(_to_pil_rgb(rgb))
            if max_frames is not None and len(frames) >= max_frames:
                decoded += 1
                break

        decoded += 1

    cap.release()
    fps = fps_raw if fps_raw > 0 else None
    return frames, fps, total_frames_meta, decoded


def _decode_video_frames_imageio(
    video_path: Path,
    stride: int,
    max_frames: int | None,
) -> tuple[list[Image.Image], float | None, int | None, int]:
    import imageio.v3 as iio

    metadata = iio.immeta(str(video_path))
    fps_raw = metadata.get("fps")
    fps = float(fps_raw) if isinstance(fps_raw, (int, float)) and fps_raw > 0 else None

    total_raw = metadata.get("nframes")
    total_frames_meta = int(total_raw) if isinstance(total_raw, (int, float)) and total_raw > 0 else None

    frames: list[Image.Image] = []
    decoded = 0
    for frame in iio.imiter(str(video_path)):
        if decoded % stride == 0:
            frames.append(_to_pil_rgb(frame))
            if max_frames is not None and len(frames) >= max_frames:
                decoded += 1
                break
        decoded += 1

    return frames, fps, total_frames_meta, decoded


def _decode_video_frames(
    video_path: Path,
    stride: int,
    max_frames: int | None,
) -> tuple[list[Image.Image], float | None, int | None, int, str]:
    errors: list[str] = []

    try:
        frames, fps, total_frames_meta, decoded = _decode_video_frames_cv2(video_path, stride=stride, max_frames=max_frames)
        return frames, fps, total_frames_meta, decoded, "cv2"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"cv2: {exc}")

    try:
        frames, fps, total_frames_meta, decoded = _decode_video_frames_imageio(
            video_path,
            stride=stride,
            max_frames=max_frames,
        )
        return frames, fps, total_frames_meta, decoded, "imageio"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"imageio: {exc}")

    raise RuntimeError(f"Failed to decode video '{video_path}'. Attempts: {' ; '.join(errors)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TOPReward on a local video file")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--prefix_stride_frames", type=int, default=None)
    parser.add_argument("--n_final_frames", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=None)

    parser.add_argument(
        "--score_mode",
        type=str,
        default="true_log_prob",
        choices=["true_log_prob", "true_minus_false"],
        help="Reward scoring mode: logP(True) or margin logP(True)-logP(False).",
    )
    parser.add_argument("--true_token", type=str, default="True")
    parser.add_argument("--false_token", type=str, default="False")

    parser.add_argument("--max_frames", type=int, default=None, help="Cap decoded/sampled frames.")
    parser.add_argument(
        "--max_prefix_frames",
        type=int,
        default=None,
        help="Cap frames shown to the model per forward pass (evenly sampled from the prefix). "
             "Fixes prepare time for long episodes.",
    )
    parser.add_argument(
        "--subsample_fps",
        type=float,
        default=None,
        help="Target FPS for loading frames from the source video.",
    )
    parser.add_argument(
        "--video_fps",
        type=float,
        default=None,
        help="Override source video FPS for temporal metadata (use if auto-detection is wrong).",
    )

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_path = Path(args.video_path).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max_frames must be >= 1")

    # First decode once with stride=1 if we need metadata-derived stride.
    # To avoid two passes, we infer stride from declared/override FPS first.
    source_fps_hint = float(args.video_fps) if args.video_fps is not None else None

    stride = 1
    if args.subsample_fps is not None:
        if args.subsample_fps <= 0:
            raise ValueError("--subsample_fps must be > 0")
        if source_fps_hint is not None:
            stride = max(1, int(round(source_fps_hint / args.subsample_fps)))

    t0 = time.perf_counter()
    frames, detected_fps, total_frames_meta, decoded_frames, decoder = _decode_video_frames(
        video_path,
        stride=stride,
        max_frames=args.max_frames,
    )
    if not frames:
        raise ValueError(f"No frames decoded from video: {video_path}")
    print(f"[timing] video decode: {time.perf_counter() - t0:.2f}s  ({len(frames)} frames, decoder={decoder})")

    source_fps = source_fps_hint if source_fps_hint is not None else detected_fps

    if args.subsample_fps is not None and source_fps is not None and source_fps_hint is None:
        stride = max(1, int(round(source_fps / args.subsample_fps)))
        if stride > 1:
            t0 = time.perf_counter()
            frames, detected_fps, total_frames_meta, decoded_frames, decoder = _decode_video_frames(
                video_path,
                stride=stride,
                max_frames=args.max_frames,
            )
            print(f"[timing] video decode (stride={stride}): {time.perf_counter() - t0:.2f}s  ({len(frames)} frames)")
            source_fps = source_fps_hint if source_fps_hint is not None else detected_fps

    effective_fps: float | None
    if source_fps is not None:
        effective_fps = float(source_fps) / float(max(stride, 1))
    elif args.subsample_fps is not None:
        effective_fps = float(args.subsample_fps)
        warnings.warn(
            "Could not detect source video FPS. Using --subsample_fps as effective model FPS metadata.",
            stacklevel=2,
        )
    else:
        effective_fps = None

    t0 = time.perf_counter()
    model = TOPRewardModel(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    if effective_fps is not None:
        model.video_fps = float(effective_fps)
    print(f"[timing] model load: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    progress = estimate_progress(
        model,
        frames,
        args.instruction,
        k=args.K,
        prefix_stride=args.prefix_stride_frames,
        score_mode=args.score_mode,
        true_token=args.true_token,
        false_token=args.false_token,
        max_prefix_frames=args.max_prefix_frames,
    )
    print(f"[timing] estimate_progress: {time.perf_counter() - t0:.2f}s  ({len(progress['raw_rewards'])} forward passes)")

    t0 = time.perf_counter()
    voc = compute_voc(progress["normalized"], progress["timestamps"])
    print(f"[timing] compute_voc: {time.perf_counter() - t0:.3f}s")

    t0 = time.perf_counter()
    success = detect_success(
        model,
        frames,
        args.instruction,
        n_final_frames=args.n_final_frames,
        threshold=args.threshold,
        score_mode=args.score_mode,
        true_token=args.true_token,
        false_token=args.false_token,
        max_prefix_frames=args.max_prefix_frames,
    )
    print(f"[timing] detect_success: {time.perf_counter() - t0:.2f}s")

    result: dict[str, Any] = {
        "video_path": str(video_path),
        "instruction": args.instruction,
        "score_mode": args.score_mode,
        "true_token": args.true_token,
        "false_token": args.false_token,
        "metadata": {
            "decoder": decoder,
            "source_fps": source_fps,
            "effective_fps": effective_fps,
            "stride": int(stride),
            "decoded_frames": int(decoded_frames),
            "total_frames_meta": total_frames_meta,
            "loaded_frames": len(frames),
        },
        "progress": progress,
        "voc": voc,
        "success": success,
    }

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f"Wrote result to: {output}")


if __name__ == "__main__":
    main()
