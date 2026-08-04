"""Parallel MP4 encoding with explicit NVENC selection and CPU fallback."""

from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np


def validate_rgb_frame(
    frame: Any,
    *,
    camera_name: str,
    frame_idx: int,
) -> np.ndarray:
    """Return one contiguous RGB frame or reject it with useful context."""
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise ValueError(
            f"{camera_name} frame {frame_idx} must be a uint8 numpy array"
        )
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"{camera_name} frame {frame_idx} must have shape (H, W, 3)"
        )
    if frame.size == 0:
        raise ValueError(f"{camera_name} frame {frame_idx} is empty")
    return np.ascontiguousarray(frame)


@lru_cache(maxsize=4)
def probe_nvenc(ffmpeg_bin: str = "ffmpeg") -> bool:
    """Exercise the encoder instead of trusting the FFmpeg encoder listing."""
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=size=640x480:rate=1",
        "-frames:v",
        "1",
        "-c:v",
        "h264_nvenc",
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def resolve_video_codec(
    backend: str,
    *,
    probe_nvenc: Callable[[], bool] = probe_nvenc,
) -> str:
    """Resolve ``auto``, ``nvenc`` or ``cpu`` to an FFmpeg encoder name."""
    if backend not in {"auto", "nvenc", "cpu"}:
        raise ValueError(f"unsupported video encoder backend: {backend}")
    if backend == "cpu":
        return "libx264"
    available = probe_nvenc()
    if available:
        return "h264_nvenc"
    if backend == "nvenc":
        raise RuntimeError("NVENC was requested but is unavailable")
    return "libx264"


def build_ffmpeg_command(
    output_path: str | os.PathLike[str],
    *,
    width: int,
    height: int,
    fps: float,
    codec: str,
    ffmpeg_bin: str = "ffmpeg",
) -> list[str]:
    """Build a stable raw-RGB-to-H.264 MP4 command."""
    if width <= 0 or height <= 0:
        raise ValueError("video dimensions must be positive")
    if fps <= 0:
        raise ValueError("video fps must be positive")
    if codec not in {"h264_nvenc", "libx264"}:
        raise ValueError(f"unsupported FFmpeg codec: {codec}")

    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{float(fps):g}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        codec,
    ]
    if codec == "h264_nvenc":
        command.extend(["-preset", "p4", "-cq", "23", "-b:v", "0"])
    else:
        command.extend(["-preset", "veryfast", "-crf", "23"])
    command.extend(
        [
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    return command


def encode_rgb_frames(
    output_path: str | os.PathLike[str],
    frames: Iterable[np.ndarray],
    *,
    fps: float,
    codec: str,
    ffmpeg_bin: str = "ffmpeg",
) -> int:
    """Stream one camera's RGB frames to FFmpeg and return the frame count."""
    output = Path(output_path)
    iterator = iter(frames)
    try:
        first = validate_rgb_frame(
            next(iterator),
            camera_name=output.stem,
            frame_idx=0,
        )
    except StopIteration as exc:
        raise ValueError(f"{output.stem} has no frames to encode") from exc

    height, width = first.shape[:2]
    command = build_ffmpeg_command(
        output,
        width=width,
        height=height,
        fps=fps,
        codec=codec,
        ffmpeg_bin=ffmpeg_bin,
    )
    process: subprocess.Popen[bytes] | None = None
    frame_count = 0
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if process.stdin is None or process.stderr is None:
            raise RuntimeError("FFmpeg pipes were not created")

        for frame_idx, frame in enumerate(chain((first,), iterator)):
            validated = validate_rgb_frame(
                frame,
                camera_name=output.stem,
                frame_idx=frame_idx,
            )
            if validated.shape[:2] != (height, width):
                raise ValueError(
                    f"{output.stem} frame {frame_idx} resolution changed "
                    f"from {(height, width)} to {validated.shape[:2]}"
                )
            process.stdin.write(validated.tobytes())
            frame_count += 1

        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"FFmpeg exited with status {return_code} for {output.name}: "
                f"{stderr.strip()[-1000:]}"
            )
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"FFmpeg did not create {output}")
        return frame_count
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        try:
            output.unlink()
        except FileNotFoundError:
            pass
        raise


def encode_cameras_parallel(
    timesteps: Iterable[Any],
    camera_map: Mapping[str, str],
    output_dir: str | os.PathLike[str],
    *,
    fps: float,
    backend: str = "auto",
    ffmpeg_bin: str = "ffmpeg",
    max_workers: int = 4,
    encode_camera: Callable[..., int] = encode_rgb_frames,
    logger: Callable[[str], None] = print,
    frame_source: Callable[[str], Iterable[np.ndarray]] | None = None,
) -> dict[str, Path]:
    """Encode all configured cameras concurrently from immutable timesteps."""
    steps = tuple(timesteps)
    if not steps:
        raise ValueError("episode has no timesteps")
    if not camera_map:
        return {}

    codec = resolve_video_codec(
        backend,
        probe_nvenc=lambda: probe_nvenc(ffmpeg_bin),
    )
    logger(
        f"[视频编码] backend={backend}, codec={codec}, "
        f"camera_workers={min(max_workers, len(camera_map))}"
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        save_name: output_root / f"{save_name}.mp4"
        for save_name in camera_map.values()
    }

    def frames_for(camera_name: str) -> Iterable[np.ndarray]:
        if frame_source is not None:
            for frame_idx, frame in enumerate(frame_source(camera_name)):
                yield validate_rgb_frame(
                    frame,
                    camera_name=camera_name,
                    frame_idx=frame_idx,
                )
            return
        for frame_idx, timestep in enumerate(steps):
            try:
                frame = timestep.observation["images"][camera_name]
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    f"{camera_name} frame {frame_idx} is missing"
                ) from exc
            yield validate_rgb_frame(
                frame,
                camera_name=camera_name,
                frame_idx=frame_idx,
            )

    futures = {}
    try:
        with ThreadPoolExecutor(
            max_workers=min(max_workers, len(camera_map)),
            thread_name_prefix="aloha-camera-encoder",
        ) as executor:
            for camera_name, save_name in camera_map.items():
                future = executor.submit(
                    encode_camera,
                    outputs[save_name],
                    frames_for(camera_name),
                    fps=fps,
                    codec=codec,
                    ffmpeg_bin=ffmpeg_bin,
                )
                futures[future] = camera_name
            for future in as_completed(futures):
                encoded_count = future.result()
                if encoded_count != len(steps):
                    raise RuntimeError(
                        f"{futures[future]} encoded {encoded_count} frames; "
                        f"expected {len(steps)}"
                    )
    except BaseException:
        for path in outputs.values():
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    return outputs
