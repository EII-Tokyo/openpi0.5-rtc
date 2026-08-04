import threading
import time
from pathlib import Path
import shutil
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from aloha.video_encoding import (
    build_ffmpeg_command,
    encode_cameras_parallel,
    resolve_video_codec,
    validate_rgb_frame,
)


def test_validate_rgb_frame_requires_uint8_hwc_rgb():
    frame = np.zeros((8, 12, 3), dtype=np.uint8)

    validated = validate_rgb_frame(frame, camera_name="cam_high", frame_idx=0)

    assert validated.shape == (8, 12, 3)
    assert validated.dtype == np.uint8
    with pytest.raises(ValueError, match="cam_high.*frame 1.*uint8"):
        validate_rgb_frame(
            frame.astype(np.float32),
            camera_name="cam_high",
            frame_idx=1,
        )
    with pytest.raises(ValueError, match="cam_high.*frame 2.*H, W, 3"):
        validate_rgb_frame(
            np.zeros((8, 12), dtype=np.uint8),
            camera_name="cam_high",
            frame_idx=2,
        )


def test_auto_codec_prefers_nvenc_and_falls_back_to_cpu():
    assert resolve_video_codec("auto", probe_nvenc=lambda: True) == "h264_nvenc"
    assert resolve_video_codec("auto", probe_nvenc=lambda: False) == "libx264"
    assert resolve_video_codec("cpu", probe_nvenc=lambda: True) == "libx264"

    with pytest.raises(RuntimeError, match="NVENC.*unavailable"):
        resolve_video_codec("nvenc", probe_nvenc=lambda: False)


def test_ffmpeg_command_accepts_raw_rgb_and_produces_compatible_mp4(tmp_path):
    output = tmp_path / "cam_high.mp4"

    command = build_ffmpeg_command(
        output,
        width=640,
        height=480,
        fps=50,
        codec="h264_nvenc",
    )

    assert command[:2] == ["ffmpeg", "-hide_banner"]
    assert ["-f", "rawvideo"] == command[
        command.index("-f") : command.index("-f") + 2
    ]
    assert "rgb24" in command
    assert "640x480" in command
    assert "h264_nvenc" in command
    assert "yuv420p" in command
    assert command[-1] == str(output)


def _timesteps(camera_names, frame_count=3):
    result = []
    for value in range(frame_count):
        images = {
            camera_name: np.full(
                (6, 10, 3),
                value,
                dtype=np.uint8,
            )
            for camera_name in camera_names
        }
        result.append(SimpleNamespace(observation={"images": images}))
    return result


def test_four_cameras_are_encoded_concurrently(tmp_path):
    camera_map = {
        "camera_high": "cam_high",
        "camera_low": "cam_low",
        "camera_wrist_left": "cam_left_wrist",
        "camera_wrist_right": "cam_right_wrist",
    }
    lock = threading.Lock()
    active = 0
    max_active = 0
    started = threading.Barrier(4)

    def fake_encoder(path, frames, *, fps, codec, ffmpeg_bin):
        nonlocal active, max_active
        frames = list(frames)
        with lock:
            active += 1
            max_active = max(max_active, active)
        started.wait(timeout=1.0)
        time.sleep(0.02)
        Path(path).write_bytes(b"mp4")
        with lock:
            active -= 1
        return len(frames)

    outputs = encode_cameras_parallel(
        _timesteps(camera_map),
        camera_map,
        tmp_path,
        fps=50,
        backend="cpu",
        encode_camera=fake_encoder,
    )

    assert max_active == 4
    assert set(outputs) == set(camera_map.values())
    assert all(path.is_file() for path in outputs.values())


def test_parallel_failure_removes_all_owned_video_outputs(tmp_path):
    camera_map = {
        "camera_high": "cam_high",
        "camera_low": "cam_low",
    }

    def fake_encoder(path, frames, *, fps, codec, ffmpeg_bin):
        Path(path).write_bytes(b"partial")
        if "cam_low" in str(path):
            raise RuntimeError("injected encoder failure")
        return len(list(frames))

    with pytest.raises(RuntimeError, match="injected encoder failure"):
        encode_cameras_parallel(
            _timesteps(camera_map),
            camera_map,
            tmp_path,
            fps=50,
            backend="cpu",
            encode_camera=fake_encoder,
        )

    assert list(tmp_path.glob("*.mp4")) == []


def test_parallel_encoder_can_read_frames_from_external_source(tmp_path):
    camera_map = {
        "camera_high": "cam_high",
        "camera_low": "cam_low",
    }
    source_values = {
        "camera_high": (1, 2, 3),
        "camera_low": (11, 12, 13),
    }
    seen = {}

    def frame_source(camera_name):
        for value in source_values[camera_name]:
            yield np.full((4, 5, 3), value, dtype=np.uint8)

    def fake_encoder(path, frames, *, fps, codec, ffmpeg_bin):
        seen[Path(path).stem] = [
            int(frame[0, 0, 0])
            for frame in frames
        ]
        Path(path).write_bytes(b"mp4")
        return len(seen[Path(path).stem])

    encode_cameras_parallel(
        [
            SimpleNamespace(observation={"images": {}})
            for _ in range(3)
        ],
        camera_map,
        tmp_path,
        fps=50,
        backend="cpu",
        encode_camera=fake_encoder,
        frame_source=frame_source,
    )

    assert seen == {
        "cam_high": [1, 2, 3],
        "cam_low": [11, 12, 13],
    }


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg unavailable")
def test_real_ffmpeg_encodes_four_readable_cpu_mp4_files(tmp_path):
    camera_map = {
        "camera_high": "cam_high",
        "camera_low": "cam_low",
        "camera_wrist_left": "cam_left_wrist",
        "camera_wrist_right": "cam_right_wrist",
    }

    outputs = encode_cameras_parallel(
        _timesteps(camera_map, frame_count=5),
        camera_map,
        tmp_path,
        fps=50,
        backend="cpu",
    )

    for output in outputs.values():
        capture = cv2.VideoCapture(str(output))
        try:
            assert capture.isOpened()
            frame_count = 0
            while True:
                readable, _frame = capture.read()
                if not readable:
                    break
                frame_count += 1
        finally:
            capture.release()
        assert frame_count == 5
