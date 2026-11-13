"""Stream the collector's raw-RGB -> H.264 MP4 encoding profile without ROS."""

from functools import lru_cache
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np


@lru_cache(maxsize=1)
def nvenc_options():
    """The ROS1 image has older FFmpeg; use its medium preset if p4 is unavailable."""
    help_text = subprocess.run(["ffmpeg", "-hide_banner", "-h", "encoder=h264_nvenc"],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=5).stdout
    preset = "p4" if "p4" in help_text else "medium"
    return ["-preset", preset, "-cq", "23", "-b:v", "0"]


@lru_cache(maxsize=1)
def select_codec():
    """Probe NVENC like aloha/video_encoding.py; fall back explicitly to CPU H.264."""
    try:
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
            "color=size=640x480:rate=1", "-frames:v", "1", "-c:v", "h264_nvenc"
        ] + nvenc_options() + ["-pix_fmt", "yuv420p", "-f", "null", "-"]
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        return "h264_nvenc" if result.returncode == 0 else "libx264"
    except subprocess.TimeoutExpired:
        return "libx264"


class CameraVideo:
    """One FFmpeg process per camera; the recording worker owns all subprocesses."""

    def __init__(self, path, image, fps, codec):
        image = self.prepare(image)
        self.height, self.width = image.shape[:2]
        self.path = Path(path)
        self.frames = 0
        self._error = tempfile.TemporaryFile()
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-threads", "1",
                   "-filter_threads", "1", "-filter_complex_threads", "1", "-f", "rawvideo",
                   "-pix_fmt", "rgb24", "-video_size", f"{self.width}x{self.height}",
                   "-framerate", f"{fps:g}", "-i", "pipe:0", "-an", "-c:v", codec]
        command += nvenc_options() if codec == "h264_nvenc" else ["-preset", "veryfast", "-crf", "23"]
        command += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)]
        try:
            self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._error)
        except Exception:
            self._error.close()
            raise

    @staticmethod
    def prepare(image):
        """Require stable uint8 RGB frames instead of silently changing captured pixels."""
        image = np.asarray(image)
        if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
            image = image.transpose(1, 2, 0)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("Invalid RGB camera frame")
        if image.shape[0] % 2 or image.shape[1] % 2:
            raise ValueError("H.264 yuv420p requires even image dimensions")
        return np.ascontiguousarray(image)

    def write(self, image):
        """Submit one frame; rawvideo gives exactly one encoded frame per submitted image."""
        image = self.prepare(image)
        if image.shape[:2] != (self.height, self.width):
            raise ValueError("Camera resolution changed during recording")
        self._process.stdin.write(image.tobytes())
        self.frames += 1

    def close(self, validate=True):
        """Finish MP4 indexes and verify codec, dimensions, and encoded frame count."""
        try:
            if self._process.stdin is not None:
                try:
                    self._process.stdin.close()
                except BrokenPipeError:
                    pass  # Still reap the failed encoder and report its stderr below.
            try:
                code = self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
                raise RuntimeError(f"FFmpeg finalization timed out: {self.path.name}")
            if code:
                self._error.seek(0)
                raise RuntimeError(f"FFmpeg failed: {self.path.name}: {self._error.read(4096).decode(errors='replace')}")
            if validate:
                capture = cv2.VideoCapture(str(self.path))
                try:
                    ok, image = capture.read()
                    if not ok or image.shape[:2] != (self.height, self.width):
                        raise ValueError(f"Unreadable camera video: {self.path.name}")
                    if int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != self.frames:
                        raise ValueError(f"Video/action frame count mismatch: {self.path.name}")
                finally:
                    capture.release()
        finally:
            self._error.close()
