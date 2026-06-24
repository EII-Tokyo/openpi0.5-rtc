from __future__ import annotations

import base64
from collections import deque
import logging
import threading
import time
from typing import Any

import cv2
import numpy as np

from .config import settings


class CameraBridge:
    camera_names = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def __init__(self, encode_jpeg: bool = True) -> None:
        self._encode_jpeg = encode_jpeg
        self._lock = threading.Lock()
        self._latest_jpegs: dict[str, bytes] = {}
        self._latest_timestamps: dict[str, float] = {}
        self._stats: dict[str, dict[str, Any]] = {name: self._new_camera_stats() for name in self.camera_names}
        self._running = False
        self._thread: threading.Thread | None = None
        self._error: str | None = None

    def _new_camera_stats(self) -> dict[str, Any]:
        return {
            "raw_frames_total": 0,
            "encoded_frames_total": 0,
            "dropped_frames_total": 0,
            "error_count": 0,
            "last_error": None,
            "last_frame_wall_time": None,
            "last_encode_wall_time": None,
            "last_encoding": None,
            "last_width": None,
            "last_height": None,
            "latest_jpeg_bytes": 0,
            "encode_ms_recent": deque(maxlen=120),
            "source_times_recent": deque(maxlen=120),
            "encoded_times_recent": deque(maxlen=120),
        }

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        try:
            import rospy
            from aloha.msg import RGBGrayscaleImage

            if not rospy.core.is_initialized():
                rospy.init_node("eii_pilot_backend", anonymous=True)
            quality = max(10, min(settings.camera_jpeg_quality, 95))
            encode_args = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

            def image_callback(camera_name: str):
                def _callback(message: RGBGrayscaleImage) -> None:
                    received_at = time.time()
                    self._record_source_frame(camera_name, received_at)
                    if not message.images:
                        self._record_drop(camera_name, "RGBGrayscaleImage contains no images")
                        return
                    self._record_image_metadata(camera_name, message.images[0])
                    if not self._encode_jpeg:
                        with self._lock:
                            self._latest_timestamps[camera_name] = received_at
                        return
                    encode_start = time.perf_counter()
                    frame = self._image_msg_to_bgr(message.images[0])
                    if frame is None:
                        self._record_drop(camera_name, f"Could not decode image encoding={getattr(message.images[0], 'encoding', None)}")
                        return
                    ok, jpeg = cv2.imencode(".jpg", frame, encode_args)
                    if not ok:
                        self._record_drop(camera_name, "cv2.imencode returned false")
                        return
                    encode_ms = (time.perf_counter() - encode_start) * 1000.0
                    jpeg_bytes = jpeg.tobytes()
                    with self._lock:
                        stats = self._stats.setdefault(camera_name, self._new_camera_stats())
                        now = time.time()
                        stats["encoded_frames_total"] += 1
                        stats["last_encode_wall_time"] = now
                        stats["last_encoding"] = getattr(message.images[0], "encoding", None)
                        stats["last_width"] = int(getattr(message.images[0], "width", 0) or 0)
                        stats["last_height"] = int(getattr(message.images[0], "height", 0) or 0)
                        stats["latest_jpeg_bytes"] = len(jpeg_bytes)
                        stats["encode_ms_recent"].append(encode_ms)
                        stats["encoded_times_recent"].append(now)
                        self._latest_jpegs[camera_name] = jpeg_bytes
                        self._latest_timestamps[camera_name] = now

                return _callback

            subscribers = [
                rospy.Subscriber(f"/{camera_name}", RGBGrayscaleImage, image_callback(camera_name))
                for camera_name in self.camera_names
            ]

            while self._running and not rospy.is_shutdown():
                time.sleep(0.1)

            for subscriber in subscribers:
                subscriber.unregister()
        except Exception as exc:
            self._error = str(exc)
            logging.exception("Camera bridge failed")

    def get_latest_jpeg(self, camera_name: str) -> bytes | None:
        with self._lock:
            return self._latest_jpegs.get(camera_name)

    def get_latest_jpeg_with_timestamp(self, camera_name: str) -> tuple[bytes, float] | None:
        with self._lock:
            jpeg = self._latest_jpegs.get(camera_name)
            timestamp = self._latest_timestamps.get(camera_name)
            if jpeg is None or timestamp is None:
                return None
            return jpeg, timestamp

    def get_camera_status(self) -> dict[str, bool]:
        with self._lock:
            return {
                name: bool(self._stats.setdefault(name, self._new_camera_stats())["last_frame_wall_time"])
                for name in self.camera_names
            }

    def get_camera_timestamps(self) -> dict[str, float | None]:
        with self._lock:
            return {name: self._latest_timestamps.get(name) for name in self.camera_names}

    def snapshot_jpeg_b64_all(self) -> dict[str, str]:
        with self._lock:
            return {name: base64.b64encode(jpeg).decode("ascii") for name, jpeg in self._latest_jpegs.items()}

    def get_diagnostics(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            cameras = {}
            for name in self.camera_names:
                stats = self._stats.setdefault(name, self._new_camera_stats())
                encode_samples = list(stats["encode_ms_recent"])
                cameras[name] = {
                    "has_frame": bool(stats["last_frame_wall_time"]),
                    "frame_age_seconds": self._age_seconds(stats["last_frame_wall_time"], now),
                    "source_fps_recent": self._fps_from_times(stats["source_times_recent"]),
                    "encoded_fps_recent": self._fps_from_times(stats["encoded_times_recent"]),
                    "raw_frames_total": stats["raw_frames_total"],
                    "encoded_frames_total": stats["encoded_frames_total"],
                    "dropped_frames_total": stats["dropped_frames_total"],
                    "error_count": stats["error_count"],
                    "last_error": stats["last_error"],
                    "last_encoding": stats["last_encoding"],
                    "last_width": stats["last_width"],
                    "last_height": stats["last_height"],
                    "latest_jpeg_bytes": stats["latest_jpeg_bytes"],
                    "encode_ms_mean_recent": self._mean(encode_samples),
                    "encode_ms_max_recent": max(encode_samples) if encode_samples else None,
                }
            return {
                "bridge_running": self._running,
                "bridge_error": self._error,
                "jpeg_quality": max(10, min(settings.camera_jpeg_quality, 95)),
                "cameras": cameras,
            }

    def _record_source_frame(self, camera_name: str, timestamp: float) -> None:
        with self._lock:
            stats = self._stats.setdefault(camera_name, self._new_camera_stats())
            stats["raw_frames_total"] += 1
            stats["last_frame_wall_time"] = timestamp
            stats["source_times_recent"].append(timestamp)

    def _record_image_metadata(self, camera_name: str, image_msg: Any) -> None:
        with self._lock:
            stats = self._stats.setdefault(camera_name, self._new_camera_stats())
            stats["last_encoding"] = getattr(image_msg, "encoding", None)
            stats["last_width"] = int(getattr(image_msg, "width", 0) or 0)
            stats["last_height"] = int(getattr(image_msg, "height", 0) or 0)

    def _record_drop(self, camera_name: str, reason: str) -> None:
        with self._lock:
            stats = self._stats.setdefault(camera_name, self._new_camera_stats())
            stats["dropped_frames_total"] += 1
            stats["error_count"] += 1
            stats["last_error"] = reason

    @staticmethod
    def _fps_from_times(times: deque[float]) -> float | None:
        if len(times) < 2:
            return None
        duration = times[-1] - times[0]
        if duration <= 0:
            return None
        return (len(times) - 1) / duration

    @staticmethod
    def _age_seconds(timestamp: float | None, now: float) -> float | None:
        if timestamp is None:
            return None
        return max(0.0, now - timestamp)

    @staticmethod
    def _mean(values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) / len(values)

    def _image_msg_to_bgr(self, image_msg) -> np.ndarray | None:
        dtype = np.uint8
        channels_by_encoding = {
            "rgb8": 3,
            "bgr8": 3,
            "rgba8": 4,
            "bgra8": 4,
            "mono8": 1,
        }
        channels = channels_by_encoding.get(image_msg.encoding)
        if channels is None:
            if not self._error:
                self._error = f"Unsupported image encoding: {image_msg.encoding}"
                logging.error(self._error)
            return None

        frame = np.frombuffer(image_msg.data, dtype=dtype)
        expected_size = image_msg.height * image_msg.width * channels
        if frame.size < expected_size:
            logging.warning(
                "Camera frame for %sx%s %s was truncated: got %s, expected %s",
                image_msg.width,
                image_msg.height,
                image_msg.encoding,
                frame.size,
                expected_size,
            )
            return None

        frame = frame[:expected_size].reshape((image_msg.height, image_msg.width, channels))
        if image_msg.encoding == "rgb8":
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if image_msg.encoding == "bgr8":
            # The current Aloha camera publisher labels frames as bgr8 even though the
            # payload has already been channel-swapped into RGB order.
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if image_msg.encoding == "rgba8":
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        if image_msg.encoding == "bgra8":
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if image_msg.encoding == "mono8":
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame
