from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

from .config import settings


class CameraBridge:
    camera_names = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_jpegs: dict[str, bytes] = {}
        self._latest_timestamps: dict[str, float] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._error: str | None = None

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
                rospy.init_node("voice_assistant_web_backend", anonymous=True, disable_signals=True)
            quality = max(10, min(settings.camera_jpeg_quality, 95))
            encode_args = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

            def image_callback(camera_name: str):
                def _callback(message: RGBGrayscaleImage) -> None:
                    if not message.images:
                        return
                    frame = self._image_msg_to_bgr(message.images[0])
                    if frame is None:
                        return
                    ok, jpeg = cv2.imencode(".jpg", frame, encode_args)
                    if not ok:
                        return
                    with self._lock:
                        self._latest_jpegs[camera_name] = jpeg.tobytes()
                        self._latest_timestamps[camera_name] = time.time()

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

    def get_camera_status(self) -> dict[str, bool]:
        with self._lock:
            return {name: name in self._latest_jpegs for name in self.camera_names}

    def get_camera_timestamps(self) -> dict[str, float | None]:
        with self._lock:
            return {name: self._latest_timestamps.get(name) for name in self.camera_names}

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
