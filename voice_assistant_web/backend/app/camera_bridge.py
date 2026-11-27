from __future__ import annotations

import base64
import logging
import os
from collections import deque
import threading
import time

import cv2
import numpy as np

from .config import settings
from .lossless_preview import encode_preview
from .frame_exposure import FrameExposureCache


class CameraBridge:
    camera_names = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_raw_bgr: dict[str, np.ndarray] = {}
        self._latest_timestamps: dict[str, float] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._error: str | None = None
        self._preview_packet = None
        self._preview_pngs = {}
        self._preview_header = {}
        self._preview_times = deque(maxlen=120)
        self._encoder_error = None
        self._preview_thread = None
        self._stop_event = threading.Event()
        self.frame_exposure = FrameExposureCache()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._stop_event.clear()
        self._preview_thread = threading.Thread(target=self._encode_preview, daemon=True, name='web-png-preview')
        self._preview_thread.start()

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._preview_thread:
            self._preview_thread.join(timeout=1.0)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        try:
            import rospy
            from aloha.msg import RGBGrayscaleImage
            from std_msgs.msg import String

            if not rospy.core.is_initialized():
                rospy.init_node("voice_assistant_web_backend", anonymous=True)

            def image_callback(camera_name: str):
                def _callback(message: RGBGrayscaleImage) -> None:
                    if not message.images:
                        return
                    frame = self._image_msg_to_bgr(message.images[0])
                    if frame is None:
                        return
                    with self._lock:
                        self._latest_raw_bgr[camera_name] = frame.copy()
                        self._latest_timestamps[camera_name] = time.time()

                return _callback

            subscribers = [
                rospy.Subscriber(f"/{camera_name}", RGBGrayscaleImage, image_callback(camera_name), queue_size=1, buff_size=4194304)
                for camera_name in self.camera_names
            ]
            subscribers.append(rospy.Subscriber('/camera_exposure_metadata', String,
                lambda message: self.frame_exposure.update(message.data), queue_size=1, buff_size=32768))

            while self._running and not rospy.is_shutdown():
                time.sleep(0.1)

            for subscriber in subscribers:
                subscriber.unregister()
        except Exception as exc:
            self._error = str(exc)
            logging.exception("Camera bridge failed")

    def _encode_preview(self):
        # Only this web-preview thread changes priority; model/control threads are untouched.
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)
        except (OSError, AttributeError):
            pass
        cv2.setNumThreads(1)  # This process is the web backend, never the policy process.
        sequence = 0
        while self._running:
            started = time.monotonic()
            try:
                with self._lock:
                    frames = {name: frame.copy() for name, frame in self._latest_raw_bgr.items()}
                    stamps = dict(self._latest_timestamps)
                if set(frames) == set(self.camera_names):
                    sequence += 1
                    packet, images, header = encode_preview(frames, stamps, sequence)
                    with self._lock:
                        # One latest packet only, shared by every preview viewer.
                        self._preview_packet, self._preview_pngs, self._preview_header = packet, images, header
                        self._preview_times.append(time.monotonic())
                        self._encoder_error = None
            except Exception as exc:
                self._encoder_error = str(exc)
                logging.exception('Lossless web preview failed')
            self._stop_event.wait(max(0., .1 - (time.monotonic() - started)))

    def get_preview(self):
        with self._lock:
            return self._preview_header.get('sequence'), self._preview_packet

    def get_latest_png(self, camera_name):
        with self._lock:
            return self._preview_pngs.get(camera_name)

    def get_preview_metrics(self):
        with self._lock:
            times = list(self._preview_times)
            return {'server_unix': time.time(), 'codec': 'png', 'width': 640, 'height': 480,
                    'target_hz': 10, 'encoder_hz': ((len(times)-1)/(times[-1]-times[0]) if len(times)>1 else 0),
                    'last_packet_bytes': len(self._preview_packet or b''),
                    'last_header': self._preview_header, 'error': self._encoder_error,
                    'queue_limit_packets': 1, 'encoder_nice': 10}

    def fresh_frames(self, camera_names, max_age: float = .5):
        with self._lock:
            now = time.time()
            return {name: self._latest_raw_bgr[name].copy() for name in camera_names
                    if name in self._latest_raw_bgr and now - self._latest_timestamps.get(name, 0.) <= max_age}

    def get_latest_jpeg(self, camera_name: str) -> bytes | None:
        # Legacy JPEG URLs remain available on demand; live web preview uses PNG only.
        with self._lock:
            frame = self._latest_raw_bgr.get(camera_name)
            frame = None if frame is None else frame.copy()
        if frame is None:
            return None
        ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, max(10,min(settings.camera_jpeg_quality,95))])
        return encoded.tobytes() if ok else None

    def snapshot_for_high_level(self) -> dict:
        # Encode on demand at1Hz; retain original pixels, no JPEG recompression or resize.
        with self._lock:
            frames = {name: frame.copy() for name, frame in self._latest_raw_bgr.items()}
            timestamps = dict(self._latest_timestamps)
        encoded = {}
        for name, frame in frames.items():
            ok, png = cv2.imencode('.png', frame)
            if not ok:
                raise ValueError('Could not encode camera PNG')
            encoded[name] = base64.b64encode(png).decode('ascii')
        return {'camera_png_b64': encoded, 'camera_timestamps': timestamps, 'captured_unix': time.time()}

    def get_camera_status(self) -> dict[str, bool]:
        with self._lock:
            return {name: name in self._latest_raw_bgr for name in self.camera_names}

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
