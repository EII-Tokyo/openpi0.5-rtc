#!/usr/bin/env python3
"""Read-only ROS2 camera bridge for the ALOHA calibration workbench.

The bridge creates subscriptions only.  It never opens a RealSense device,
publishes a ROS message, calls a robot service/action, or changes camera options.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import argparse
import json
import threading
import time
from urllib.parse import parse_qs
from urllib.parse import urlparse

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image


ROLE_TOPIC_ROOTS = {
    "cam_high": "/camera_high/camera/color",
    "cam_low": "/camera_low/camera/color",
    "wrist_left": "/camera_wrist_left/camera/color",
    "wrist_right": "/camera_wrist_right/camera/color",
}


@dataclass(frozen=True)
class CachedFrame:
    rgb: np.ndarray
    stamp_ns: int
    frame_id: str
    received_monotonic: float


class CameraCache(Node):
    def __init__(self) -> None:
        super().__init__("aloha_calibration_camera_bridge")
        self._condition = threading.Condition()
        self._bridge = CvBridge()
        self._camera_info: dict[str, dict[str, object]] = {}
        self._frames: dict[str, CachedFrame] = {}
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        for role, root in ROLE_TOPIC_ROOTS.items():
            self.create_subscription(
                CameraInfo,
                f"{root}/camera_info",
                partial(self._on_camera_info, role),
                qos,
            )
            self.create_subscription(
                Image,
                f"{root}/image_raw",
                partial(self._on_image, role),
                qos,
            )

    def _on_camera_info(self, role: str, message: CameraInfo) -> None:
        stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)
        payload: dict[str, object] = {
            "role": role,
            "width": int(message.width),
            "height": int(message.height),
            "distortion_model": str(message.distortion_model),
            "d": [float(value) for value in message.d],
            "k": [float(value) for value in message.k],
            "frame_id": str(message.header.frame_id),
            "stamp_ns": stamp_ns,
        }
        with self._condition:
            self._camera_info[role] = payload
            self._condition.notify_all()

    def _on_image(self, role: str, message: Image) -> None:
        try:
            rgb = np.asarray(self._bridge.imgmsg_to_cv2(message, desired_encoding="rgb8"), dtype=np.uint8).copy()
        except Exception as exc:  # Keep the ROS executor alive and expose the stale source in /health.
            self.get_logger().error(f"{role} image conversion failed: {type(exc).__name__}")
            return
        stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)
        frame = CachedFrame(
            rgb=rgb,
            stamp_ns=stamp_ns,
            frame_id=str(message.header.frame_id),
            received_monotonic=time.monotonic(),
        )
        with self._condition:
            self._frames[role] = frame
            self._condition.notify_all()

    def camera_info(self, role: str) -> dict[str, object] | None:
        with self._condition:
            payload = self._camera_info.get(role)
            return None if payload is None else dict(payload)

    def wait_for_frame(self, role: str, after_stamp_ns: int, timeout_s: float) -> CachedFrame | None:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                frame = self._frames.get(role)
                if frame is not None and frame.stamp_ns > after_stamp_ns:
                    return CachedFrame(
                        rgb=frame.rgb.copy(),
                        stamp_ns=frame.stamp_ns,
                        frame_id=frame.frame_id,
                        received_monotonic=frame.received_monotonic,
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)

    def status(self) -> dict[str, object]:
        now = time.monotonic()
        cameras: list[dict[str, object]] = []
        with self._condition:
            for role in ROLE_TOPIC_ROOTS:
                frame = self._frames.get(role)
                info = self._camera_info.get(role)
                age_ms = None if frame is None else max(0.0, (now - frame.received_monotonic) * 1000.0)
                ready = frame is not None and info is not None and age_ms is not None and age_ms <= 2000.0
                cameras.append(
                    {
                        "role": role,
                        "ready": ready,
                        "frame_age_ms": age_ms,
                        "stamp_ns": None if frame is None else frame.stamp_ns,
                        "frame_id": None if frame is None else frame.frame_id,
                        "width": None if info is None else info["width"],
                        "height": None if info is None else info["height"],
                    }
                )
        return {
            "service": "ros-calibration-camera-bridge",
            "status": "ok" if all(camera["ready"] for camera in cameras) else "not_ready",
            "source": "ros2-subscriptions",
            "bind_policy": "localhost-only",
            "robot_command_api": False,
            "publishers_created": False,
            "cameras": cameras,
        }


class BridgeHttpServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], cache: CameraCache):
        super().__init__(address, BridgeRequestHandler)
        self.cache = cache


class BridgeRequestHandler(BaseHTTPRequestHandler):
    server: BridgeHttpServer

    def log_message(self, format_string: str, *args: object) -> None:
        return

    def _json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        encoded = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._json(HTTPStatus.OK, self.server.cache.status())
            return
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) != 4 or parts[:2] != ["api", "cameras"]:
            self._json(HTTPStatus.NOT_FOUND, {"detail": "Not Found"})
            return
        role, resource = parts[2], parts[3]
        if role not in ROLE_TOPIC_ROOTS:
            self._json(HTTPStatus.NOT_FOUND, {"detail": "Unknown camera role"})
            return
        if resource == "camera-info":
            payload = self.server.cache.camera_info(role)
            if payload is None:
                self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"detail": f"{role} CameraInfo is unavailable"})
            else:
                self._json(HTTPStatus.OK, payload)
            return
        if resource != "frame.png":
            self._json(HTTPStatus.NOT_FOUND, {"detail": "Not Found"})
            return
        query = parse_qs(parsed.query)
        try:
            after_stamp_ns = int(query.get("after_stamp_ns", ["-1"])[0])
            timeout_s = min(10.0, max(0.1, float(query.get("timeout_s", ["5"])[0])))
        except ValueError:
            self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"detail": "Invalid frame query"})
            return
        frame = self.server.cache.wait_for_frame(role, after_stamp_ns, timeout_s)
        if frame is None:
            self._json(HTTPStatus.GATEWAY_TIMEOUT, {"detail": f"No new {role} ROS image before timeout"})
            return
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        encoded_ok, encoded = cv2.imencode(".png", bgr)
        if not encoded_ok:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": "OpenCV PNG encoding failed"})
            return
        payload = encoded.tobytes()
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Ros-Stamp-Ns", str(frame.stamp_ns))
        self.send_header("X-Ros-Frame-Id", frame.frame_id)
        self.end_headers()
        self.wfile.write(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8018)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("The calibration camera bridge must remain localhost-only")
    rclpy.init()
    cache = CameraCache()
    server = BridgeHttpServer((args.host, args.port), cache)
    server_thread = threading.Thread(target=server.serve_forever, name="bridge-http", daemon=True)
    server_thread.start()
    try:
        rclpy.spin(cache)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        cache.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
