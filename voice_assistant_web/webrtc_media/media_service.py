from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
from fastapi import FastAPI
from fastapi import HTTPException
import numpy as np
from pydantic import BaseModel
from pydantic import Field


GST_REQUIRED_PLUGINS = ("webrtcbin", "videotestsrc", "videoconvert", "fakesink")
ROS_CAMERA_TOPICS: Dict[str, str] = {
    "cam_high": "/cam_high",
    "cam_low": "/cam_low",
    "cam_left_wrist": "/cam_left_wrist",
    "cam_right_wrist": "/cam_right_wrist",
}

app = FastAPI(title="EII Camera WebRTC Media Service")


@dataclass(frozen=True)
class CommandResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: Optional[int]


class VideoTestSrcSmokeRequest(BaseModel):
    num_buffers: int = Field(default=30, ge=1, le=300)


class RosCameraSmokeRequest(BaseModel):
    camera_name: str = Field(default="cam_high")
    timeout_seconds: float = Field(default=3.0, ge=0.2, le=15.0)
    jpeg_quality: int = Field(default=90, ge=10, le=100)


def _run_command(command: List[str], timeout: float) -> CommandResult:
    try:
        completed = subprocess.run(command, capture_output=True, check=False, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        return CommandResult(ok=False, stdout="", stderr=str(exc), returncode=None)
    except subprocess.TimeoutExpired as exc:
        return CommandResult(ok=False, stdout=exc.stdout or "", stderr=exc.stderr or "command timed out", returncode=None)
    return CommandResult(
        ok=completed.returncode == 0,
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )


def _import_gst_webrtc_bindings() -> None:
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstWebRTC", "1.0")
    from gi.repository import Gst  # noqa: F401
    from gi.repository import GstWebRTC  # noqa: F401


def probe_python_gstreamer_bindings() -> Dict[str, Any]:
    try:
        _import_gst_webrtc_bindings()
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
        }
    return {
        "available": True,
        "error": None,
    }


def probe_gstreamer() -> Dict[str, Any]:
    plugins: Dict[str, Dict[str, Any]] = {}
    for plugin in GST_REQUIRED_PLUGINS:
        result = _run_command(["gst-inspect-1.0", plugin], timeout=5)
        plugins[plugin] = {
            "available": result.ok,
            "error": None if result.ok else (result.stderr or result.stdout),
            "returncode": result.returncode,
        }
    python_bindings = probe_python_gstreamer_bindings()
    return {
        "available": all(plugin["available"] for plugin in plugins.values()) and python_bindings["available"],
        "plugins": plugins,
        "python_bindings": python_bindings,
    }


def run_videotestsrc_smoke(num_buffers: int = 30) -> Dict[str, Any]:
    command = [
        "gst-launch-1.0",
        "-q",
        "videotestsrc",
        f"num-buffers={num_buffers}",
        "!",
        "videoconvert",
        "!",
        "fakesink",
    ]
    result = _run_command(command, timeout=10)
    return {
        "ok": result.ok,
        "command": command,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


def get_ros_camera_config() -> Dict[str, Any]:
    return {
        "available": True,
        "cameras": {name: {"topic": topic} for name, topic in ROS_CAMERA_TOPICS.items()},
    }


def validate_ros_camera_name(camera_name: str) -> Dict[str, Any]:
    topic = ROS_CAMERA_TOPICS.get(camera_name)
    if topic is None:
        return {
            "ok": False,
            "error": f"Unknown camera {camera_name!r}",
            "known_cameras": sorted(ROS_CAMERA_TOPICS),
        }
    return {
        "ok": True,
        "error": None,
        "topic": topic,
    }


def image_msg_to_bgr(image_msg: Any) -> np.ndarray:
    channels_by_encoding = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
    }
    channels = channels_by_encoding.get(getattr(image_msg, "encoding", None))
    if channels is None:
        raise ValueError(f"Unsupported image encoding: {getattr(image_msg, 'encoding', None)!r}")

    width = int(getattr(image_msg, "width", 0) or 0)
    height = int(getattr(image_msg, "height", 0) or 0)
    expected_size = width * height * channels
    frame = np.frombuffer(getattr(image_msg, "data", b""), dtype=np.uint8)
    if frame.size < expected_size:
        raise ValueError(f"Camera frame was truncated: got {frame.size}, expected {expected_size}")
    frame = frame[:expected_size].reshape((height, width, channels))

    encoding = image_msg.encoding
    if encoding == "rgb8":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if encoding == "bgr8":
        # Current robot camera diagnostics show RGB semantics are expected downstream.
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if encoding == "mono8":
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def build_jpeg_fakesink_command(jpeg_path: str) -> List[str]:
    return [
        "gst-launch-1.0",
        "-q",
        "filesrc",
        f"location={jpeg_path}",
        "!",
        "jpegdec",
        "!",
        "videoconvert",
        "!",
        "fakesink",
    ]


def _ensure_rospy_node() -> Any:
    import rospy

    if not rospy.core.is_initialized():
        rospy.init_node("eii_webrtc_media_probe", anonymous=True, disable_signals=True)
    return rospy


def capture_ros_camera_frame(camera_name: str, timeout_seconds: float) -> Dict[str, Any]:
    validation = validate_ros_camera_name(camera_name)
    if not validation["ok"]:
        raise ValueError(validation["error"])

    rospy = _ensure_rospy_node()
    from aloha.msg import RGBGrayscaleImage

    topic = validation["topic"]
    event = threading.Event()
    holder: Dict[str, Any] = {}

    def callback(message: Any) -> None:
        if event.is_set():
            return
        holder["message"] = message
        holder["received_at"] = time.time()
        event.set()

    subscriber = rospy.Subscriber(topic, RGBGrayscaleImage, callback, queue_size=1)
    started_at = time.time()
    try:
        if not event.wait(timeout=timeout_seconds):
            raise TimeoutError(f"Timed out waiting for {topic}")
    finally:
        subscriber.unregister()

    message = holder["message"]
    if not getattr(message, "images", None):
        raise ValueError(f"{topic} message contains no images")
    image_msg = message.images[0]
    frame = image_msg_to_bgr(image_msg)
    return {
        "camera_name": camera_name,
        "topic": topic,
        "frame": frame,
        "encoding": getattr(image_msg, "encoding", None),
        "width": int(getattr(image_msg, "width", 0) or 0),
        "height": int(getattr(image_msg, "height", 0) or 0),
        "wait_seconds": max(0.0, holder["received_at"] - started_at),
    }


def run_ros_camera_smoke(camera_name: str, timeout_seconds: float = 3.0, jpeg_quality: int = 90) -> Dict[str, Any]:
    captured = capture_ros_camera_frame(camera_name, timeout_seconds)
    encode_args = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    ok, jpeg = cv2.imencode(".jpg", captured["frame"], encode_args)
    if not ok:
        raise RuntimeError("cv2.imencode returned false")

    fd, jpeg_path = tempfile.mkstemp(prefix=f"eii-{camera_name}-", suffix=".jpg")
    os.close(fd)
    try:
        with open(jpeg_path, "wb") as file:
            file.write(jpeg.tobytes())
        command = build_jpeg_fakesink_command(jpeg_path)
        result = _run_command(command, timeout=10)
    finally:
        try:
            os.unlink(jpeg_path)
        except FileNotFoundError:
            pass

    return {
        "ok": result.ok,
        "camera_name": camera_name,
        "topic": captured["topic"],
        "encoding": captured["encoding"],
        "width": captured["width"],
        "height": captured["height"],
        "wait_seconds": captured["wait_seconds"],
        "jpeg_bytes": len(jpeg.tobytes()),
        "command": command,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/media/gstreamer")
def gstreamer_status() -> Dict[str, Any]:
    return probe_gstreamer()


@app.post("/api/media/smoke/videotestsrc")
def videotestsrc_smoke(request: VideoTestSrcSmokeRequest) -> Dict[str, Any]:
    return run_videotestsrc_smoke(num_buffers=request.num_buffers)


@app.get("/api/media/ros/cameras")
def ros_camera_config() -> Dict[str, Any]:
    return get_ros_camera_config()


@app.post("/api/media/smoke/ros-camera")
def ros_camera_smoke(request: RosCameraSmokeRequest) -> Dict[str, Any]:
    try:
        return run_ros_camera_smoke(
            camera_name=request.camera_name,
            timeout_seconds=request.timeout_seconds,
            jpeg_quality=request.jpeg_quality,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
