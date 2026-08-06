from __future__ import annotations

import cv2
import httpx
import numpy as np
import pytest

from calibration_workbench.models import CandidateCamera
from calibration_workbench.models import CandidateRegistry
from calibration_workbench.models import OwnershipState
from calibration_workbench.models import ProductionProfile
from calibration_workbench.preflight import CameraObservation
from calibration_workbench.preflight import PreflightService
from calibration_workbench.ros_bridge_camera import RosBridgeCameraBackend
from calibration_workbench.ros_bridge_camera import RosBridgeDeviceProbe
from calibration_workbench.ros_bridge_camera import RosCameraBridgeClient


def _health(*, publishers_created: bool = False) -> dict:
    return {
        "service": "ros-calibration-camera-bridge",
        "status": "ok",
        "source": "ros2-subscriptions",
        "robot_command_api": False,
        "publishers_created": publishers_created,
        "cameras": [
            {
                "role": "cam_high",
                "ready": True,
                "frame_age_ms": 5.0,
                "stamp_ns": 100,
                "frame_id": "camera_color_optical_frame",
                "width": 640,
                "height": 480,
            }
        ],
    }


def _transport(*, publishers_created: bool = False) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json=_health(publishers_created=publishers_created))
        if request.url.path == "/api/cameras/cam_high/camera-info":
            return httpx.Response(
                200,
                json={
                    "role": "cam_high",
                    "width": 640,
                    "height": 480,
                    "distortion_model": "plumb_bob",
                    "d": [0.1, -0.1, 0.0, 0.0, 0.0],
                    "k": [601.0, 0.0, 319.5, 0.0, 602.0, 239.5, 0.0, 0.0, 1.0],
                },
            )
        if request.url.path == "/api/cameras/cam_high/frame.png":
            after = int(request.url.params["after_stamp_ns"])
            stamp = max(100, after + 1)
            bgr = np.zeros((480, 640, 3), dtype=np.uint8)
            bgr[:, :, 2] = 255
            ok, encoded = cv2.imencode(".png", bgr)
            assert ok
            return httpx.Response(
                200,
                content=encoded.tobytes(),
                headers={"X-Ros-Stamp-Ns": str(stamp), "X-Ros-Frame-Id": "camera_color_optical_frame"},
            )
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_ros_bridge_client_reads_camera_info_and_unique_lossless_frames() -> None:
    client = RosCameraBridgeClient(transport=_transport())
    intrinsics = client.camera_info("cam_high")
    assert (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy) == (601.0, 602.0, 319.5, 239.5)
    assert intrinsics.distortion_model == "brown_conrady"

    running = RosBridgeCameraBackend(client, {"130322270656": "cam_high"}).start(
        "130322270656",
        ProductionProfile(width=640, height=480, fps=60, format="rgb8"),
    )
    first = running.next_frame()
    second = running.next_frame()

    assert first.rgb.shape == (480, 640, 3)
    assert first.rgb[0, 0].tolist() == [255, 0, 0]
    assert second.frame_number > first.frame_number


def test_ros_bridge_client_rejects_a_bridge_that_can_publish() -> None:
    client = RosCameraBridgeClient(transport=_transport(publishers_created=True))

    with pytest.raises(RuntimeError, match="read-only"):
        client.health()


class FakePhysicalProbe:
    def enumerate(self) -> list[CameraObservation]:
        return [
            CameraObservation(
                serial="130322270656",
                product_name="Intel RealSense D405",
                firmware="5.17.0.10",
                recommended_firmware="5.17.0.10",
                usb_type="3.2",
                physical_port="/sys/example/video0",
                production_profile_supported=True,
                ownership=OwnershipState.BUSY,
                owner_processes=["pid=1:realsense2_camera_node"],
                video_nodes=["/dev/video0"],
            )
        ]


def test_ros_preflight_treats_the_verified_ros_publisher_as_the_source_not_a_conflict() -> None:
    profile = ProductionProfile(width=640, height=480, fps=60, format="rgb8")
    registry = CandidateRegistry(
        source_path="/project/aloha_stationary.yaml",
        source_sha256="a" * 64,
        cameras=[CandidateCamera(role="cam_high", config_name="camera_high", serial="130322270656")],
        profile=profile,
    )
    client = RosCameraBridgeClient(transport=_transport())
    probe = RosBridgeDeviceProbe(FakePhysicalProbe(), client, {"130322270656": "cam_high"}, profile)

    report = PreflightService(
        registry=registry,
        probe=probe,
        exclusive_capture_required=False,
    ).run()

    assert report.status == "READY"
    assert report.exclusive_capture_required is False
    assert report.cameras[0].ownership is OwnershipState.ROS_SOURCE
    assert report.cameras[0].owner_processes == ["ros2:cam_high"]
