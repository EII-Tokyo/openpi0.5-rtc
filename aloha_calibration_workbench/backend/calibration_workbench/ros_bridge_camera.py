from __future__ import annotations

from collections.abc import Mapping

import cv2
import httpx
import numpy as np

from .intrinsics_capture import FramePacket
from .models import FactoryIntrinsics
from .models import OwnershipState
from .models import ProductionProfile
from .preflight import CameraObservation
from .preflight import DeviceProbe


class RosCameraBridgeClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8018",
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=10.0, transport=transport)

    def health(self) -> dict:
        response = self._client.get("/health")
        response.raise_for_status()
        payload = response.json()
        if payload.get("robot_command_api") is not False or payload.get("publishers_created") is not False:
            raise RuntimeError("ROS camera bridge did not attest its read-only contract")
        return payload

    def camera_info(self, role: str) -> FactoryIntrinsics:
        response = self._client.get(f"/api/cameras/{role}/camera-info")
        response.raise_for_status()
        payload = response.json()
        matrix = payload["k"]
        if len(matrix) != 9:
            raise RuntimeError(f"ROS CameraInfo K for {role} must contain 9 values")
        distortion_model = str(payload["distortion_model"])
        if distortion_model == "plumb_bob":
            distortion_model = "brown_conrady"
        return FactoryIntrinsics(
            width=int(payload["width"]),
            height=int(payload["height"]),
            fx=float(matrix[0]),
            fy=float(matrix[4]),
            cx=float(matrix[2]),
            cy=float(matrix[5]),
            distortion_model=distortion_model,
            distortion_coefficients=[float(value) for value in payload["d"]],
        )

    def next_frame(self, role: str, after_stamp_ns: int) -> tuple[FramePacket, int]:
        response = self._client.get(
            f"/api/cameras/{role}/frame.png",
            params={"after_stamp_ns": after_stamp_ns, "timeout_s": 5.0},
            timeout=10.0,
        )
        response.raise_for_status()
        stamp_ns = int(response.headers["X-Ros-Stamp-Ns"])
        encoded = np.frombuffer(response.content, dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"ROS camera bridge returned an invalid PNG for {role}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return (
            FramePacket(
                rgb=rgb,
                frame_number=stamp_ns,
                device_timestamp_ms=stamp_ns / 1_000_000.0,
            ),
            stamp_ns,
        )


class RosBridgeRunningCamera:
    def __init__(self, client: RosCameraBridgeClient, role: str) -> None:
        self._client = client
        self._role = role
        self._last_stamp_ns = -1

    def factory_intrinsics(self) -> FactoryIntrinsics:
        return self._client.camera_info(self._role)

    def next_frame(self) -> FramePacket:
        packet, stamp_ns = self._client.next_frame(self._role, self._last_stamp_ns)
        if stamp_ns <= self._last_stamp_ns:
            raise RuntimeError(f"ROS camera bridge returned a duplicate {self._role} timestamp")
        self._last_stamp_ns = stamp_ns
        return packet

    def stop(self) -> None:
        return


class RosBridgeCameraBackend:
    shared_source = True

    def __init__(self, client: RosCameraBridgeClient, role_by_serial: Mapping[str, str]) -> None:
        self._client = client
        self._role_by_serial = dict(role_by_serial)

    def start(self, serial: str, profile: ProductionProfile) -> RosBridgeRunningCamera:
        if profile.stream != "color" or profile.format.lower() != "rgb8":
            raise ValueError("ROS bridge Stage 1 supports RGB8 color images only")
        try:
            role = self._role_by_serial[serial]
        except KeyError as exc:
            raise ValueError(f"No ROS camera role is registered for serial {serial}") from exc
        health = self._client.health()
        camera = next((item for item in health.get("cameras", []) if item.get("role") == role), None)
        if camera is None or not camera.get("ready"):
            raise RuntimeError(f"ROS camera source is not ready: {role}")
        actual = (int(camera["width"]), int(camera["height"]))
        expected = (profile.width, profile.height)
        if actual != expected:
            raise RuntimeError(f"ROS camera source {role} has resolution {actual}, expected {expected}")
        return RosBridgeRunningCamera(self._client, role)


class RosBridgeDeviceProbe:
    """Merge physical device metadata with live, read-only ROS source readiness."""

    def __init__(
        self,
        physical_probe: DeviceProbe,
        client: RosCameraBridgeClient,
        role_by_serial: Mapping[str, str],
        profile: ProductionProfile,
    ) -> None:
        self._physical_probe = physical_probe
        self._client = client
        self._role_by_serial = dict(role_by_serial)
        self._profile = profile

    def enumerate(self) -> list[CameraObservation]:
        physical = self._physical_probe.enumerate()
        health = self._client.health()
        by_role = {item["role"]: item for item in health.get("cameras", [])}
        observations: list[CameraObservation] = []
        for observation in physical:
            role = self._role_by_serial.get(observation.serial)
            source = by_role.get(role) if role is not None else None
            source_ready = bool(
                source
                and source.get("ready")
                and int(source.get("width", -1)) == self._profile.width
                and int(source.get("height", -1)) == self._profile.height
            )
            observations.append(
                observation.model_copy(
                    update={
                        "production_profile_supported": (
                            observation.production_profile_supported and source_ready
                        ),
                        "ownership": (
                            OwnershipState.ROS_SOURCE if source_ready else OwnershipState.UNKNOWN
                        ),
                        "owner_processes": (
                            [f"ros2:{role}"] if source_ready and role is not None else observation.owner_processes
                        ),
                    }
                )
            )
        return observations
