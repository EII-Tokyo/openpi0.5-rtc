from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from .capture_api import create_capture_app
from .charuco_analyzer import CharucoAnalyzer
from .intrinsics_capture import IntrinsicsCaptureService
from .preflight import PreflightService
from .realsense_camera import PyRealSenseBackend
from .registry import load_candidate_registry
from .ros_bridge_camera import RosBridgeCameraBackend
from .ros_bridge_camera import RosBridgeDeviceProbe
from .ros_bridge_camera import RosCameraBridgeClient
from .rs_cli_probe import RsEnumerateCliProbe

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROBOT_CONFIG = PROJECT_ROOT / "third_party/aloha_collection/config/robot/aloha_stationary.yaml"
registry = load_candidate_registry(ROBOT_CONFIG)
process_signatures = {camera.serial: [camera.role, camera.config_name] for camera in registry.cameras}
physical_probe = RsEnumerateCliProbe(registry.profile, process_signatures_by_serial=process_signatures)
CAMERA_SOURCE = os.environ.get("ALOHA_CALIBRATION_CAMERA_SOURCE", "realsense").strip().lower()
if CAMERA_SOURCE == "ros_bridge":
    bridge_client = RosCameraBridgeClient(
        os.environ.get("ALOHA_CALIBRATION_ROS_BRIDGE_URL", "http://127.0.0.1:8018")
    )
    role_by_serial = {camera.serial: camera.role for camera in registry.cameras}
    preflight = PreflightService(
        registry=registry,
        probe=RosBridgeDeviceProbe(physical_probe, bridge_client, role_by_serial, registry.profile),
        exclusive_capture_required=False,
    )
    camera_backend = RosBridgeCameraBackend(bridge_client, role_by_serial)
elif CAMERA_SOURCE == "realsense":
    preflight = PreflightService(registry=registry, probe=physical_probe)
    camera_backend = PyRealSenseBackend()
else:
    raise RuntimeError(f"Unsupported ALOHA_CALIBRATION_CAMERA_SOURCE: {CAMERA_SOURCE}")
CAPTURE_ROOT = PROJECT_ROOT / ".calibration_captures"
capture = IntrinsicsCaptureService(
    preflight=preflight,
    profile=registry.profile,
    backend=camera_backend,
    analyzer=CharucoAnalyzer(),
    artifact_root=CAPTURE_ROOT,
)
app = create_capture_app(preflight, capture)


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8017)
