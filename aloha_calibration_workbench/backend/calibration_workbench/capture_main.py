from __future__ import annotations

from pathlib import Path

import uvicorn

from .capture_api import create_capture_app
from .charuco_analyzer import CharucoAnalyzer
from .intrinsics_capture import IntrinsicsCaptureService
from .preflight import PreflightService
from .realsense_camera import PyRealSenseBackend
from .registry import load_candidate_registry
from .rs_cli_probe import RsEnumerateCliProbe

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROBOT_CONFIG = PROJECT_ROOT / "third_party/aloha_collection/config/robot/aloha_stationary.yaml"
registry = load_candidate_registry(ROBOT_CONFIG)
process_signatures = {camera.serial: [camera.role, camera.config_name] for camera in registry.cameras}
preflight = PreflightService(
    registry=registry,
    probe=RsEnumerateCliProbe(registry.profile, process_signatures_by_serial=process_signatures),
)
CAPTURE_ROOT = PROJECT_ROOT / ".calibration_captures"
capture = IntrinsicsCaptureService(
    preflight=preflight,
    profile=registry.profile,
    backend=PyRealSenseBackend(),
    analyzer=CharucoAnalyzer(),
    artifact_root=CAPTURE_ROOT,
)
app = create_capture_app(preflight, capture)


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8017)
