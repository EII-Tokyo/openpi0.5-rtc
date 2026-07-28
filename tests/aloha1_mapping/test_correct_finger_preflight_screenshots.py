from __future__ import annotations

import json
from pathlib import Path

from tools.aloha1_mapping.screenshot_manifest import validate_screenshot

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_preflight_capture_uses_local_isaac_5_1_camera_runtime() -> None:
    entrypoint = (
        PROJECT_ROOT / "tools/capture_aloha1_correct_finger_preflight.py"
    ).read_text(encoding="utf-8")
    helper = (
        PROJECT_ROOT / "tools/aloha1_mapping/isaac_screenshot.py"
    ).read_text(encoding="utf-8")
    source = entrypoint + helper

    assert "from isaacsim.sensors.camera import Camera" in source
    assert ".initialize()" in source
    assert "world.step(render=True)" in source
    assert "get_rgba()" in source
    assert "BLENDER" not in source


def test_preflight_screenshot_report_contains_eight_verified_absolute_paths() -> None:
    report = json.loads(
        (
            PROJECT_ROOT
            / "reports/aloha1_mapping/gripper_correct_finger_preflight_screenshots.json"
        ).read_text(encoding="utf-8")
    )

    assert report["status"] == "PASS"
    assert report["manifest"]["status"] == "PASS"
    assert report["manifest"]["required_capture_count"] == 8
    assert len(report["manifest"]["captures"]) == 8
    root = Path(report["manifest"]["artifact_root"])
    assert root.is_absolute()
    for capture in report["manifest"]["captures"]:
        path = Path(capture["absolute_path"])
        assert path.is_absolute()
        assert path.is_file()
        readback = validate_screenshot(
            path,
            artifact_root=root,
            phase=capture["phase"],
            capture_name=capture["capture_name"],
            gate_status=capture["capture_gate_status"],
            camera=capture["camera"],
            simulation=capture["simulation"],
        )
        assert readback["file_sha256"] == capture["file_sha256"]
        assert (
            readback["decoded_pixel_sha256"]
            == capture["decoded_pixel_sha256"]
        )


def test_preflight_numeric_gate_proves_open_aperture_exceeds_closed() -> None:
    report = json.loads(
        (
            PROJECT_ROOT
            / "reports/aloha1_mapping/gripper_correct_finger_preflight_screenshots.json"
        ).read_text(encoding="utf-8")
    )

    for robot, states in report["robots"].items():
        assert states["open"]["surface_gap_m"] > states["closed"]["surface_gap_m"], robot
        assert states["open"]["left_finger_readback_m"] > states["closed"][
            "left_finger_readback_m"
        ]
        assert states["open"]["right_finger_readback_m"] < states["closed"][
            "right_finger_readback_m"
        ]
        assert states["gates"]["aperture_monotonic"] is True
        assert states["gates"]["legal_joint_readback"] is True
