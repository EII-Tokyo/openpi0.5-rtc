from pathlib import Path

import yaml

from tools.probe_aloha1_cam_high_ros1 import build_dry_run_report
from tools.probe_aloha1_cam_high_ros1 import camera_contract
from tools.probe_aloha1_cam_high_ros1 import classify_capture

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "tools/probe_aloha1_cam_high_ros1.py"
COMPOSE = ROOT / "configs/aloha1_home_sleep_live_readback.compose.yaml"


def test_camera_contract_is_only_the_deployed_cam_high() -> None:
    contract = camera_contract()

    assert contract == {
        "name": "cam_high",
        "serial": "130322270656",
        "width": 640,
        "height": 480,
        "fps": 60,
        "realsense_format": "bgr8",
        "ros_encoding": "bgr8",
        "topic": "/cam_high",
        "message_type": "aloha.msg/RGBGrayscaleImage",
        "serial_evidence": "REMOTE_DEPLOYED_CONFIG_READBACK",
    }


def test_probe_source_has_no_reset_or_other_camera_serials() -> None:
    source = PROBE.read_text(encoding="utf-8")

    assert ".hardware_reset(" not in source
    assert "130322272542" not in source
    assert "218622270440" not in source
    assert "218622278936" not in source
    assert "/puppet_left/commands" not in source


def test_probe_defaults_to_no_hardware_access() -> None:
    report = build_dry_run_report()

    assert report["status"] == "NOT_RUN_EXPLICIT_CAMERA_FLAG_REQUIRED"
    assert report["camera_opened"] is False
    assert report["ros_publisher_constructed"] is False
    assert report["robot_command_publishers"] == 0
    assert report["hardware_resets"] == 0


def test_capture_requires_frames_and_exact_encoding() -> None:
    passed = classify_capture(
        frame_count=120,
        publisher_count=120,
        width=640,
        height=480,
        ros_encoding="bgr8",
        serial="130322270656",
    )
    failed = classify_capture(
        frame_count=0,
        publisher_count=0,
        width=640,
        height=480,
        ros_encoding="bgr8",
        serial="130322270656",
    )

    assert passed == "PASS_CAM_HIGH_SINGLE_CAMERA_RUNTIME"
    assert failed == "FAIL_NO_CAMERA_FRAMES"


def test_compose_override_starts_only_left_driver_and_camera_probe() -> None:
    config = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    services = config["services"]

    assert set(services) == {"aloha_ros_nodes", "aloha_cam_high_probe"}
    driver_command = services["aloha_ros_nodes"]["command"]
    camera_command = services["aloha_cam_high_probe"]["command"]
    rendered = COMPOSE.read_text(encoding="utf-8")
    assert driver_command == (
        "roslaunch --wait aloha "
        "aloha1_home_sleep_puppet_left_only_candidate.launch"
    )
    assert "--execute-camera-hardware" in camera_command
    assert "master_left" not in rendered
    assert "master_right" not in rendered
    assert "puppet_right" not in rendered
    assert "rosbridge" not in rendered
