from __future__ import annotations

from pathlib import Path


LAUNCHER = Path(__file__).resolve().parents[2] / "tools/launch_aloha1_dual_sleep_gui.sh"


def test_launcher_aligns_sleep_before_starting_gui() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert text.index("run_startup_sleep_alignment") < text.index(".venv_issac/bin/python tools/open_aloha1_runtime_sleep_gui.py")
    assert "--allow-startup-sleep-align" in text
    assert "--rate-hz 50 --move-seconds 5" in text


def test_launcher_verifies_started_ros_shutdown_before_exit() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "docker compose stop --timeout 20 aloha_ros_nodes" in text
    assert "docker compose stop --timeout 20 ros_master" in text
    assert "remote ROS shutdown incomplete" in text


def test_launcher_exports_bundled_isaac_ros2_environment() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "export ROS_DISTRO=jazzy" in text
    assert "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" in text
    assert "isaacsim.ros2.bridge/jazzy/lib" in text
