from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_isaac_extension_manifest_is_present() -> None:
    manifest = ROOT / "visual_tutor/isaac_extensions/my.isaac.visual_tutor/config/extension.toml"
    text = manifest.read_text(encoding="utf-8")
    assert 'title = "My Isaac Visual Tutor"' in text
    assert '"omni.ui"' in text
    assert '"omni.timeline"' in text


def test_isaac_extension_is_simulation_only() -> None:
    extension = ROOT / "visual_tutor/isaac_extensions/my.isaac.visual_tutor/my/isaac/visual_tutor/extension.py"
    text = extension.read_text(encoding="utf-8")
    assert "real_robot_control_disabled" in text
    assert "ros_publish_disabled" in text
    assert "timeline.pause()" in text
