from __future__ import annotations

from pathlib import Path


def test_reference_fk_source_is_archived_robot_description_not_isaac_usd() -> None:
    text = Path("reports/aloha_model_audit/raw/robot_descriptions/puppet_left_robot_description.urdf").read_text()
    assert '<robot name="vx300s">' in text
    assert "ee_gripper_link" in text

