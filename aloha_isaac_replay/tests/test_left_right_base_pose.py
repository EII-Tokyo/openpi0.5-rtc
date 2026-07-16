from __future__ import annotations

from pathlib import Path


def test_puppet_base_pose_source_is_static_transform_from_audit() -> None:
    text = Path("reports/aloha_model_audit/raw/remote_103_focused_audit.txt").read_text()
    assert 'args="0 0.25 0 0 0 0 /world /$(arg robot_name_puppet_left)/base_link"' in text
    assert 'args="0 0.25 0 0 0 0 /world /$(arg robot_name_puppet_right)/base_link"' in text

