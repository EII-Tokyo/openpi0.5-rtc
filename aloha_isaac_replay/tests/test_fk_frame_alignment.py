from __future__ import annotations

from pathlib import Path


def test_frame_alignment_sources_include_world_to_puppet_base_transform() -> None:
    text = Path("reports/aloha_model_audit/raw/remote_103_focused_audit.txt").read_text()
    assert "/world /$(arg robot_name_puppet_left)/base_link" in text
    assert "/world /$(arg robot_name_puppet_right)/base_link" in text

