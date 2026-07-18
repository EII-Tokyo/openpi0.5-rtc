from __future__ import annotations

from aloha_isaac_replay.scripts.build_aloha1_bbox_proxy_runtime_stage import _robot_root_from_side
from aloha_isaac_replay.scripts.build_aloha1_bbox_proxy_runtime_stage import _side_from_path


def test_bbox_proxy_builder_can_select_scene_base_link_rigid_bodies() -> None:
    assert _side_from_path("/scene/left_base_link/left_left_finger_link", "scene_base_link") == "left"
    assert _side_from_path("/scene/right_base_link/right_right_finger_link", "scene_base_link") == "right"
    assert _robot_root_from_side("left", "scene_base_link") == "/scene/left_base_link"
    assert _robot_root_from_side("right", "scene_base_link") == "/scene/right_base_link"


def test_bbox_proxy_builder_keeps_legacy_puppet_default() -> None:
    assert _side_from_path("/puppet_left_vx300s/puppet_left_left_finger_link") == "left"
    assert _side_from_path("/puppet_right_vx300s/puppet_right_right_finger_link") == "right"
    assert _robot_root_from_side("left") == "/puppet_left_vx300s"
    assert _robot_root_from_side("right") == "/puppet_right_vx300s"
