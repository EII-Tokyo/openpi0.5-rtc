from __future__ import annotations

from aloha_isaac_replay.scripts.build_aloha1_bbox_proxy_runtime_stage import _robot_root_from_side
from aloha_isaac_replay.scripts.build_aloha1_bbox_proxy_runtime_stage import (
    _known_scene_base_link_finger_collision_instance_root,
    _known_scene_base_link_finger_collision_paths,
)
from aloha_isaac_replay.scripts.build_aloha1_bbox_proxy_runtime_stage import _should_disable_selected_source_collision
from aloha_isaac_replay.scripts.build_aloha1_bbox_proxy_runtime_stage import _side_from_path
from aloha_isaac_replay.scripts.build_aloha1_bbox_proxy_runtime_stage import _normalized_paths
from aloha_isaac_replay.scripts.build_aloha1_bbox_proxy_runtime_stage import _source_bbox_path_for_rigid_body


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


def test_scene_base_link_finger_proxies_use_local_site_bbox_sources() -> None:
    assert (
        _source_bbox_path_for_rigid_body("scene_base_link", "/scene/left_base_link/left_left_finger_link")
        == "/scene/left_base_link/left_left_finger_link/sites/left_left_finger"
    )
    assert (
        _source_bbox_path_for_rigid_body("scene_base_link", "/scene/left_base_link/left_right_finger_link")
        == "/scene/left_base_link/left_right_finger_link/sites/left_right_finger"
    )
    assert (
        _source_bbox_path_for_rigid_body("scene_base_link", "/scene/right_base_link/right_left_finger_link")
        == "/scene/right_base_link/right_left_finger_link/sites/right_left_finger"
    )
    assert (
        _source_bbox_path_for_rigid_body("scene_base_link", "/scene/right_base_link/right_right_finger_link")
        == "/scene/right_base_link/right_right_finger_link/sites/right_right_finger"
    )


def test_scene_base_link_non_finger_proxy_source_defaults_to_rigid_body_path() -> None:
    path = "/scene/left_base_link/left_wrist_link"

    assert _source_bbox_path_for_rigid_body("scene_base_link", path) == path


def test_selected_source_collision_disable_rule_keeps_proxy_but_disables_old_descendant_collision() -> None:
    selected_root = "/scene/left_base_link/left_left_finger_link"
    proxy_path = f"{selected_root}/bbox_collision_proxy"

    assert (
        _should_disable_selected_source_collision(
            selected_root=selected_root,
            proxy_path=proxy_path,
            collision_path=f"{selected_root}/collisions/left_left_g1/left_left_g1",
        )
        is True
    )
    assert (
        _should_disable_selected_source_collision(
            selected_root=selected_root,
            proxy_path=proxy_path,
            collision_path=proxy_path,
        )
        is False
    )
    assert (
        _should_disable_selected_source_collision(
            selected_root=selected_root,
            proxy_path=proxy_path,
            collision_path="/scene/left_base_link/left_wrist_link/collisions/wrist",
        )
        is False
    )


def test_scene_base_link_known_left_finger_collisions_include_group_and_leaf_paths() -> None:
    selected_root = "/scene/left_base_link/left_left_finger_link"

    paths = _known_scene_base_link_finger_collision_paths(selected_root)

    assert f"{selected_root}/collisions/left_left_g0" in paths
    assert f"{selected_root}/collisions/left_left_g0/left_left_g0" in paths
    assert f"{selected_root}/collisions/left_left_g1/left_left_g1" in paths
    assert f"{selected_root}/collisions/left_left_g2/left_left_g2" in paths
    assert (
        f"{selected_root}/collisions/vx300s_8_custom_finger_left/vx300s_8_custom_finger_left"
        in paths
    )
    assert len(paths) == 8


def test_scene_base_link_known_right_finger_collisions_follow_arm_and_finger_side() -> None:
    selected_root = "/scene/right_base_link/right_right_finger_link"

    paths = _known_scene_base_link_finger_collision_paths(selected_root)

    assert f"{selected_root}/collisions/right_right_g0/right_right_g0" in paths
    assert f"{selected_root}/collisions/right_right_g1/right_right_g1" in paths
    assert f"{selected_root}/collisions/right_right_g2/right_right_g2" in paths
    assert (
        f"{selected_root}/collisions/vx300s_8_custom_finger_right/vx300s_8_custom_finger_right"
        in paths
    )
    assert len(paths) == 8


def test_scene_base_link_known_finger_collision_instance_root_only_for_known_fingers() -> None:
    selected_root = "/scene/left_base_link/left_right_finger_link"

    assert (
        _known_scene_base_link_finger_collision_instance_root(selected_root)
        == "/scene/left_base_link/left_right_finger_link/collisions"
    )
    assert _known_scene_base_link_finger_collision_instance_root("/scene/left_base_link/left_wrist_link") is None


def test_explicit_collision_paths_are_normalized_and_deduplicated() -> None:
    paths = _normalized_paths(
        [
            "/scene/worldBody/table/collisions/table/table/table/",
            "/scene/worldBody/table/collisions/table/table/table",
            "",
            "/scene/worldBody/__22/collisions/__22/__22/extrusion_1220/",
        ]
    )

    assert paths == [
        "/scene/worldBody/__22/collisions/__22/__22/extrusion_1220",
        "/scene/worldBody/table/collisions/table/table/table",
    ]
