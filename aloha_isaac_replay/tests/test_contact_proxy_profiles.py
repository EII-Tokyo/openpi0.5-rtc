from __future__ import annotations

import pytest

from aloha_isaac_replay.validation.contact_proxy_profiles import contact_proxy_namespace_roots
from aloha_isaac_replay.validation.contact_proxy_profiles import contact_proxy_profile_names
from aloha_isaac_replay.validation.contact_proxy_profiles import finger_dof_names_for_side
from aloha_isaac_replay.validation.contact_proxy_profiles import proxy_path_for_rigid_body
from aloha_isaac_replay.validation.contact_proxy_profiles import resolve_contact_proxy_paths
from aloha_isaac_replay.validation.contact_proxy_profiles import side_from_rigid_body_path
from aloha_isaac_replay.validation.contact_proxy_profiles import stage_units_in_meters_for_profile
from aloha_isaac_replay.validation.contact_proxy_profiles import stage_up_axis_for_profile


def test_scene_base_link_profile_uses_scene_articulation_roots_and_fingertip_proxies() -> None:
    paths = resolve_contact_proxy_paths("scene_base_link")

    assert paths["left"]["articulation"] == "/scene/left_base_link/left_base_link"
    assert paths["right"]["articulation"] == "/scene/right_base_link/right_base_link"
    assert paths["left"]["left_finger"] == "/scene/left_base_link/left_left_finger_link/bbox_collision_proxy"
    assert paths["left"]["right_finger"] == "/scene/left_base_link/left_right_finger_link/bbox_collision_proxy"
    assert paths["right"]["left_finger"] == "/scene/right_base_link/right_left_finger_link/bbox_collision_proxy"
    assert paths["right"]["right_finger"] == "/scene/right_base_link/right_right_finger_link/bbox_collision_proxy"
    assert contact_proxy_namespace_roots(paths) == ["scene"]
    assert stage_units_in_meters_for_profile("scene_base_link") == 1.0
    assert stage_up_axis_for_profile("scene_base_link") == "Z"
    assert finger_dof_names_for_side("scene_base_link", "left") == {
        "left_finger": "left_left_finger",
        "right_finger": "left_right_finger",
    }
    assert finger_dof_names_for_side("scene_base_link", "right") == {
        "left_finger": "right_left_finger",
        "right_finger": "right_right_finger",
    }


def test_scene_base_link_profile_maps_rigid_body_paths_to_selected_side_and_proxy_path() -> None:
    assert side_from_rigid_body_path("scene_base_link", "/scene/left_base_link/left_left_finger_link") == "left"
    assert side_from_rigid_body_path("scene_base_link", "/scene/right_base_link/right_right_finger_link") == "right"
    assert side_from_rigid_body_path("scene_base_link", "/puppet_left_vx300s/puppet_left_left_finger_link") == "unknown"
    assert (
        proxy_path_for_rigid_body("scene_base_link", "/scene/left_base_link/left_left_finger_link")
        == "/scene/left_base_link/left_left_finger_link/bbox_collision_proxy"
    )


def test_legacy_puppet_profile_remains_available_for_old_runtime_stages() -> None:
    paths = resolve_contact_proxy_paths("legacy_puppet")

    assert paths["left"]["articulation"] == "/puppet_left_vx300s/root_joint"
    assert paths["right"]["articulation"] == "/puppet_right_vx300s/root_joint"
    assert contact_proxy_namespace_roots(paths) == ["puppet_left_vx300s", "puppet_right_vx300s"]
    assert "legacy_puppet" in contact_proxy_profile_names()
    assert stage_units_in_meters_for_profile("legacy_puppet") == 0.01
    assert stage_up_axis_for_profile("legacy_puppet") == "Y"
    assert finger_dof_names_for_side("legacy_puppet", "left") == {
        "left_finger": "left_finger",
        "right_finger": "right_finger",
    }


def test_unknown_contact_proxy_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown contact proxy profile"):
        resolve_contact_proxy_paths("does_not_exist")
