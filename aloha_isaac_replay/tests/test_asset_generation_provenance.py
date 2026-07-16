from __future__ import annotations

from pathlib import Path


def test_real_robot_description_sources_are_archived_locally() -> None:
    for path in (
        Path("reports/aloha_model_audit/raw/robot_descriptions/puppet_left_robot_description.urdf"),
        Path("reports/aloha_model_audit/raw/robot_descriptions/puppet_right_robot_description.urdf"),
    ):
        assert path.exists()
        assert path.read_text().count("vx300s_meshes") >= 10


def test_aloha2_assets_are_not_allowed_as_main_original_aloha_asset() -> None:
    path = Path("local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose/config.yaml")
    assert path.exists()
    text = path.read_text()
    assert "aloha2_menagerie_scene" in text

