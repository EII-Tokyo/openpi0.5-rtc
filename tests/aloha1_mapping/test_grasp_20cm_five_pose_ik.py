from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/aloha1_grasp_20cm_five_pose_ik.yaml"


def test_five_pose_config_freezes_joint_sampling_and_diversity() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert config["schema_version"] == 2
    assert config["sampling"]["seed"] == 2026073102
    assert config["sampling"]["formal_sample_count"] == 5
    assert config["sampling"]["candidate_count"] == 256
    assert config["sampling"]["bottle_line_yaw_domain_deg"] == [0.0, 180.0]
    assert config["gates"]["minimum_bottle_line_yaw_separation_deg"] == 25.0
    assert config["gates"]["minimum_initial_ee_separation_m"] == 0.050
    assert (
        config["formal_structure"]["sample_01"]["bottle_center_world_x_m"]
        == 0.0
    )
    assert (
        config["formal_structure"]["sample_01"]["bottle_center_y_sign"]
        == "positive"
    )
    assert (
        config["formal_structure"]["sample_04"]["bottle_center_world_x_m"]
        == 0.0
    )
    assert (
        config["formal_structure"]["sample_04"]["bottle_center_y_sign"]
        == "negative"
    )
    assert config["runtime"]["allow_runtime_resampling"] is False
    assert config["runtime"]["required_primary_videos"] == 5
    assert config["boundaries"]["task8"] == "NOT_RUN"
