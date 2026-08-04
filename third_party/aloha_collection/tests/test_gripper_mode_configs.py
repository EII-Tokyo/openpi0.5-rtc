from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
LEFT_CONFIG = ROOT / "config/follower_modes_left.yaml"
RIGHT_CONFIG = ROOT / "config/follower_modes_right.yaml"


def _gripper_config(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))["singles"]["gripper"]


def test_left_bringup_uses_approved_current_based_position_profile():
    assert _gripper_config(LEFT_CONFIG) == {
        "operating_mode": "current_based_position",
        "profile_type": "velocity",
        "profile_velocity": 50,
        "profile_acceleration": 10,
        "current_limit": 300,
        "torque_enable": True,
    }


def test_right_bringup_profile_remains_unchanged():
    assert _gripper_config(RIGHT_CONFIG) == {
        "operating_mode": "current_based_position",
        "profile_type": "velocity",
        "profile_velocity": 0,
        "profile_acceleration": 0,
        "current_limit": 500,
        "torque_enable": True,
    }
