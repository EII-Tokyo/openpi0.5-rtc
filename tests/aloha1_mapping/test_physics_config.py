from pathlib import Path

from tools.aloha1_mapping.physics_config import build_physics_plan

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_physics_plan_uses_source_home_and_disables_mimic_drive() -> None:
    plan = build_physics_plan(PROJECT_ROOT)

    left = plan["robots"][0]
    assert left["name"] == "follower_left"
    assert left["home_si"] == [
        0.0,
        -0.96,
        1.16,
        0.0,
        -0.3,
        0.0,
        0.0,
        0.02239,
        -0.02239,
    ]
    assert left["dofs"][8]["mimic"] is True
    assert left["dofs"][8]["author_drive"] is False


def test_force_profile_is_present_but_not_claimed_as_calibrated() -> None:
    plan = build_physics_plan(PROJECT_ROOT)

    assert plan["profiles"]["debug_acceleration_drive"]["drive_type"] == "acceleration"
    assert plan["profiles"]["sim2real_force_drive"]["drive_type"] == "force"
    assert plan["profiles"]["sim2real_force_drive"]["status"] == "CALIBRATION_PENDING"
    assert plan["default_profile"] == "debug_acceleration_drive"
    assert plan["fingertip_material"]["status"] == "TEMPORARY_PLACEHOLDER"
