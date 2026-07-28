from __future__ import annotations

from pathlib import Path

import pytest

from tools.aloha1_mapping.correct_finger_asset import load_correct_finger_profile
from tools.aloha1_mapping.correct_finger_asset import parse_binary_stl_inventory
from tools.aloha1_mapping.correct_finger_asset import verify_correct_finger_sources

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = PROJECT_ROOT / "configs/aloha1_gripper_correct_finger_profiles.yaml"


def test_profile_is_fixed_to_the_verified_gym_aloha_source() -> None:
    profile = load_correct_finger_profile(PROFILE_PATH, PROJECT_ROOT)
    source = profile["source"]

    assert source["repository"] == "https://github.com/huggingface/gym-aloha.git"
    assert source["branch"] == "user/aliberts/2024_05_07_remove_upper_bounds"
    assert source["commit"] == "51837ba5f7d5b96255f01c3d39d53dea473b4829"
    assert source["package_version"] == "0.1.1"
    assert source["license"] == "Apache-2.0"
    assert source["historical_usd"]["sha256"] == (
        "b24afe3678155654892c69517fc58ecd970108d68cec56b02dc6fdcb8bf4493e"
    )


def test_custom_finger_stl_hashes_and_triangle_counts_are_verified() -> None:
    profile = load_correct_finger_profile(PROFILE_PATH, PROJECT_ROOT)
    evidence = verify_correct_finger_sources(profile, PROJECT_ROOT)

    assert evidence["status"] == "PASS"
    assert evidence["meshes"]["left"]["sha256"] == (
        "df73ae5b9058e5d50a6409ac2ab687dade75053a86591bb5e23ab051dbf2d659"
    )
    assert evidence["meshes"]["right"]["sha256"] == (
        "56fb3cc1236d4193106038adf8e457c7252ae9e86c7cee6dabf0578c53666358"
    )
    assert evidence["meshes"]["left"]["triangle_count"] == 1666
    assert evidence["meshes"]["right"]["triangle_count"] == 1666
    assert evidence["rejected_generic_mesh"]["verified_rejected"] is True


def test_binary_stl_inventory_rejects_truncated_triangle_payload(
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "invalid.stl"
    invalid.write_bytes(b"\0" * 80 + (2).to_bytes(4, "little") + b"\0" * 50)

    with pytest.raises(ValueError, match="size"):
        parse_binary_stl_inventory(invalid)


def test_mjcf_installation_transforms_and_joint_limits_are_source_readback() -> None:
    profile = load_correct_finger_profile(PROFILE_PATH, PROJECT_ROOT)
    evidence = verify_correct_finger_sources(profile, PROJECT_ROOT)
    installs = evidence["mjcf_installation_readback"]

    for robot in ("vx300s_left", "vx300s_right"):
        assert installs[robot]["left"]["position_m"] == [0.005, -0.052, 0.0]
        assert installs[robot]["left"]["euler_rad"] == [3.14, 1.57, 0.0]
        assert installs[robot]["left"]["joint_axis"] == [0.0, 1.0, 0.0]
        assert installs[robot]["left"]["joint_range_m"] == [0.021, 0.057]
        assert installs[robot]["right"]["position_m"] == [0.005, 0.052, 0.0]
        assert installs[robot]["right"]["euler_rad"] == [3.14, 1.57, 0.0]
        assert installs[robot]["right"]["joint_axis"] == [0.0, 1.0, 0.0]
        assert installs[robot]["right"]["joint_range_m"] == [-0.057, -0.021]


def test_task5_non_geometry_variables_remain_frozen() -> None:
    profile = load_correct_finger_profile(PROFILE_PATH, PROJECT_ROOT)
    frozen = profile["frozen"]

    assert frozen["friction"] == 0.7
    assert frozen["restitution"] == 0.0
    assert frozen["bottle_mass_kg"] == 0.020
    assert frozen["bottle_diameter_m"] == 0.065
    assert frozen["physics_frequency_hz"] == 60
    assert frozen["solve_articulation_contact_last"] is True
    assert frozen["hold_interval_s"] == 2.0
    assert frozen["drop_gate_m"] == 0.010
    assert frozen["self_collision"] is False
    assert frozen["bottle_collision"] is True
    assert frozen["surface_gripper_allowed"] is False
    assert frozen["post_release_fixed_constraint_allowed"] is False


def test_screenshot_contract_has_required_phases_and_fixed_resolution() -> None:
    profile = load_correct_finger_profile(PROFILE_PATH, PROJECT_ROOT)
    screenshots = profile["screenshots"]

    assert screenshots["resolution"] == [1280, 900]
    assert set(screenshots["required_captures"]) == {
        "asset_preflight",
        "collider_geometry",
        "bilateral_contact",
        "release_hold",
    }
    assert screenshots["runtime_renderer"] == "ISAAC_SIM_5_1_CAMERA_RGB"
    assert screenshots["missing_capture_gate"] == "FAIL"
