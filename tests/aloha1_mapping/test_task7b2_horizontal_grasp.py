from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/aloha1_task7b2_horizontal_grasp.yaml"


def test_horizontal_config_freezes_geometry_and_task_boundaries() -> None:
    assert CONFIG.is_file(), f"missing horizontal Task 7B.2 config: {CONFIG}"
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert config["schema_version"] == 2
    assert config["task_geometry"] == "HORIZONTAL_DYNAMIC_TABLE_SUPPORTED"
    assert config["runtime"] == {
        "isaac_sim": "5.1.0.0",
        "kit": "107.3.3",
        "physx": "107.3.26",
        "motion_generation_extension": "8.0.26",
    }
    assert config["bottle"]["axis"]["a_local_m"] == [0.0, 0.0, 0.0]
    assert config["bottle"]["axis"]["b_local_m"] == [0.0, 0.0, 0.206]
    assert config["bottle"]["body_interval_m"] == [0.018, 0.120]
    assert config["bottle"]["grasp_coordinate_m"] == 0.069
    assert config["episode18"]["frames_inclusive"] == [208, 244]
    assert config["episode18"]["use_action_as_command"] is True
    assert config["episode18"]["use_qpos_as_readback"] is True
    assert config["motion"]["approach_direction_world"] == [0.0, 0.0, -1.0]
    assert config["motion"]["lift_direction_world"] == [0.0, 0.0, 1.0]
    assert config["physics"]["mass_kg"] == 0.020
    assert config["physics"]["friction"] == 0.7
    assert config["physics"]["frequency_hz"] == 60
    assert config["physics"]["hold_interval_s"] == 2.0
    assert config["physics"]["drop_gate_m"] == 0.010
    assert config["boundaries"]["task8"] == "NOT_RUN"
    assert (
        config["legacy"]["upright_shoulder_sweep"]["acceptance_eligible"]
        is False
    )


def test_horizontal_config_freezes_exact_sources() -> None:
    assert CONFIG.is_file(), f"missing horizontal Task 7B.2 config: {CONFIG}"
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    sources = config["frozen_inputs"]

    assert sources["task7a_stage"]["sha256"] == (
        "d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf"
    )
    assert sources["project_bottle_cad"]["sha256"] == (
        "3594f60200e54181bc8480a229484293a0d386c146d3f235b32e31a0c16bbf8a"
    )
    assert sources["project_bottle_usd"]["sha256"] == (
        "16427135f152ec951de2321fd689366d745a2dd389cbe260976631783952533e"
    )
    assert sources["follower_left_urdf"]["sha256"] == (
        "d9e4b32723ee71dfce26fb4e78546cfcfef147b2d7dbf5e53e3620e3d8aa96bd"
    )
    assert sources["joint_map"]["sha256"] == (
        "2c40a637d95d0ae960d11ae4f120f0ca06a77146917ef50051baca1d3a8c496d"
    )
    assert sources["episode18"]["sha256"] == (
        "f073a21c6a790e738e36085d791482924a82832ca6d80cece04a26353b9fc745"
    )
