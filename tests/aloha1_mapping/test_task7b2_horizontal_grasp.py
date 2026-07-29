from __future__ import annotations

import ast
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

from tools.probe_aloha1_task7b2_horizontal_kinematics import detect_lift_onset

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/aloha1_task7b2_horizontal_grasp.yaml"
DESCRIPTOR = ROOT / "configs/aloha1_lula_follower_left.yaml"
URDF = ROOT / "generated/urdf/follower_left.urdf"
JOINT_MAP = ROOT / "configs/aloha1_joint_map.yaml"
KINEMATICS_PROBE = (
    ROOT / "tools/probe_aloha1_task7b2_horizontal_kinematics.py"
)
KINEMATICS_REPORT = (
    ROOT / "reports/aloha1_mapping/aloha1_task7b2_horizontal_kinematics.json"
)
EXPECTED_CSPACE = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]


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
    assert config["robot"]["articulation_path"] == (
        "/World/follower_left/vx300s_left/root_joint"
    )
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


def test_lula_descriptor_matches_explicit_urdf_and_joint_map_order() -> None:
    assert DESCRIPTOR.is_file(), f"missing Lula descriptor: {DESCRIPTOR}"
    descriptor = yaml.safe_load(DESCRIPTOR.read_text(encoding="utf-8"))

    assert descriptor["api_version"] == 1.0
    assert descriptor["cspace"] == EXPECTED_CSPACE
    assert descriptor["root_link"] == "follower_left_base_link"
    assert descriptor["default_q"] == [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
    assert descriptor["cspace_to_urdf_rules"] == [
        {"name": "gripper", "rule": "fixed", "value": 0.0},
        {"name": "left_finger", "rule": "fixed", "value": 0.057},
        {"name": "right_finger", "rule": "fixed", "value": -0.057},
    ]

    urdf_root = ET.parse(URDF).getroot()
    urdf_nonfixed = [
        joint.attrib["name"]
        for joint in urdf_root.findall("joint")
        if joint.attrib["type"] != "fixed"
    ]
    assert urdf_nonfixed[:6] == EXPECTED_CSPACE
    assert {"gripper", "left_finger", "right_finger"} <= set(urdf_nonfixed)
    assert urdf_root.find("link[@name='follower_left_base_link']") is not None
    assert (
        urdf_root.find("link[@name='follower_left_gripper_link']")
        is not None
    )

    joint_map = yaml.safe_load(JOINT_MAP.read_text(encoding="utf-8"))
    left = joint_map["robots"]["follower_left"]
    assert left["isaac_dof_order"][:6] == EXPECTED_CSPACE
    assert [record["name"] for record in left["dofs"][:6]] == EXPECTED_CSPACE


def test_kinematics_probe_obeys_isaac51_startup_and_frozen_stage_contract() -> (
    None
):
    source = KINEMATICS_PROBE.read_text(encoding="utf-8")
    required = {
        "SimulationApp",
        "LulaKinematicsSolver",
        "compute_forward_kinematics",
        "set_robot_base_pose",
        "follower_left_gripper_link",
        "d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf",
        "8.0.26",
        "HARD_BLOCKER_LULA_USD_FRAME_CORRESPONDENCE",
    }
    assert required <= set(source.split()) | {
        token for token in required if token in source
    }

    tree = ast.parse(source)
    simulation_app_line = min(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "SimulationApp")
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "SimulationApp"
            )
        )
    )
    forbidden_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            continue
        if any(
            module.startswith(("pxr", "omni", "isaacsim"))
            for module in modules
        ):
            forbidden_imports.append(node.lineno)
    assert forbidden_imports
    assert min(forbidden_imports) > simulation_app_line


def test_lift_onset_requires_two_positive_fk_steps_after_readback() -> None:
    frames = np.arange(208, 245)
    delta_z = np.zeros(frames.size, dtype=np.float64)
    delta_z[frames == 237] = 0.003
    delta_z[frames == 238] = 0.004
    delta_z[frames > 238] = 0.001
    z_positions = np.cumsum(delta_z)

    result = detect_lift_onset(
        frames=frames,
        delta_z=delta_z,
        z_positions=z_positions,
        close_command_start_frame=222,
        readback_response_start_frame=226,
    )

    assert result.lift_onset_frame == 237
    assert result.threshold > 0.0
    assert result.candidates
    selected = next(
        candidate
        for candidate in result.candidates
        if candidate["frame"] == 237
    )
    assert selected["two_consecutive_above_threshold"] is True
    assert selected["positive_cumulative_z_to_end"] is True


def test_lift_onset_uses_directional_noise_not_downward_approach() -> None:
    frames = np.arange(208, 245)
    delta_z = np.zeros(frames.size, dtype=np.float64)
    delta_z[frames <= 219] = -0.002
    delta_z[frames == 220] = 0.00001
    delta_z[frames == 221] = 0.00002
    delta_z[frames == 222] = -0.001
    delta_z[frames == 234] = 0.0011
    delta_z[frames == 235] = 0.0004
    delta_z[frames >= 240] = 0.0007
    z_positions = np.cumsum(delta_z)

    result = detect_lift_onset(
        frames=frames,
        delta_z=delta_z,
        z_positions=z_positions,
        close_command_start_frame=222,
        readback_response_start_frame=226,
    )

    assert result.lift_onset_frame == 234
    assert result.threshold < 0.0004


def test_kinematics_report_binds_episode_fk_placement_and_ik() -> None:
    assert KINEMATICS_REPORT.is_file(), (
        f"missing kinematics report: {KINEMATICS_REPORT}"
    )
    report = json.loads(KINEMATICS_REPORT.read_text(encoding="utf-8"))

    assert report["status"] in {"PASS", "PARTIAL", "FAIL"}
    assert report["task8"] == "NOT_RUN"
    assert report["bindings"]["episode"]["sha256"] == (
        "f073a21c6a790e738e36085d791482924a82832ca6d80cece04a26353b9fc745"
    )
    assert report["bindings"]["urdf"]["sha256"] == (
        "d9e4b32723ee71dfce26fb4e78546cfcfef147b2d7dbf5e53e3620e3d8aa96bd"
    )
    assert report["bindings"]["stage"]["sha256"] == (
        "d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf"
    )
    assert report["bindings"]["articulation_path"] == (
        "/World/follower_left/vx300s_left/root_joint"
    )
    assert report["bindings"]["base_frame"] == "follower_left_base_link"
    assert report["bindings"]["end_effector_frame"] == (
        "follower_left_gripper_link"
    )

    records = report["episode_fk"]["records"]
    assert len(records) == 37
    required = {
        "frame",
        "qpos_arm_6d",
        "action_arm_6d",
        "ee_position_robot_base_m",
        "ee_orientation_wxyz",
        "ee_delta_m",
        "ee_delta_z_m",
        "gripper_action",
        "gripper_qpos",
    }
    assert all(required <= record.keys() for record in records)
    assert all(
        np.isfinite(record["ee_position_robot_base_m"]).all()
        for record in records
    )

    lift = report["lift_detection"]
    assert 226 < lift["lift_onset_frame"] <= 244
    assert lift["candidates"]
    assert report["placement"]["bottle_axis"]["status"] == "PASS"
    assert report["placement"]["bottle_axis"]["a_world_m"] != (
        report["placement"]["bottle_axis"]["b_world_m"]
    )
    assert report["ik"]["position_tolerance_m"] == pytest.approx(0.001)
    assert report["ik"]["orientation_tolerance_rad"] == pytest.approx(0.005)
    assert report["ik"]["waypoints"]
