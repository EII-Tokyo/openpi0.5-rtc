from __future__ import annotations

import ast
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

from tools.aloha1_mapping.task7b2_horizontal_grasp import canonical_horizontal_signature
from tools.aloha1_mapping.task7b2_horizontal_grasp import evaluate_horizontal_trial
from tools.aloha1_mapping.task7b2_horizontal_grasp import summarize_horizontal_trials
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
RUNTIME_SCRIPT = ROOT / "tools/validate_aloha1_task7b2_horizontal_grasp.py"
EXPECTED_CSPACE = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]


def _passing_horizontal_trial(*, trial_index: int = 0) -> dict:
    return {
        "trial_index": trial_index,
        "fresh_world_reset": True,
        "bottle_dynamic_during_settle": True,
        "support_contact_before_grasp": True,
        "axis_horizontal_pass": True,
        "gripper_axis_perpendicular_pass": True,
        "vertical_descent_pass": True,
        "ik_reachable": True,
        "left_physical_contact_before_lift": True,
        "right_physical_contact_before_lift": True,
        "contact_points_in_body_interval": True,
        "bottle_left_support": True,
        "bilateral_contact_through_hold": True,
        "hold_drop_m": 0.002,
        "drop_gate_m": 0.010,
        "finite_state": True,
        "persistent_penetration": False,
        "numerical_ejection": False,
        "forbidden_contact": False,
        "forbidden_constraint": False,
        "surface_gripper_used": False,
        "parent_attachment_used": False,
        "contact_lost_before_hold": False,
        "free_fall_after_contact_loss": False,
        "rotation_induced_escape": False,
        "normal_force_decay": False,
        "continuous_slip": False,
        "phase_frame_counts": {
            "support_settle": 60,
            "vertical_descent": 24,
            "hold_end": 120,
        },
        "joint_trajectories": [[0.0, -0.96], [0.01, -0.95]],
        "contacts": [
            {"frame": 120, "side": "left", "impulse_ns": 0.01},
            {"frame": 120, "side": "right", "impulse_ns": 0.01},
        ],
        "bottle_poses": [
            {"frame": 0, "position_m": [0.0, 0.0, 0.0]},
            {"frame": 240, "position_m": [0.0, 0.0, 0.02]},
        ],
        "runtime_seconds": 99.0,
        "artifact_absolute_path": "/tmp/attempt-a",
    }


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


def test_horizontal_trial_passes_only_complete_physical_hold() -> None:
    result = evaluate_horizontal_trial(_passing_horizontal_trial())

    assert result["status"] == "PASS"
    assert result["failure_mode"] == "stable_hold"
    assert result["task8"] == "NOT_RUN"


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"support_contact_before_grasp": False}, "support_settle_failed"),
        ({"axis_horizontal_pass": False}, "horizontal_geometry_failed"),
        (
            {"gripper_axis_perpendicular_pass": False},
            "gripper_axis_correspondence_failed",
        ),
        ({"vertical_descent_pass": False}, "vertical_ik_unreachable"),
        (
            {"left_physical_contact_before_lift": False},
            "contact_not_established",
        ),
        (
            {
                "bilateral_contact_through_hold": False,
                "contact_lost_before_hold": True,
                "free_fall_after_contact_loss": True,
            },
            "contact_lost_then_free_fall",
        ),
        (
            {
                "bilateral_contact_through_hold": True,
                "continuous_slip": True,
                "hold_drop_m": 0.02,
            },
            "bilateral_contact_continuous_slip",
        ),
        (
            {
                "bilateral_contact_through_hold": False,
                "rotation_induced_escape": True,
            },
            "rotation_induced_escape",
        ),
        (
            {
                "bilateral_contact_through_hold": False,
                "normal_force_decay": True,
            },
            "normal_force_decay",
        ),
        (
            {"persistent_penetration": True},
            "numerical_penetration_or_ejection",
        ),
        ({"bottle_left_support": False}, "support_clearance_failed"),
        ({"forbidden_contact": True}, "forbidden_contact"),
        (
            {"bilateral_contact_through_hold": False},
            "inconclusive",
        ),
    ],
)
def test_horizontal_failure_classifications_are_exact(
    updates: dict,
    expected: str,
) -> None:
    trial = _passing_horizontal_trial()
    trial.update(updates)

    result = evaluate_horizontal_trial(trial)

    assert result["status"] == "FAIL"
    assert result["failure_mode"] == expected


def test_horizontal_failure_precedence_is_fail_closed() -> None:
    trial = _passing_horizontal_trial()
    trial.update(
        {
            "axis_horizontal_pass": False,
            "support_contact_before_grasp": False,
            "left_physical_contact_before_lift": False,
            "persistent_penetration": True,
            "bottle_left_support": False,
        }
    )

    assert evaluate_horizontal_trial(trial)["failure_mode"] == (
        "horizontal_geometry_failed"
    )


def test_horizontal_signature_excludes_runtime_and_artifact_path() -> None:
    first = _passing_horizontal_trial()
    second = _passing_horizontal_trial()
    second["runtime_seconds"] = 1.0
    second["artifact_absolute_path"] = "/different/attempt"

    assert canonical_horizontal_signature(first) == (
        canonical_horizontal_signature(second)
    )

    second["bottle_poses"][-1]["position_m"][2] = 0.019
    assert canonical_horizontal_signature(first) != (
        canonical_horizontal_signature(second)
    )


def test_horizontal_summary_requires_twenty_fresh_deterministic_passes() -> None:
    smoke = summarize_horizontal_trials([_passing_horizontal_trial()])
    assert smoke["status"] == "PARTIAL"
    assert smoke["trial_count"] == 1
    assert smoke["pass_count"] == 1

    trials = [
        _passing_horizontal_trial(trial_index=index) for index in range(20)
    ]
    accepted = summarize_horizontal_trials(trials)
    assert accepted["status"] == "PASS"
    assert accepted["trial_count"] == 20
    assert accepted["pass_count"] == 20
    assert accepted["fresh_world_reset_count"] == 20
    assert accepted["unique_deterministic_signature_count"] == 1

    trials[-1]["hold_drop_m"] = 0.02
    rejected = summarize_horizontal_trials(trials)
    assert rejected["status"] == "FAIL"
    assert rejected["pass_count"] == 19


def test_horizontal_runtime_source_contract() -> None:
    assert RUNTIME_SCRIPT.is_file(), (
        f"missing Isaac 5.1 horizontal runtime: {RUNTIME_SCRIPT}"
    )
    source = RUNTIME_SCRIPT.read_text(encoding="utf-8")
    required = {
        "SimulationApp",
        "open_stage",
        "set_solve_articulation_contact_last(True)",
        "LulaKinematicsSolver",
        "compute_inverse_kinematics",
        "/Bottle500",
        "user_confirmed_table",
        "GetKinematicEnabledAttr().Set(False)",
        "support_settle",
        "open_pregrasp",
        "vertical_descent",
        "bilateral_contact",
        "release_dynamic",
        "support_clear",
        "hold_end",
        "subscribe_contact_report_events",
        "overview",
        "gripper_closeup",
        "frame_manifest",
        "runtime_trial_signature",
    }
    for token in required:
        assert token in source, f"missing required runtime token: {token}"

    forbidden = {
        "SurfaceGripper",
        "CreateFixedJoint",
        "parent_attachment",
        "APPROACH_FRAME = 98",
        "LIFT_DELTA = -0.08",
        "source_layer.Save",
    }
    for token in forbidden:
        assert token not in source, f"forbidden runtime token: {token}"


def test_horizontal_runtime_constructs_simulation_app_before_isaac_imports() -> None:
    source = RUNTIME_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    simulation_app_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (
                isinstance(node.func, ast.Name)
                and node.func.id == "SimulationApp"
            )
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "SimulationApp"
            )
        )
    ]
    assert simulation_app_lines, "SimulationApp construction not found"
    first_app_line = min(simulation_app_lines)

    protected_import_lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            protected_import_lines.extend(
                node.lineno
                for alias in node.names
                if alias.name.split(".", maxsplit=1)[0]
                in {"pxr", "omni", "isaacsim"}
            )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".", maxsplit=1)[0]
            in {
                "pxr",
                "omni",
                "isaacsim",
            }
        ):
            protected_import_lines.append(node.lineno)

    assert protected_import_lines
    assert first_app_line < min(protected_import_lines)


def test_horizontal_runtime_requires_complete_two_view_frame_streams() -> None:
    source = RUNTIME_SCRIPT.read_text(encoding="utf-8")
    assert 'VIDEO_VIEWS = ("overview", "gripper_closeup")' in source
    assert '"missing_physics_frames"' in source
    assert '"phase_frame_ranges"' in source
    assert '"camera_world_matrix"' in source
    assert '"render_fps"' in source
    assert '"first_physics_frame"' in source
    assert '"last_physics_frame"' in source
