from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from tools.aloha1_mapping.grasp_20cm_five_pose_ik import apply_frozen_bottle_transform
from tools.aloha1_mapping.grasp_20cm_five_pose_ik import canonical_five_pose_signature
from tools.aloha1_mapping.grasp_20cm_five_pose_ik import compose_initial_command
from tools.aloha1_mapping.grasp_20cm_five_pose_ik import derive_sample_geometry
from tools.aloha1_mapping.grasp_20cm_five_pose_ik import line_yaw_distance_deg
from tools.aloha1_mapping.grasp_20cm_five_pose_ik import minimum_pairwise_ee_distance_m
from tools.aloha1_mapping.grasp_20cm_five_pose_ik import minimum_pairwise_line_yaw_separation_deg
from tools.aloha1_mapping.grasp_20cm_five_pose_ik import place_bottle_center_and_yaw
from tools.aloha1_mapping.grasp_20cm_five_pose_ik import sample_bottle_center_yaw_candidates
from tools.aloha1_mapping.grasp_20cm_five_pose_ik import sample_initial_arm_joint_candidates
from tools.aloha1_mapping.grasp_20cm_five_pose_ik import select_diverse_records

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/aloha1_grasp_20cm_five_pose_ik.yaml"


def _horizontal_world_from_object() -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    return result


def _candidate_records() -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"sample_{index + 1:02d}",
            "preflight_status": "PASS",
            "bottle_line_yaw_deg": yaw,
            "initial_ee_position_world_m": [0.06 * index, 0.0, 0.30],
        }
        for index, yaw in enumerate((4.0, 34.0, 64.0, 94.0, 124.0))
    ]


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


def test_bottle_transform_places_cad_center_on_vertical_centerline() -> None:
    nominal = _horizontal_world_from_object()
    nominal[:3, 3] = [0.2, -0.1, 0.03]

    result = place_bottle_center_and_yaw(
        nominal_world_from_object=nominal,
        geometric_center_local_m=[0.0, 0.0, 0.103],
        desired_center_xy_m=[0.0, 0.08],
        yaw_delta_rad=np.deg2rad(47.0),
    )

    center = (
        result[:3, :3] @ np.array([0.0, 0.0, 0.103])
        + result[:3, 3]
    )
    assert center[:2] == pytest.approx([0.0, 0.08], abs=1e-12)
    assert center[2] == pytest.approx(
        (nominal[:3, :3] @ np.array([0.0, 0.0, 0.103]) + nominal[:3, 3])[2]
    )
    assert np.linalg.det(result[:3, :3]) == pytest.approx(1.0)


def test_rotated_ab_and_grasp_transform_follow_object_yaw() -> None:
    world_from_object = place_bottle_center_and_yaw(
        nominal_world_from_object=_horizontal_world_from_object(),
        geometric_center_local_m=[0.0, 0.0, 0.103],
        desired_center_xy_m=[0.0, 0.08],
        yaw_delta_rad=np.deg2rad(82.0),
    )

    result = derive_sample_geometry(
        world_from_object=world_from_object,
        a_local_m=[0.0, 0.0, 0.0],
        b_local_m=[0.0, 0.0, 0.206],
        object_from_gripper=np.eye(4),
    )

    assert result["line_yaw_deg"] == pytest.approx(82.0)
    assert result["axis_to_world_z_deg"] == pytest.approx(90.0)
    assert result["world_from_gripper"] == pytest.approx(world_from_object)


def test_line_yaw_distance_is_modulo_180_degrees() -> None:
    assert line_yaw_distance_deg(5.0, 175.0) == pytest.approx(10.0)
    assert line_yaw_distance_deg(15.0, 47.0) == pytest.approx(32.0)


def test_five_selected_samples_meet_yaw_and_ee_distance_gates() -> None:
    selected = select_diverse_records(
        records=_candidate_records(),
        count=5,
        minimum_line_yaw_separation_deg=25.0,
        minimum_ee_separation_m=0.050,
    )

    assert len(selected) == 5
    assert minimum_pairwise_line_yaw_separation_deg(selected) >= 25.0
    assert minimum_pairwise_ee_distance_m(selected) >= 0.050


def test_joint_candidate_sampling_is_fixed_seed_and_within_limits() -> None:
    lower = np.array([-1.0, -0.8, -1.2, -1.5, -1.0, -2.0])
    upper = np.array([1.0, 0.9, 1.3, 1.5, 1.1, 2.0])

    first = sample_initial_arm_joint_candidates(
        lower_limits=lower,
        upper_limits=upper,
        seed=2026073102,
        count=256,
    )
    second = sample_initial_arm_joint_candidates(
        lower_limits=lower,
        upper_limits=upper,
        seed=2026073102,
        count=256,
    )

    assert np.array_equal(first, second)
    assert np.all(first >= lower)
    assert np.all(first <= upper)


@pytest.mark.parametrize(
    ("formal_sample_index", "x_relation", "y_sign"),
    [
        (0, "zero", "positive"),
        (1, "negative", "any"),
        (3, "zero", "negative"),
    ],
)
def test_bottle_candidate_sampling_obeys_formal_spatial_structure(
    formal_sample_index: int,
    x_relation: str,
    y_sign: str,
) -> None:
    records = sample_bottle_center_yaw_candidates(
        center_xy_bounds={"minimum": [-0.30, -0.20], "maximum": [0.10, 0.25]},
        yaw_domain_deg=[0.0, 180.0],
        seed=2026073102,
        count=8,
        formal_sample_index=formal_sample_index,
    )
    repeat = sample_bottle_center_yaw_candidates(
        center_xy_bounds={"minimum": [-0.30, -0.20], "maximum": [0.10, 0.25]},
        yaw_domain_deg=[0.0, 180.0],
        seed=2026073102,
        count=8,
        formal_sample_index=formal_sample_index,
    )

    assert records == repeat
    assert all(0.0 <= record["bottle_line_yaw_deg"] < 180.0 for record in records)
    if x_relation == "zero":
        assert all(record["bottle_center_xy_m"][0] == 0.0 for record in records)
    else:
        assert all(record["bottle_center_xy_m"][0] < 0.0 for record in records)
    if y_sign == "positive":
        assert all(record["bottle_center_xy_m"][1] > 0.0 for record in records)
    elif y_sign == "negative":
        assert all(record["bottle_center_xy_m"][1] < 0.0 for record in records)


def test_apply_frozen_transform_preserves_t_o_g_and_input_profile() -> None:
    nominal = _horizontal_world_from_object()
    nominal[:3, 3] = [-0.10, -0.15, 0.034]
    object_from_gripper = np.eye(4)
    object_from_gripper[:3, 3] = [0.0, 0.0, 0.069]
    original_grasp = nominal @ object_from_gripper
    profile = {
        "kinematics": {
            "placement": {
                "placement_matrix": nominal.tolist(),
                "bottle_axis": {
                    "a_world_m": (nominal @ [0.0, 0.0, 0.0, 1.0])[:3].tolist(),
                    "b_world_m": (nominal @ [0.0, 0.0, 0.206, 1.0])[:3].tolist(),
                    "grasp_point_world_m": original_grasp[:3, 3].tolist(),
                },
                "target_poses": {
                    "object_from_gripper": object_from_gripper.tolist(),
                    "grasp_ee_position_world_m": original_grasp[:3, 3].tolist(),
                    "pregrasp_ee_position_world_m": (
                        original_grasp[:3, 3] + [0.0, 0.0, 0.08]
                    ).tolist(),
                    "lift_ee_position_world_m": (
                        original_grasp[:3, 3] + [0.0, 0.0, 0.21]
                    ).tolist(),
                    "orientation_world_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
            }
        }
    }
    original_matrix = np.asarray(
        profile["kinematics"]["placement"]["placement_matrix"]
    ).copy()
    frozen = place_bottle_center_and_yaw(
        nominal_world_from_object=nominal,
        geometric_center_local_m=[0.0, 0.0, 0.103],
        desired_center_xy_m=[0.0, 0.10],
        yaw_delta_rad=np.deg2rad(35.0),
    )

    result = apply_frozen_bottle_transform(
        profile,
        world_from_object=frozen,
    )

    placement = result["kinematics"]["placement"]
    expected_world_from_gripper = frozen @ object_from_gripper
    assert placement["placement_matrix"] == pytest.approx(frozen)
    assert placement["target_poses"]["object_from_gripper"] == pytest.approx(
        object_from_gripper
    )
    assert placement["target_poses"][
        "grasp_ee_position_world_m"
    ] == pytest.approx(expected_world_from_gripper[:3, 3])
    assert (
        np.asarray(placement["target_poses"]["pregrasp_ee_position_world_m"])
        - expected_world_from_gripper[:3, 3]
    ) == pytest.approx([0.0, 0.0, 0.08])
    assert np.asarray(
        profile["kinematics"]["placement"]["placement_matrix"]
    ) == pytest.approx(original_matrix)


def test_canonical_signature_is_deterministic_and_pose_sensitive() -> None:
    records = [
        {
            "sample_id": "sample_01",
            "candidate_index": 7,
            "bottle_geometric_center_world_m": [0.0, 0.08, 0.034],
            "bottle_line_yaw_deg": 15.0,
            "world_from_object": np.eye(4).tolist(),
            "initial_arm_q_rad": [0.0, -0.9, 1.1, 0.0, -0.3, 0.0],
            "initial_ee_position_world_m": [-0.2, 0.0, 0.3],
        }
    ]

    first = canonical_five_pose_signature(records)
    second = canonical_five_pose_signature(records)
    changed = [dict(records[0], bottle_line_yaw_deg=16.0)]

    assert first == second
    assert len(first) == 64
    assert first != canonical_five_pose_signature(changed)


def test_initial_command_replaces_only_explicit_six_arm_dofs() -> None:
    baseline = np.arange(9, dtype=float)
    sampled_arm = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])

    result = compose_initial_command(
        baseline,
        sampled_arm,
        arm_dof_indices=[0, 1, 2, 3, 4, 5],
    )

    assert result[:6] == pytest.approx(sampled_arm)
    assert result[6:] == pytest.approx(baseline[6:])


def test_initial_command_rejects_duplicate_or_out_of_range_indices() -> None:
    with pytest.raises(ValueError, match="unique"):
        compose_initial_command(
            np.zeros(9),
            np.zeros(6),
            arm_dof_indices=[0, 1, 2, 3, 4, 4],
        )
    with pytest.raises(ValueError, match="out of range"):
        compose_initial_command(
            np.zeros(9),
            np.zeros(6),
            arm_dof_indices=[0, 1, 2, 3, 4, 9],
        )
