from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.aloha1_mapping.grasp_frame_contract import CAD_CONTACT_HELPER_SUFFIX
from tools.aloha1_mapping.grasp_frame_contract import CANONICAL_GRIPPER_LINK
from tools.aloha1_mapping.grasp_frame_contract import closure_error
from tools.aloha1_mapping.grasp_frame_contract import convert_contact_pose_to_gripper_pose
from tools.aloha1_mapping.grasp_frame_contract import derive_urdf_fixed_transform
from tools.aloha1_mapping.grasp_frame_contract import rigid_transform
from tools.aloha1_mapping.grasp_frame_contract import validate_native_gripper_dofs
from tools.aloha1_mapping.grasp_frame_contract import validate_rigid_transform

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / "generated/urdf/follower_left.urdf"


def _candidate_object_from_contact() -> np.ndarray:
    quaternion_wxyz = np.asarray(
        [
            0.0035300239827964607,
            0.9978157704829151,
            -0.06596341901664869,
            -0.00023336216765075293,
        ],
        dtype=np.float64,
    )
    w, x, y, z = quaternion_wxyz / np.linalg.norm(quaternion_wxyz)
    rotation = np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )
    return rigid_transform(rotation, [0.0, 0.0, 0.069])


def test_canonical_frame_names_are_unambiguous() -> None:
    assert CANONICAL_GRIPPER_LINK == "follower_left_ee_gripper_link"
    assert CAD_CONTACT_HELPER_SUFFIX == ("/follower_left_ee_gripper_link/aloha1_supplier_cad_clearance_grasp_frame")


def test_urdf_gripper_to_ee_gripper_fixed_chain_is_107_point_2_mm() -> None:
    gripper_from_ee = derive_urdf_fixed_transform(
        URDF,
        source_link="follower_left_gripper_link",
        target_link="follower_left_ee_gripper_link",
    )
    assert gripper_from_ee[:3, :3] == pytest.approx(np.eye(3), abs=1e-12)
    assert gripper_from_ee[:3, 3] == pytest.approx(
        [0.1072, 0.0, 0.0],
        abs=1e-12,
    )


def test_contact_candidate_is_converted_to_ee_gripper_not_relabelled() -> None:
    object_from_contact = _candidate_object_from_contact()
    gripper_link_from_contact = rigid_transform(
        np.eye(3),
        [0.13552080444282988, 0.0, 0.0],
    )
    gripper_link_from_ee = derive_urdf_fixed_transform(
        URDF,
        source_link="follower_left_gripper_link",
        target_link="follower_left_ee_gripper_link",
    )
    ee_from_contact = np.linalg.inv(gripper_link_from_ee) @ gripper_link_from_contact

    object_from_ee = convert_contact_pose_to_gripper_pose(
        object_from_contact=object_from_contact,
        gripper_from_contact=ee_from_contact,
    )

    assert ee_from_contact[:3, 3] == pytest.approx(
        [0.028320804442829875, 0.0, 0.0],
        abs=1e-12,
    )
    assert object_from_ee[:3, 3] == pytest.approx(
        [-0.028074343938903335, 0.0037281599602399626, 0.069],
        abs=1e-12,
    )
    assert object_from_ee != pytest.approx(object_from_contact, abs=1e-12)
    reconstructed = object_from_ee @ ee_from_contact
    error = closure_error(object_from_contact, reconstructed)
    assert error.translation_m < 1e-12
    assert error.rotation_rad < 1e-12


def test_world_object_gripper_and_base_chains_close() -> None:
    world_from_base = rigid_transform(
        np.eye(3),
        [-0.4695, -0.019, 0.11090000152587891],
    )
    world_from_object = rigid_transform(
        np.eye(3),
        [-0.058, -0.175, 0.0329],
    )
    object_from_gripper = rigid_transform(
        np.eye(3),
        [-0.004, 0.0005, 0.069],
    )
    world_from_gripper = world_from_object @ object_from_gripper
    base_from_gripper = np.linalg.inv(world_from_base) @ world_from_gripper
    error = closure_error(
        world_from_gripper,
        world_from_base @ base_from_gripper,
    )
    assert error.translation_m < 1e-12
    assert error.rotation_rad < 1e-12


@pytest.mark.parametrize(
    "bad_matrix",
    [
        np.diag([-1.0, 1.0, 1.0, 1.0]),
        np.diag([2.0, 1.0, 1.0, 1.0]),
        np.asarray(
            [
                [1.0, 0.0, 0.0, np.nan],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ],
)
def test_invalid_rigid_transforms_are_rejected(
    bad_matrix: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="rigid transform|rotation"):
        validate_rigid_transform(bad_matrix)


def test_native_grasp_editor_uses_only_active_left_finger() -> None:
    result = validate_native_gripper_dofs(
        cspace_position={"left_finger": 0.048316874538855845},
        pregrasp_cspace_position={"left_finger": 0.057},
        active_joint="left_finger",
        mimic_joint="right_finger",
    )
    assert result == {
        "active_joint": "left_finger",
        "mimic_joint": "right_finger",
        "active_keys": ["left_finger"],
        "status": "PASS",
    }


def test_native_grasp_editor_rejects_explicit_mimic_joint() -> None:
    with pytest.raises(ValueError, match="mimic"):
        validate_native_gripper_dofs(
            cspace_position={
                "left_finger": 0.048316874538855845,
                "right_finger": -0.048316874538855845,
            },
            pregrasp_cspace_position={
                "left_finger": 0.057,
                "right_finger": -0.057,
            },
            active_joint="left_finger",
            mimic_joint="right_finger",
        )
