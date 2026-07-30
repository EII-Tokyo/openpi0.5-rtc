from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from tools.aloha1_mapping.task_frames import ClosureError
from tools.aloha1_mapping.task_frames import closure_error
from tools.aloha1_mapping.task_frames import rigid_transform
from tools.aloha1_mapping.task_frames import tabletop_task_frame
from tools.aloha1_mapping.task_frames import validate_rigid_transform

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_rigid_transform_constructs_finite_homogeneous_matrix() -> None:
    transform = rigid_transform(np.eye(3), [1.0, -2.0, 3.0])

    np.testing.assert_allclose(
        transform,
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, -2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
    assert validate_rigid_transform(transform) is transform


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.full((4, 4), np.nan), "finite 4x4"),
        (np.eye(3), "finite 4x4"),
        (
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                ]
            ),
            "homogeneous",
        ),
        (
            np.array(
                [
                    [2.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            "orthogonal",
        ),
        (np.diag([-1.0, 1.0, 1.0, 1.0]), "determinant"),
    ],
)
def test_invalid_rigid_transforms_are_rejected(matrix: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_rigid_transform(matrix)


def test_tabletop_frame_moves_stage_table_top_to_zero() -> None:
    world_from_table = tabletop_task_frame(
        table_center_world_m=[0.0, 0.0, -0.0984000015258789],
        table_size_world_m=[1.1, 0.6, 0.015],
    )
    table_from_world = np.linalg.inv(world_from_table)
    top_world = np.array([0.0, 0.0, -0.0909000015258789, 1.0])

    assert world_from_table[:3, 3] == pytest.approx([0.0, 0.0, -0.0909000015258789])
    assert table_from_world @ top_world == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_world_object_grasp_and_base_chains_close() -> None:
    world_from_base = rigid_transform(np.eye(3), [-0.4695, -0.019, 0.02])
    world_from_object = rigid_transform(np.eye(3), [0.01, -0.16, -0.058])
    object_from_gripper = rigid_transform(np.eye(3), [0.0, 0.0, 0.15])
    world_from_gripper = world_from_object @ object_from_gripper
    base_from_gripper = np.linalg.inv(world_from_base) @ world_from_gripper

    error = closure_error(world_from_gripper, world_from_base @ base_from_gripper)

    assert isinstance(error, ClosureError)
    assert error.translation_m < 1e-12
    assert error.rotation_rad < 1e-12


def test_closure_error_reports_translation_and_rotation_separately() -> None:
    expected = np.eye(4)
    observed = rigid_transform(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        [0.03, -0.04, 0.0],
    )

    error = closure_error(expected, observed)

    assert error.translation_m == pytest.approx(0.05)
    assert error.rotation_rad == pytest.approx(np.pi / 2.0)


def test_frozen_tabletop_config_marks_digital_calibration_boundary() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs/aloha1_table_task_frame.yaml").read_text(encoding="utf-8"))

    assert config["status"] == "DIGITAL_STAGE_READBACK_NOT_REAL_CALIBRATION"
    assert config["stage"]["sha256"] == ("2b3f76365ed67532f478d995ae859a88b5639975ac07cb7ac8a53ac679e8205c")
    assert config["table"]["center_world_m"] == [0.0, 0.0, -0.0075]
    assert config["table"]["size_world_m"] == [1.1, 0.6, 0.015]
    assert config["task_world"]["world_from_task_translation_m"] == [
        0.0,
        0.0,
        0.0,
    ]
    assert config["boundaries"]["task8"] == "NOT_RUN"
