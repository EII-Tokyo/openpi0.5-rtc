from __future__ import annotations

import numpy as np
import pytest

from tools.aloha1_mapping import aloha_kinematics_reference as reference


def test_reference_constants_match_aloha_vx300s_description() -> None:
    expected_slist = np.asarray(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -0.12705, 0.0, 0.0],
            [0.0, 1.0, 0.0, -0.42705, 0.0, 0.05955],
            [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0],
            [0.0, 1.0, 0.0, -0.42705, 0.0, 0.35955],
            [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0],
        ],
        dtype=np.float64,
    ).T
    expected_home = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.536494],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.42705],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    assert reference.SLIST.shape == (6, 6)
    assert reference.Slist is reference.SLIST
    assert reference.M.shape == (4, 4)
    assert np.array_equal(reference.SLIST, expected_slist)
    assert np.array_equal(reference.M, expected_home)
    assert reference.SOURCE_CLASS == "aloha_vx300s"
    assert reference.SOURCE_SHA256 == (
        "9412f1496f0cf1f3e23995ba3f0c10f250624cdd3798274a7191b1cad6248388"
    )


def test_vec_to_se3_maps_angular_and_linear_components() -> None:
    result = reference.vec_to_se3([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    expected = np.asarray(
        [
            [0.0, -3.0, 2.0, 4.0],
            [3.0, 0.0, -1.0, 5.0],
            [-2.0, 1.0, 0.0, 6.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    assert result.shape == (4, 4)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "invalid_twist",
    [
        [0.0] * 5,
        [0.0] * 7,
        np.zeros((6, 1)),
        [0.0, 0.0, np.nan, 0.0, 0.0, 0.0],
    ],
)
def test_vec_to_se3_rejects_invalid_twists(invalid_twist: object) -> None:
    with pytest.raises(ValueError, match=r"shape \(6,\)|finite"):
        reference.vec_to_se3(invalid_twist)


def test_fk_space_at_zero_is_home_configuration() -> None:
    transform = reference.fk_space(np.zeros(6))

    assert transform.shape == (4, 4)
    assert np.array_equal(transform, reference.M)


@pytest.mark.parametrize(
    "invalid_q",
    [
        0.0,
        [0.0] * 5,
        [0.0] * 7,
        np.zeros((6, 1)),
        [0.0, 0.0, np.nan, 0.0, 0.0, 0.0],
        [0.0, 0.0, np.inf, 0.0, 0.0, 0.0],
        [0.0, 0.0, -np.inf, 0.0, 0.0, 0.0],
    ],
)
def test_fk_space_rejects_invalid_joint_vectors(invalid_q: object) -> None:
    with pytest.raises(ValueError, match=r"q.*shape \(6,\)|q.*finite"):
        reference.fk_space(invalid_q)


@pytest.mark.parametrize("joint_index", range(6))
@pytest.mark.parametrize("delta", [-1.0e-6, 1.0e-6])
def test_each_joint_perturbation_produces_repeatable_se3(
    joint_index: int,
    delta: float,
) -> None:
    q = np.zeros(6)
    q[joint_index] = delta

    first = reference.fk_space(q)
    second = reference.fk_space(q)
    rotation = first[:3, :3]

    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert first[3] == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1.0e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1.0e-12)


def test_positive_waist_rotates_home_position_toward_positive_y() -> None:
    positive = reference.fk_space([1.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0])
    negative = reference.fk_space([-1.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0])

    assert positive[1, 3] > 0.0
    assert negative[1, 3] < 0.0
    assert positive[1, 3] == pytest.approx(-negative[1, 3])


def test_positive_shoulder_moves_home_position_forward_and_down() -> None:
    home = reference.fk_space(np.zeros(6))
    positive = reference.fk_space([0.0, 1.0e-4, 0.0, 0.0, 0.0, 0.0])
    negative = reference.fk_space([0.0, -1.0e-4, 0.0, 0.0, 0.0, 0.0])

    assert positive[0, 3] > home[0, 3]
    assert positive[2, 3] < home[2, 3]
    assert negative[0, 3] < home[0, 3]
    assert negative[2, 3] > home[2, 3]
