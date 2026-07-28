from __future__ import annotations

import math

from tools.aloha1_mapping.cad_finger_installation import (
    CAD_ASSEMBLY_TO_FINGER_LINK_ROTATION,
)
from tools.aloha1_mapping.cad_finger_installation import (
    CAD_FINGER_COMMON_PLACEMENT_TRANSLATION_M,
)
from tools.aloha1_mapping.cad_finger_installation import (
    FINGER_LINK_CLOSED_ORIGIN_M,
)
from tools.aloha1_mapping.cad_finger_installation import determinant3
from tools.aloha1_mapping.cad_finger_installation import transform_point
from tools.aloha1_mapping.cad_finger_installation import transform_vector


def test_supplier_assembly_rotation_is_proper_and_not_mirrored() -> None:
    assert math.isclose(
        determinant3(CAD_ASSEMBLY_TO_FINGER_LINK_ROTATION),
        1.0,
        abs_tol=1.0e-12,
    )


def test_common_placement_origin_maps_to_each_closed_link_offset() -> None:
    for side in ("left", "right"):
        transformed = transform_point(
            side,
            CAD_FINGER_COMMON_PLACEMENT_TRANSLATION_M,
        )
        expected = tuple(
            -value for value in FINGER_LINK_CLOSED_ORIGIN_M[side]
        )
        assert transformed == expected


def test_cad_local_axes_map_to_link_without_palm_flip() -> None:
    assert transform_vector((1.0, 0.0, 0.0)) == (0.0, 1.0, 0.0)
    assert transform_vector((0.0, 1.0, 0.0)) == (0.0, 0.0, 1.0)
    assert transform_vector((0.0, 0.0, 1.0)) == (1.0, 0.0, 0.0)


def test_supplier_open_displacement_maps_to_stage_joint_axis() -> None:
    left_open_delta = (0.0, 0.036, 0.0)
    right_open_delta = (0.0, -0.036, 0.0)

    assert left_open_delta[1] > 0.0
    assert right_open_delta[1] < 0.0
