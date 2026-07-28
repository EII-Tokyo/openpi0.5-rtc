"""Traceable supplier-CAD to ALOHA Viper finger-link transforms.

The production OBJ vertices are expressed in the STEP assembly's global
coordinates, in metres.  The transform therefore:

1. removes the common embedded-finger placement;
2. applies the proper supplier-assembly -> Stage gripper rotation; and
3. removes the legal closed-state finger-link origin for the selected side.

At the legal closed qpos, composing the finger-link pose reconstructs the
supplier CAD pair exactly.  No reflection, negative scale, or per-side roll is
used.
"""

from __future__ import annotations

from collections.abc import Sequence

CAD_FINGER_COMMON_PLACEMENT_TRANSLATION_M = (
    0.0000998902576627736,
    -0.43029999973392,
    0.42680133373174,
)

# Supplier embedded-finger local axes -> STEP global axes.
CAD_FINGER_LOCAL_TO_GLOBAL_ROTATION = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
)

# Supplier gripper/common assembly axes -> Stage gripper/finger-link axes:
# local X -> link +Y, local Y -> link +Z, local Z -> link +X.
CAD_ASSEMBLY_TO_FINGER_LINK_ROTATION = (
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)

FINGER_LINK_CLOSED_ORIGIN_M = {
    "left": (0.0687, 0.021, 0.0),
    "right": (0.0687, -0.021, 0.0),
}

# The mathematically equivalent STEP-global -> Stage gripper rotation after
# removing the common placement.
CAD_GLOBAL_TO_GRIPPER_ROTATION = (
    (0.0, -1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
)


def _clean(value: float) -> float:
    return 0.0 if abs(value) < 1.0e-15 else value


def _matvec(
    matrix: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> tuple[float, float, float]:
    values = tuple(float(value) for value in vector)
    return tuple(
        _clean(
            sum(matrix[row][column] * values[column] for column in range(3))
        )
        for row in range(3)
    )


def transform_point(
    side: str,
    point_cad_global_m: Sequence[float],
) -> tuple[float, float, float]:
    """Map a baked STEP-global OBJ point into one finger-link local frame."""
    if side not in FINGER_LINK_CLOSED_ORIGIN_M:
        raise ValueError(f"unsupported finger side: {side}")
    relative_global = tuple(
        float(value) - CAD_FINGER_COMMON_PLACEMENT_TRANSLATION_M[index]
        for index, value in enumerate(point_cad_global_m)
    )
    gripper_point = _matvec(
        CAD_GLOBAL_TO_GRIPPER_ROTATION,
        relative_global,
    )
    closed_origin = FINGER_LINK_CLOSED_ORIGIN_M[side]
    return tuple(
        _clean(gripper_point[index] - closed_origin[index])
        for index in range(3)
    )


def transform_vector(
    vector_cad_assembly_local: Sequence[float],
) -> tuple[float, float, float]:
    """Map a supplier assembly-local direction into the Stage link frame."""
    return _matvec(
        CAD_ASSEMBLY_TO_FINGER_LINK_ROTATION,
        vector_cad_assembly_local,
    )


def cad_global_to_finger_link_matrix(
    side: str,
) -> tuple[tuple[float, float, float, float], ...]:
    """Return the explicit column-vector affine matrix for a selected side."""
    if side not in FINGER_LINK_CLOSED_ORIGIN_M:
        raise ValueError(f"unsupported finger side: {side}")
    rotated_origin = _matvec(
        CAD_GLOBAL_TO_GRIPPER_ROTATION,
        CAD_FINGER_COMMON_PLACEMENT_TRANSLATION_M,
    )
    closed_origin = FINGER_LINK_CLOSED_ORIGIN_M[side]
    translation = tuple(
        -rotated_origin[index] - closed_origin[index]
        for index in range(3)
    )
    rows = [
        (
            *CAD_GLOBAL_TO_GRIPPER_ROTATION[row],
            _clean(translation[row]),
        )
        for row in range(3)
    ]
    rows.append((0.0, 0.0, 0.0, 1.0))
    return tuple(rows)


def determinant3(matrix: Sequence[Sequence[float]]) -> float:
    a, b, c = matrix[0][:3]
    d, e, f = matrix[1][:3]
    g, h, i = matrix[2][:3]
    return (
        a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g)
    )
