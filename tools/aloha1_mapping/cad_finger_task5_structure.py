"""Pure gates for the supplier-CAD follower-left Task 5 structure test."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

FINGER_DOF_NAMES = (
    "vx300s_left_left_finger",
    "vx300s_left_right_finger",
)

LEGAL_POSES_M = {
    "closed": (0.021, -0.021),
    "partial": (0.039, -0.039),
    "maximum_legal_aperture": (0.057, -0.057),
}

POSE_ALIASES = {
    "open": "maximum_legal_aperture",
}

VIEW_RADII_M = {
    "true_top": 0.95,
    "true_bottom": 0.95,
    "tip_end": 0.58,
    "base_oblique_tool": 0.38,
    "base_oblique_top": 0.25,
    "base_oblique_closing": 0.14,
}

_GRIPPER_PROP_VISUALS = (
    "/workcell/vx300s_left/"
    "vx300s_left_gripper_prop_link/visuals"
)
_GRIPPER_SHELL_VISUALS = (
    "/workcell/vx300s_left/"
    "vx300s_left_gripper_link/visuals"
)
GLOBAL_SESSION_HIDDEN_VISUALS = (
    "/workcell/vx300s_left/vx300s_left_gripper_prop_link",
    "/workcell/vx300s_left/vx300s_left_camera_focus",
)
VIEW_HIDDEN_VISUALS = {
    "true_top": (),
    "true_bottom": (),
    "tip_end": (),
    "base_oblique": (_GRIPPER_SHELL_VISUALS,),
}

_KEPT_ROBOT_VISUAL_TOKENS = (
    "/vx300s_left_gripper_link/visuals",
    "/vx300s_left_gripper_prop_link/visuals",
    "/vx300s_left_left_finger_link/visuals",
    "/vx300s_left_right_finger_link/visuals",
)

LIMIT_TOLERANCE_M = 1.0e-6
READBACK_TOLERANCE_M = 1.0e-6
SYMMETRY_TOLERANCE_M = 1.0e-6


def hide_non_target_robot_visual(path: str, prim_name: str) -> bool:
    """Select imported robot visuals that obscure the gripper evidence."""

    return (
        prim_name == "visuals"
        and path.startswith("/workcell/vx300s_left/")
        and not any(token in path for token in _KEPT_ROBOT_VISUAL_TOKENS)
    )


def hide_robot_debug_container(path: str, prim_name: str) -> bool:
    """Select imported render-only site and collider display containers."""

    return (
        path.startswith("/workcell/vx300s_left/")
        and prim_name in {"sites", "collisions"}
    )


def hide_non_target_robot_gprim(path: str) -> bool:
    """Select render geometry outside the two fingers and gripper shell."""

    if not path.startswith("/workcell/vx300s_left/"):
        return False
    kept = (
        "/vx300s_left_left_finger_link/visuals/"
        "diagnostic_supplier_cad_left_finger/",
        "/vx300s_left_right_finger_link/visuals/"
        "diagnostic_supplier_cad_right_finger/",
        "/vx300s_left_gripper_link/visuals/",
    )
    return not any(token in path for token in kept)


def summarize_image_projection(
    points: Sequence[Sequence[float]],
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Summarize Isaac Camera world-to-image projection readback."""

    finite = [
        [float(point[0]), float(point[1])]
        for point in points
        if len(point) >= 2
        and math.isfinite(float(point[0]))
        and math.isfinite(float(point[1]))
    ]
    if not finite:
        raise ValueError("projection contains no finite image point")
    minimum = [min(point[index] for point in finite) for index in (0, 1)]
    maximum = [max(point[index] for point in finite) for index in (0, 1)]
    center = [
        (minimum[index] + maximum[index]) / 2.0 for index in (0, 1)
    ]
    return {
        "finite_point_count": len(finite),
        "bbox_min_px": minimum,
        "bbox_max_px": maximum,
        "bbox_center_px": center,
        "fully_in_frame": (
            minimum[0] >= 0.0
            and minimum[1] >= 0.0
            and maximum[0] < float(width)
            and maximum[1] < float(height)
        ),
    }


def validate_pose_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate legal injected poses without claiming dynamic drive control."""

    by_name = {str(record["state"]): record for record in records}
    expected = set(LEGAL_POSES_M)
    if set(by_name) != expected:
        raise ValueError(
            f"expected exactly {sorted(expected)}, got {sorted(by_name)}"
        )

    legal = []
    readback = []
    symmetric = []
    gaps = []
    for state, target in LEGAL_POSES_M.items():
        record = by_name[state]
        left, right = (float(value) for value in record["readback_m"])
        limits = record["limits_m"]
        left_limit = [float(value) for value in limits["left"]]
        right_limit = [float(value) for value in limits["right"]]
        legal.append(
            left_limit[0] - LIMIT_TOLERANCE_M
            <= left
            <= left_limit[1] + LIMIT_TOLERANCE_M
            and right_limit[0] - LIMIT_TOLERANCE_M
            <= right
            <= right_limit[1] + LIMIT_TOLERANCE_M
        )
        readback.append(
            abs(left - target[0]) <= READBACK_TOLERANCE_M
            and abs(right - target[1]) <= READBACK_TOLERANCE_M
        )
        symmetric.append(abs(left + right) <= SYMMETRY_TOLERANCE_M)
        gaps.append(float(record["surface_gap_m"]))

    monotonic = gaps[0] < gaps[1] < gaps[2]
    gates = {
        "all_readbacks_within_limits": all(legal),
        "injected_pose_readback": all(readback),
        "left_right_pose_symmetry": all(symmetric),
        "aperture_monotonicity": monotonic,
        "default_zero_pose_rejected": all(
            abs(value) > 0.0
            for target in LEGAL_POSES_M.values()
            for value in target
        ),
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "surface_gaps_m": dict(
            zip(LEGAL_POSES_M, gaps, strict=True)
        ),
        "acceptance_boundary": (
            "Legal-pose injection/readback and geometry only. This does not "
            "prove drive tracking, PhysX mimic, contact, or grasp."
        ),
    }


def drive_mimic_status(
    *,
    physx_mimic_api_present: bool,
    left_max_force: float,
    right_max_force: float,
) -> dict[str, Any]:
    """Classify whether dynamic mimic/control can be accepted."""

    positive_force = left_max_force > 0.0 and right_max_force > 0.0
    if physx_mimic_api_present and positive_force:
        status = "NOT_RUN"
        reason = "runtime mimic trajectory is still required"
    elif not physx_mimic_api_present and not positive_force:
        status = "FAIL"
        reason = (
            "PhysxMimicJointAPI is absent and both authored drive maxForce "
            "values are non-positive"
        )
    elif not physx_mimic_api_present:
        status = "FAIL"
        reason = "PhysxMimicJointAPI is absent"
    else:
        status = "FAIL"
        reason = "authored drive maxForce is not positive"
    return {
        "status": status,
        "physx_mimic_api_present": physx_mimic_api_present,
        "positive_drive_max_force": positive_force,
        "reason": reason,
        "teleport_or_pose_injection_counts_as_dynamic_pass": False,
    }
