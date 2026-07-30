"""Pure geometry gates for the complete ALOHA1 gripper grasp frame."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any


def _normalize(vector: Sequence[float]) -> list[float]:
    values = [float(value) for value in vector]
    if len(values) != 3:
        raise ValueError("expected a three-dimensional vector")
    length = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("vector must have finite nonzero length")
    return [value / length for value in values]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right, strict=True)))


def _cross(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [
        float(left[1] * right[2] - left[2] * right[1]),
        float(left[2] * right[0] - left[0] * right[2]),
        float(left[0] * right[1] - left[1] * right[0]),
    ]


def select_chebyshev_grasp_station(
    *,
    pad_interval_m: tuple[float, float],
    forbidden_max_x_m: Mapping[str, float],
    bottle_radius_m: float,
    pad_inward_normal_x: float,
    rejected_station_m: float,
) -> dict[str, Any]:
    pad_min, pad_max = (float(value) for value in pad_interval_m)
    radius = float(bottle_radius_m)
    if not math.isfinite(pad_min) or not math.isfinite(pad_max):
        raise ValueError("pad interval must be finite")
    if pad_min >= pad_max:
        raise ValueError("pad interval must be strictly increasing")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("bottle radius must be finite and positive")
    normal_x = float(pad_inward_normal_x)
    if not math.isfinite(normal_x) or not -1.0 <= normal_x <= 1.0:
        raise ValueError("pad inward-normal X component must be finite")
    if not forbidden_max_x_m:
        raise ValueError("at least one forbidden envelope is required")
    envelopes = {
        str(name): float(value)
        for name, value in forbidden_max_x_m.items()
    }
    if not all(math.isfinite(value) for value in envelopes.values()):
        raise ValueError("forbidden envelope coordinates must be finite")

    controlling_name, controlling_max = max(
        envelopes.items(),
        key=lambda item: item[1],
    )
    clearance_boundary = controlling_max + radius
    pad_normal_center_offset = radius * normal_x
    bottle_center_pad_interval = (
        pad_min + pad_normal_center_offset,
        pad_max + pad_normal_center_offset,
    )
    feasible_min = max(bottle_center_pad_interval[0], clearance_boundary)
    feasible_max = bottle_center_pad_interval[1]
    rejected_hard_clearance = (
        float(rejected_station_m) - clearance_boundary
    )
    rejected = {
        "station_m": float(rejected_station_m),
        "hard_clearance_m": rejected_hard_clearance,
        "inside_hard_feasible_interval": (
            feasible_min <= float(rejected_station_m) <= feasible_max
        ),
        "runtime_rejected": True,
        "reason": (
            "RUN13_GRIPPER_BAR_CONTACT_ENVELOPE_AND_NO_BILATERAL_CONTACT"
        ),
    }
    if feasible_min > feasible_max:
        return {
            "status": "FAIL",
            "classification": "NO_COMPLETE_GRIPPER_FEASIBLE_INTERVAL",
            "selection_rule": (
                "CHEBYSHEV_CENTER_OF_COMPLETE_GRIPPER_FEASIBLE_INTERVAL"
            ),
            "pad_interval_m": [pad_min, pad_max],
            "bottle_center_pad_interval_m": list(
                bottle_center_pad_interval
            ),
            "feasible_interval_m": [feasible_min, feasible_max],
            "selected_station_m": None,
            "selected_pad_contact_station_m": None,
            "selected_minimum_margin_m": None,
            "controlling_forbidden_envelope": controlling_name,
            "forbidden_max_x_m": envelopes,
            "bottle_radius_m": radius,
            "pad_inward_normal_x": normal_x,
            "pad_normal_bottle_center_offset_m": (
                pad_normal_center_offset
            ),
            "rejected_station": rejected,
        }

    selected = (feasible_min + feasible_max) / 2.0
    minimum_margin = min(
        selected - feasible_min,
        feasible_max - selected,
    )
    return {
        "status": "PASS",
        "classification": "COMPLETE_GRIPPER_FEASIBLE_INTERVAL_EXISTS",
        "selection_rule": (
            "CHEBYSHEV_CENTER_OF_COMPLETE_GRIPPER_FEASIBLE_INTERVAL"
        ),
        "pad_interval_m": [pad_min, pad_max],
        "bottle_center_pad_interval_m": list(bottle_center_pad_interval),
        "feasible_interval_m": [feasible_min, feasible_max],
        "selected_station_m": selected,
        "selected_pad_contact_station_m": (
            selected - pad_normal_center_offset
        ),
        "selected_minimum_margin_m": minimum_margin,
        "controlling_forbidden_envelope": controlling_name,
        "forbidden_max_x_m": envelopes,
        "selected_clearance_by_envelope_m": {
            name: selected - radius - value
            for name, value in sorted(envelopes.items())
        },
        "bottle_radius_m": radius,
        "pad_inward_normal_x": normal_x,
        "pad_normal_bottle_center_offset_m": pad_normal_center_offset,
        "rejected_station": rejected,
    }


def build_right_handed_grasp_frame(
    *,
    left_contact_reference_m: Sequence[float],
    right_contact_reference_m: Sequence[float],
    approach_axis_reference: Sequence[float],
    bottle_axis_reference: Sequence[float],
) -> dict[str, Any]:
    left = [float(value) for value in left_contact_reference_m]
    right = [float(value) for value in right_contact_reference_m]
    if len(left) != 3 or len(right) != 3:
        raise ValueError("contact points must be three-dimensional")
    origin = [
        (left_value + right_value) / 2.0
        for left_value, right_value in zip(left, right, strict=True)
    ]
    x_axis = _normalize(approach_axis_reference)
    raw_y = [
        left_value - right_value
        for left_value, right_value in zip(left, right, strict=True)
    ]
    y_without_x = [
        value - _dot(raw_y, x_axis) * axis_value
        for value, axis_value in zip(raw_y, x_axis, strict=True)
    ]
    y_axis = _normalize(y_without_x)
    z_axis = _normalize(_cross(x_axis, y_axis))
    bottle_axis = _normalize(bottle_axis_reference)
    bottle_alignment = _dot(z_axis, bottle_axis)
    if bottle_alignment < 1.0 - 1.0e-9:
        raise ValueError(
            "contact line and approach axes do not reproduce bottle axis"
        )
    rotation_rows = [
        [x_axis[0], y_axis[0], z_axis[0]],
        [x_axis[1], y_axis[1], z_axis[1]],
        [x_axis[2], y_axis[2], z_axis[2]],
    ]
    determinant = _dot(x_axis, _cross(y_axis, z_axis))
    matrix = [
        [*rotation_rows[0], origin[0]],
        [*rotation_rows[1], origin[1]],
        [*rotation_rows[2], origin[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return {
        "status": "PASS",
        "classification": "SUPPLIER_CAD_EFFECTIVE_PAD_CONTACT_FRAME",
        "origin_reference_m": origin,
        "approach_axis_reference": x_axis,
        "finger_line_axis_reference": y_axis,
        "bottle_axis_reference": z_axis,
        "reference_from_grasp": matrix,
        "rotation_determinant": determinant,
        "bottle_axis_alignment_dot": bottle_alignment,
        "ee_endpoint_is_grasp_center": False,
    }
