"""Pure geometry gates for the ALOHA 1 gripper orientation diagnostic."""

from __future__ import annotations

from typing import Any


def physical_side_order(left_center_y: float, right_center_y: float) -> bool:
    """Return whether the physical-left center is below physical-right in Y."""
    return float(left_center_y) < float(right_center_y)


def surface_normal_gate(
    left_inward_normal_y: float,
    right_inward_normal_y: float,
    *,
    threshold: float = 0.90,
) -> bool:
    """Require the two principal gripping surfaces to face one another."""
    return (
        float(left_inward_normal_y) >= threshold
        and float(right_inward_normal_y) <= -threshold
    )


def classify_orientation(
    *,
    side_order_ok: bool,
    inward_normals_ok: bool,
    closed_aperture_m: float,
    open_aperture_m: float,
    crossed_centerline: bool,
) -> dict[str, Any]:
    """Classify the minimal orientation gates without viewport judgment."""
    aperture_monotonic = float(open_aperture_m) > float(closed_aperture_m)
    gates = {
        "physical_side_order": bool(side_order_ok),
        "inward_normals": bool(inward_normals_ok),
        "aperture_monotonic": aperture_monotonic,
        "no_crossed_centerline": not bool(crossed_centerline),
    }
    return {
        "status": (
            "ASSEMBLY_ORIENTATION_CONFIRMED"
            if all(gates.values())
            else "ASSEMBLY_ORIENTATION_ERROR"
        ),
        "gates": gates,
    }
