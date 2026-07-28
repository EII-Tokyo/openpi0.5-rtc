from __future__ import annotations

from tools.aloha1_mapping.gripper_orientation import classify_orientation
from tools.aloha1_mapping.gripper_orientation import physical_side_order
from tools.aloha1_mapping.gripper_orientation import surface_normal_gate


def test_physical_side_order_requires_left_below_right_y() -> None:
    assert physical_side_order(0.49, 0.51) is True
    assert physical_side_order(0.51, 0.49) is False


def test_surface_normal_gate_requires_opposed_inward_normals() -> None:
    assert surface_normal_gate(0.999, -0.999) is True
    assert surface_normal_gate(-0.999, 0.999) is False


def test_orientation_classification_requires_monotonic_aperture() -> None:
    result = classify_orientation(
        side_order_ok=True,
        inward_normals_ok=True,
        closed_aperture_m=0.001,
        open_aperture_m=0.073,
        crossed_centerline=False,
    )

    assert result["status"] == "ASSEMBLY_ORIENTATION_CONFIRMED"
    assert result["gates"]["aperture_monotonic"] is True
