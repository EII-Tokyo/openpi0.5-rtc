"""Pure state and screenshot contracts for the public-CAD gripper review."""

from __future__ import annotations

from collections.abc import Iterable
import math
from pathlib import Path
from typing import Any

VIEW_IDS = ("true_top", "true_bottom", "tip_end", "base_oblique")
STATE_IDS = ("closed", "open")
VERSION_IDS = ("raw", "annotated")
MM_TO_M = 0.001

_VIEW_BASES = {
    "true_top": {
        "camera_forward": (0.0, 0.0, -1.0),
        "image_up": (0.0, 1.0, 0.0),
        "image_right": (1.0, 0.0, 0.0),
        "description": "camera on CAD +Z looking toward CAD -Z",
    },
    "true_bottom": {
        "camera_forward": (0.0, 0.0, 1.0),
        "image_up": (0.0, 1.0, 0.0),
        "image_right": (-1.0, 0.0, 0.0),
        "description": "camera on CAD -Z looking toward CAD +Z",
    },
    "tip_end": {
        "camera_forward": (0.0, 1.0, 0.0),
        "image_up": (0.0, 0.0, 1.0),
        "image_right": (1.0, 0.0, 0.0),
        "description": "camera at finger-tip CAD -Y end looking toward CAD +Y",
    },
    "base_oblique": {
        "camera_forward": (
            0.0,
            -math.sqrt(0.5),
            -math.sqrt(0.5),
        ),
        "image_up": (0.0, -math.sqrt(0.5), math.sqrt(0.5)),
        "image_right": (-1.0, 0.0, 0.0),
        "description": (
            "camera on the gripper-base CAD +Y/+Z oblique, looking toward "
            "CAD -Y/-Z so the shell does not hide both installed fingers"
        ),
    },
}


def points_mm_to_m(points_mm: Iterable[Iterable[float]]) -> list[list[float]]:
    """Convert an immutable CAD-millimetre point set for Blender metres."""
    return [
        [float(value) * MM_TO_M for value in point] for point in points_mm
    ]


def view_basis(view_id: str) -> dict[str, object]:
    """Return the proven CAD-axis camera basis for a review view."""
    try:
        return dict(_VIEW_BASES[view_id])
    except KeyError as exc:
        raise ValueError(f"unsupported view_id: {view_id}") from exc


def _dot(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def orthographic_frame(
    *,
    points_mm: Iterable[tuple[float, float, float]],
    view_id: str,
    resolution: tuple[int, int],
    margin: float,
) -> dict[str, object]:
    """Fit a fixed orthographic camera to the union of paired-state points."""
    points = list(points_mm)
    if not points:
        raise ValueError("points_mm must not be empty")
    if resolution[0] <= 0 or resolution[1] <= 0:
        raise ValueError("resolution must be positive")
    if margin <= 1.0:
        raise ValueError("margin must be greater than 1")
    basis = view_basis(view_id)
    target = tuple(
        0.5
        * (
            min(point[axis] for point in points)
            + max(point[axis] for point in points)
        )
        for axis in range(3)
    )
    relative = [
        tuple(point[axis] - target[axis] for axis in range(3))
        for point in points
    ]
    right = basis["image_right"]
    up = basis["image_up"]
    horizontal = [_dot(point, right) for point in relative]
    vertical = [_dot(point, up) for point in relative]
    horizontal_span = max(horizontal) - min(horizontal)
    vertical_span = max(vertical) - min(vertical)
    aspect = resolution[0] / resolution[1]
    ortho_height = max(
        vertical_span * margin,
        horizontal_span * margin / aspect,
    )
    ortho_width = ortho_height * aspect
    diagonal = math.sqrt(
        sum(
            (
                max(point[axis] for point in points)
                - min(point[axis] for point in points)
            )
            ** 2
            for axis in range(3)
        )
    )
    camera_distance = max(diagonal * 2.0, 100.0)
    forward = basis["camera_forward"]
    camera_location = tuple(
        target[axis] - forward[axis] * camera_distance for axis in range(3)
    )
    return {
        **basis,
        "target_mm": target,
        "camera_location_mm": camera_location,
        "ortho_height_mm": ortho_height,
        "ortho_width_mm": ortho_width,
        "resolution": resolution,
        "margin": margin,
    }


def capture_plan(*, output_root: Path) -> list[dict[str, str]]:
    """Create the eight paired CAD capture records without touching disk."""
    root = output_root.resolve()
    raw_root = root / "screenshots_raw"
    annotated_root = root / "screenshots_annotated"
    return [
        {
            "state_id": state_id,
            "view_id": view_id,
            "camera_key": f"paired_{view_id}",
            "raw_path": str(
                raw_root / f"{state_id}_{view_id}_raw.png"
            ),
            "annotated_path": str(
                annotated_root / f"{state_id}_{view_id}_annotated.png"
            ),
        }
        for state_id in STATE_IDS
        for view_id in VIEW_IDS
    ]


def infer_static_cad_state(
    *,
    cad_positive_center_mm: float,
    cad_negative_center_mm: float,
    urdf_closed_positive_center_mm: float,
    urdf_open_positive_center_mm: float,
) -> dict[str, Any]:
    """Classify the static CAD pose by signed finger-center separation."""
    cad_half_separation = 0.5 * (
        cad_positive_center_mm - cad_negative_center_mm
    )
    closed_residual = abs(
        cad_half_separation - urdf_closed_positive_center_mm
    )
    open_residual = abs(cad_half_separation - urdf_open_positive_center_mm)
    classification = (
        "CLOSED_REFERENCE"
        if closed_residual < open_residual
        else "OPEN_REFERENCE"
    )
    return {
        "status": "PASS" if closed_residual != open_residual else "INCONCLUSIVE",
        "classification": classification,
        "cad_half_separation_mm": cad_half_separation,
        "urdf_closed_half_separation_mm": urdf_closed_positive_center_mm,
        "urdf_open_half_separation_mm": urdf_open_positive_center_mm,
        "closed_residual_mm": closed_residual,
        "open_residual_mm": open_residual,
        "method": (
            "compare purchase-confirmed Simple Viper CAD finger AABB-center "
            "half-separation against generated-URDF mesh placements at the "
            "source closed/open joint limits"
        ),
    }


def state_translations_mm(
    *,
    open_delta_mm: float,
) -> dict[str, dict[str, list[float]]]:
    """Return CAD-frame translations from the static closed reference."""
    if open_delta_mm <= 0:
        raise ValueError("open_delta_mm must be positive")
    zero = [0.0, 0.0, 0.0]
    return {
        "closed": {
            "cad_positive_x_finger": zero.copy(),
            "cad_negative_x_finger": zero.copy(),
        },
        "open": {
            "cad_positive_x_finger": [open_delta_mm, 0.0, 0.0],
            "cad_negative_x_finger": [-open_delta_mm, 0.0, 0.0],
        },
    }


def required_capture_inventory() -> list[str]:
    """Return the exact raw/annotated screenshot inventory."""
    return [
        f"{state}_{view}_{version}.png"
        for state in STATE_IDS
        for view in VIEW_IDS
        for version in VERSION_IDS
    ]
