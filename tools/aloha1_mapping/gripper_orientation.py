"""Pure geometry gates for the ALOHA 1 gripper orientation diagnostic."""

from __future__ import annotations

from typing import Any

import numpy as np


def expected_capture_names() -> list[str]:
    """Return the deterministic two-state, three-view capture inventory."""
    return [f"{state}_{view}.png" for state in ("closed", "open") for view in ("closing_axis", "top", "isometric")]


def physical_side_order(left_center_y: float, right_center_y: float) -> bool:
    """Return whether physical-left is on +Y and physical-right is on -Y."""
    return float(left_center_y) > float(right_center_y)


def surface_normal_gate(
    left_inward_normal_y: float,
    right_inward_normal_y: float,
    *,
    threshold: float = 0.90,
) -> bool:
    """Require the two principal gripping surfaces to face one another."""
    return float(left_inward_normal_y) <= -threshold and float(right_inward_normal_y) >= threshold


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
        "status": ("ASSEMBLY_ORIENTATION_CONFIRMED" if all(gates.values()) else "ASSEMBLY_ORIENTATION_ERROR"),
        "gates": gates,
    }


def finger_state_targets(
    limits: dict[str, tuple[float, float]],
    state: str,
) -> dict[str, float]:
    """Return the imported asymmetric prismatic targets for one state."""
    if state == "closed":
        return {"left": limits["left"][0], "right": limits["right"][1]}
    if state == "open":
        return {"left": limits["left"][1], "right": limits["right"][0]}
    raise ValueError(f"unsupported finger state: {state}")


def obj_text(name: str, points: np.ndarray, faces: np.ndarray) -> str:
    """Serialize one triangular world-space mesh deterministically."""
    lines = [f"o {name}"]
    lines.extend(f"v {value[0]:.12g} {value[1]:.12g} {value[2]:.12g}" for value in np.asarray(points))
    lines.extend(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}" for face in np.asarray(faces))
    return "\n".join(lines) + "\n"


def inward_surface_normal_y(
    points: np.ndarray,
    faces: np.ndarray,
    side: str,
    *,
    alignment_threshold: float = 0.90,
) -> float:
    """Return the area-weighted Y normal for inward-facing triangles."""
    if side not in {"left", "right"}:
        raise ValueError(f"unsupported physical side: {side}")
    triangles = np.asarray(points, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    doubled_areas = np.linalg.norm(cross, axis=1)
    valid = doubled_areas > 1e-15
    normals = np.zeros_like(cross)
    normals[valid] = cross[valid] / doubled_areas[valid, None]
    expected_sign = -1.0 if side == "left" else 1.0
    inward = valid & (normals[:, 1] * expected_sign >= alignment_threshold)
    if not np.any(inward):
        raise ValueError(f"no inward-facing triangles found for {side}")
    return float(np.average(normals[inward, 1], weights=doubled_areas[inward]))
