from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class FingerLimits:
    left_close: float = 0.021
    left_open: float = 0.057
    right_close: float = -0.021
    right_open: float = -0.057


DEFAULT_VX300S_FINGER_LIMITS = FingerLimits()


@dataclasses.dataclass(frozen=True)
class GripperQposCalibration:
    """Dataset-side observed qpos convention for one ALOHA gripper scalar.

    The default calibration maps raw closed/open qpos anchors to the normalized
    ALOHA convention 0=closed and 1=open.  A loaded soft-bottle replay may use a
    lower raw anchor that is not physically closed; in that case
    ``standard_closed_value`` records the effective contact-width interpretation
    while keeping the original HDF5 qpos untouched.
    """

    closed_value: float = 0.0
    open_value: float = 1.0
    standard_closed_value: float = 0.0
    standard_open_value: float = 1.0
    source: str = "standard ALOHA normalized gripper qpos; do not substitute action gripper values"


DEFAULT_GRIPPER_QPOS_CALIBRATION = GripperQposCalibration()


def _require_normalized(value: np.ndarray | float) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if np.any(array < -1e-6) or np.any(array > 1.0 + 1e-6):
        raise ValueError(f"normalized gripper command must be in [0, 1], got range [{array.min()}, {array.max()}]")
    return np.clip(array, 0.0, 1.0)


def standard_gripper_to_isaac(value: np.ndarray | float, limits: FingerLimits = DEFAULT_VX300S_FINGER_LIMITS) -> dict[str, np.ndarray]:
    """Map one normalized ALOHA gripper command to two VX300S finger targets.

    The current dataset convention is 0 = closed, 1 = open. In the running URDF,
    `right_finger` mimics `left_finger` with multiplier -1, so the numeric directions
    are opposite.
    """
    g = _require_normalized(value)
    left = limits.left_close + g * (limits.left_open - limits.left_close)
    right = limits.right_close + g * (limits.right_open - limits.right_close)
    return {"left_finger": left, "right_finger": right}


def standard_gripper_value_for_symmetric_finger_gap(
    finger_gap: float,
    limits: FingerLimits = DEFAULT_VX300S_FINGER_LIMITS,
) -> float:
    """Return the normalized command whose opposed finger distance is ``finger_gap``.

    This helper is for explicit calibration reports.  It does not infer object
    size from qpos; callers must provide a measured or clearly labelled effective
    contact width.
    """

    left_target = float(finger_gap) / 2.0
    span = limits.left_open - limits.left_close
    if abs(span) < 1e-12:
        raise ValueError("left finger limits must differ")
    return float(np.clip((left_target - limits.left_close) / span, 0.0, 1.0))


def gripper_qpos_calibration_from_loaded_contact(
    *,
    raw_open_value: float,
    raw_contact_value: float,
    effective_contact_width: float,
    limits: FingerLimits = DEFAULT_VX300S_FINGER_LIMITS,
    standard_open_value: float = 1.0,
    source: str = "loaded soft-bottle qpos calibration",
) -> GripperQposCalibration:
    """Build an explicit affine qpos calibration from open and loaded anchors.

    ``raw_contact_value`` is a loaded-contact qpos plateau, not a closed-air
    gripper value.  The returned calibration must therefore be treated as a
    diagnostic/contact-proxy interpretation layer for the named replay window.
    """

    standard_contact_value = standard_gripper_value_for_symmetric_finger_gap(effective_contact_width, limits)
    if abs(raw_open_value - raw_contact_value) < 1e-12:
        raise ValueError("loaded gripper open and contact qpos anchors must differ")
    return GripperQposCalibration(
        closed_value=float(raw_contact_value),
        open_value=float(raw_open_value),
        standard_closed_value=float(standard_contact_value),
        standard_open_value=float(standard_open_value),
        source=source,
    )


def standard_gripper_qpos_to_isaac_fingers(
    value: np.ndarray | float,
    side: str,
    calibration: GripperQposCalibration = DEFAULT_GRIPPER_QPOS_CALIBRATION,
    limits: FingerLimits = DEFAULT_VX300S_FINGER_LIMITS,
) -> dict[str, np.ndarray]:
    """Map observed HDF5 qpos[6]/qpos[13] to VX300S finger qpos targets.

    This function is intentionally separate from any action/command mapping. The
    collected HDF5 has shown that qpos[6] and action[6] can live in very different
    numeric spaces, so kinematic replay must consume observed gripper qpos only.
    """
    if side not in {"left", "right"}:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    raw = np.asarray(value, dtype=np.float64)
    span = calibration.open_value - calibration.closed_value
    if abs(span) < 1e-12:
        raise ValueError("gripper qpos calibration open_value and closed_value must differ")
    normalized_raw = (raw - calibration.closed_value) / span
    standard_value = calibration.standard_closed_value + normalized_raw * (
        calibration.standard_open_value - calibration.standard_closed_value
    )
    fingers = standard_gripper_to_isaac(standard_value, limits)
    return {
        f"{side}/left_finger": fingers["left_finger"],
        f"{side}/right_finger": fingers["right_finger"],
    }


def isaac_gripper_to_standard(
    left_finger: np.ndarray | float,
    right_finger: np.ndarray | float | None = None,
    limits: FingerLimits = DEFAULT_VX300S_FINGER_LIMITS,
) -> np.ndarray:
    left = np.asarray(left_finger, dtype=np.float64)
    normalized = (left - limits.left_close) / (limits.left_open - limits.left_close)
    if right_finger is not None:
        right = np.asarray(right_finger, dtype=np.float64)
        expected_right = limits.right_close + normalized * (limits.right_open - limits.right_close)
        if not np.allclose(right, expected_right, atol=1e-4):
            raise ValueError("right finger does not match VX300S mimic direction for the supplied left finger")
    return np.clip(normalized, 0.0, 1.0)


def validate_gripper_direction(limits: FingerLimits = DEFAULT_VX300S_FINGER_LIMITS) -> dict[str, float | bool]:
    closed = standard_gripper_to_isaac(0.0, limits)
    open_ = standard_gripper_to_isaac(1.0, limits)
    left_increases = bool(open_["left_finger"] > closed["left_finger"])
    right_decreases = bool(open_["right_finger"] < closed["right_finger"])
    return {
        "closed_left": float(closed["left_finger"]),
        "open_left": float(open_["left_finger"]),
        "closed_right": float(closed["right_finger"]),
        "open_right": float(open_["right_finger"]),
        "left_increases_when_opening": left_increases,
        "right_decreases_when_opening": right_decreases,
        "valid_mimic_opposite_direction": left_increases and right_decreases,
    }
