from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import ARM_INDICES, GRIPPER_INDICES, RLINF_ROBOTWIN_PI05_SEMANTICS
from .safety_filter import finite_or_raise


@dataclass(frozen=True)
class ConvertedAction:
    raw_model_action: np.ndarray
    canonical_action: np.ndarray
    semantics: str


def rlinf_robotwin_delta_to_canonical_target(
    raw_action: np.ndarray,
    current_qpos: np.ndarray,
) -> ConvertedAction:
    """Convert RLinf RobotTwin pi0.5 output into canonical ALOHA target.

    RLinf RobotTwin pi0.5 applies delta to arm joints and keeps grippers absolute.
    This function is intentionally explicit; it does not apply any OpenPI joint
    flip or gripper angular transform.
    """

    raw_action = np.asarray(raw_action, dtype=np.float32)
    current_qpos = np.asarray(current_qpos, dtype=np.float32)
    if raw_action.shape != (14,):
        raise ValueError(f"raw_action shape {raw_action.shape} != (14,)")
    if current_qpos.shape != (14,):
        raise ValueError(f"current_qpos shape {current_qpos.shape} != (14,)")
    finite_or_raise(raw_action, "raw_action")
    finite_or_raise(current_qpos, "current_qpos")

    target = current_qpos.copy()
    target[list(ARM_INDICES)] = current_qpos[list(ARM_INDICES)] + raw_action[list(ARM_INDICES)]
    target[list(GRIPPER_INDICES)] = raw_action[list(GRIPPER_INDICES)]
    return ConvertedAction(
        raw_model_action=raw_action.copy(),
        canonical_action=target,
        semantics=RLINF_ROBOTWIN_PI05_SEMANTICS.name,
    )


def passthrough_absolute_action(raw_action: np.ndarray) -> ConvertedAction:
    raw_action = np.asarray(raw_action, dtype=np.float32)
    if raw_action.shape != (14,):
        raise ValueError(f"raw_action shape {raw_action.shape} != (14,)")
    finite_or_raise(raw_action, "raw_action")
    return ConvertedAction(raw_action.copy(), raw_action.copy(), "absolute_passthrough")

