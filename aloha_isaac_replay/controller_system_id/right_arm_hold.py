from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


RIGHT_ARM_14D_INDICES = tuple(range(7, 14))
RIGHT_ARM_JOINT_14D_INDICES = tuple(range(7, 13))
RIGHT_SHOULDER_14D_INDEX = 8


def _episode_id(path: str | Path) -> str:
    p = Path(path)
    if p.parent.name.startswith("key_region_"):
        return p.parent.name
    return p.stem


def analyze_right_arm_hold(
    path: str | Path,
    *,
    transition_frames: int = 5,
    tolerance: float = 1e-6,
    static_command_tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Detect whether runtime action froze or effectively stopped the right arm.

    RLT key-region runtime can hold right arm dimensions at the robot state
    latched when actor takeover starts. That data is still useful as executed
    runtime command evidence, but it is not a right-arm excitation dataset for
    controller identification. Some episodes are not exactly equal to qpos[0],
    but still carry almost no right-arm command variation after takeover; those
    are also excluded from right-arm controller ID.
    """
    episode_path = Path(path)
    with h5py.File(episode_path, "r") as h5:
        action = np.asarray(h5["action"][:], dtype=np.float64)
        qpos = np.asarray(h5["observations/qpos"][:], dtype=np.float64)
        reference = np.asarray(h5["reference_action"][:], dtype=np.float64) if "reference_action" in h5 else None
        reward = h5.attrs.get("reward", None)
        phase = h5.attrs.get("phase", None)
        is_key_region = h5.attrs.get("is_key_region", None)

    start = min(max(int(transition_frames), 0), action.shape[0])
    right_indices = list(RIGHT_ARM_14D_INDICES)
    right_joint_indices = list(RIGHT_ARM_JOINT_14D_INDICES)
    entry_qpos = qpos[0, right_indices]
    right_action_tail = action[start:, right_indices] if start < action.shape[0] else action[:, right_indices]
    right_joint_action_tail = (
        action[start:, right_joint_indices] if start < action.shape[0] else action[:, right_joint_indices]
    )
    hold_error = right_action_tail - entry_qpos
    right_joint_std = np.std(right_joint_action_tail, axis=0) if right_joint_action_tail.size else np.zeros(6)

    ref_stats = None
    ref_diff_max = None
    if reference is not None:
        right_ref_tail = reference[start:, right_indices] if start < reference.shape[0] else reference[:, right_indices]
        ref_stats = {
            "right_arm_reference_std_mean": float(np.mean(np.std(right_ref_tail, axis=0))),
            "right_shoulder_reference_min": float(np.min(reference[:, RIGHT_SHOULDER_14D_INDEX])),
            "right_shoulder_reference_max": float(np.max(reference[:, RIGHT_SHOULDER_14D_INDEX])),
        }
        ref_diff_max = float(np.max(np.abs(action[:, right_indices] - reference[:, right_indices])))

    hold_max_abs_after_transition = float(np.max(np.abs(hold_error))) if hold_error.size else 0.0
    hold_detected = bool(hold_max_abs_after_transition <= float(tolerance))
    static_command_detected = bool(float(np.max(right_joint_std)) <= float(static_command_tolerance))
    hold_or_static_detected = bool(hold_detected or static_command_detected)
    right_shoulder_action = action[:, RIGHT_SHOULDER_14D_INDEX]
    right_shoulder_qpos = qpos[:, RIGHT_SHOULDER_14D_INDEX]

    return {
        "episode_id": _episode_id(episode_path),
        "path": str(episode_path),
        "frames": int(action.shape[0]),
        "phase": str(phase) if phase is not None else None,
        "reward": int(reward) if reward is not None else None,
        "is_key_region": bool(is_key_region) if is_key_region is not None else None,
        "transition_frames_ignored": int(start),
        "has_reference_action": reference is not None,
        "right_arm_hold_detected": hold_detected,
        "right_arm_static_command_detected": static_command_detected,
        "right_arm_hold_or_static_detected": hold_or_static_detected,
        "right_arm_hold_tolerance": float(tolerance),
        "right_arm_static_command_tolerance": float(static_command_tolerance),
        "right_arm_hold_max_abs_after_transition": hold_max_abs_after_transition,
        "right_arm_action_joint_std_mean_after_transition": float(np.mean(right_joint_std)),
        "right_arm_action_joint_std_max_after_transition": float(np.max(right_joint_std)),
        "right_arm_action_reference_max_abs_diff": ref_diff_max,
        "right_shoulder_action_min": float(np.min(right_shoulder_action)),
        "right_shoulder_action_max": float(np.max(right_shoulder_action)),
        "right_shoulder_action_std": float(np.std(right_shoulder_action)),
        "right_shoulder_qpos_min": float(np.min(right_shoulder_qpos)),
        "right_shoulder_qpos_max": float(np.max(right_shoulder_qpos)),
        "right_shoulder_qpos_std": float(np.std(right_shoulder_qpos)),
        "usable_for_right_arm_controller_id": not hold_or_static_detected,
        "usable_for_right_arm_hold_stability": hold_or_static_detected,
        "reference_stats": ref_stats,
    }


def summarize_right_arm_hold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "episode_count": 0,
            "right_arm_hold_detected_count": 0,
            "right_arm_controller_id_usable_count": 0,
            "all_selected_have_right_arm_hold": False,
            "status": "NO_EPISODES",
        }

    hold_count = sum(1 for row in rows if row["right_arm_hold_detected"])
    static_count = sum(1 for row in rows if row["right_arm_static_command_detected"])
    hold_or_static_count = sum(1 for row in rows if row["right_arm_hold_or_static_detected"])
    usable_count = sum(1 for row in rows if row["usable_for_right_arm_controller_id"])
    max_hold_error = max(float(row["right_arm_hold_max_abs_after_transition"]) for row in rows)
    mean_action_std = float(np.mean([row["right_arm_action_joint_std_mean_after_transition"] for row in rows]))
    return {
        "episode_count": len(rows),
        "right_arm_hold_detected_count": hold_count,
        "right_arm_static_command_detected_count": static_count,
        "right_arm_hold_or_static_detected_count": hold_or_static_count,
        "right_arm_controller_id_usable_count": usable_count,
        "all_selected_have_right_arm_hold": hold_count == len(rows),
        "all_selected_have_right_arm_hold_or_static": hold_or_static_count == len(rows),
        "max_hold_error_after_transition": max_hold_error,
        "mean_right_arm_action_joint_std_after_transition": mean_action_std,
        "status": "BLOCKED_RLT_RIGHT_ARM_HOLD_OR_STATIC_COMMAND"
        if usable_count == 0
        else "PARTIAL_RIGHT_ARM_EXCITATION",
    }
