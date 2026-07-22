from __future__ import annotations

from dataclasses import dataclass


CANONICAL_JOINT_NAMES: tuple[str, ...] = (
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
)

ARM_INDICES: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
GRIPPER_INDICES: tuple[int, ...] = (6, 13)


@dataclass(frozen=True)
class ActionSpaceSemantics:
    name: str
    arm_is_delta: bool
    gripper_is_absolute: bool
    source: str


RLINF_ROBOTWIN_PI05_SEMANTICS = ActionSpaceSemantics(
    name="rlinf_robotwin_pi05",
    arm_is_delta=True,
    gripper_is_absolute=True,
    source=(
        "external/RLinf/rlinf/models/embodiment/openpi/dataconfig/"
        "robotwin_aloha_dataconfig.py"
    ),
)

