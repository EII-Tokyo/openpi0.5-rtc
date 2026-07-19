from __future__ import annotations

import numpy as np

from aloha_isaac_replay.rl.drive_target_env import summarize_step
from aloha_isaac_replay.rl.drive_target_env import target_limit_violations
from aloha_isaac_replay.rl.drive_target_env import tracking_groups
from aloha_isaac_replay.rl.drive_target_env import tracking_step_errors


def test_tracking_groups_include_arm_and_gripper_for_left_replay() -> None:
    dof_names = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_left_finger",
        "left_right_finger",
    ]
    groups = tracking_groups(
        dof_names,
        side="left",
        replay_mode="left_arm_and_gripper",
        finger_dof_names={"left_finger": "left_left_finger", "right_finger": "left_right_finger"},
    )
    assert groups["arm"] == [0, 1, 2, 3, 4, 5]
    assert groups["gripper"] == [6, 7]
    assert groups["controlled"] == [0, 1, 2, 3, 4, 5, 6, 7]


def test_tracking_and_limit_metrics_are_bounded_for_reward_readiness() -> None:
    target = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    actual = np.array([0.0, 0.49, 1.02], dtype=np.float64)
    groups = {"controlled": [0, 1, 2], "arm": [0, 1], "gripper": [2]}
    tracking = tracking_step_errors(target=target, actual=actual, groups=groups)
    assert tracking["controlled"]["max_abs_error"] == 0.020000000000000018

    limits = np.array([[-1.0, 1.0], [-1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    violations = target_limit_violations(target=target, limits=limits, groups=groups)
    assert violations["controlled"]["max_violation"] == 0.0

    summary = summarize_step(
        step_index=3,
        target=target,
        actual=actual,
        limits=limits,
        groups=groups,
        max_controlled_error=0.021,
    )
    assert summary.step_index == 3
    assert summary.reward_ready is True


def test_reward_readiness_fails_on_limit_violation() -> None:
    target = np.array([0.0, 1.2], dtype=np.float64)
    actual = np.array([0.0, 1.2], dtype=np.float64)
    limits = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float64)
    groups = {"controlled": [0, 1]}
    summary = summarize_step(
        step_index=0,
        target=target,
        actual=actual,
        limits=limits,
        groups=groups,
        max_controlled_error=0.01,
    )
    assert summary.reward_ready is False
    assert summary.target_limit_controlled_max_violation > 0.0

