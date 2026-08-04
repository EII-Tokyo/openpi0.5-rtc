"""Pure safety checks for resuming teleoperation at the current pose."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Sequence


class RearmState(Enum):
    WAITING_FOR_OPEN = "waiting_for_open"
    WAITING_FOR_CLOSE = "waiting_for_close"
    READY = "ready"


class DualGripperRearmDetector:
    """Require a deliberate, debounced dual-gripper open-to-close gesture."""

    def __init__(
        self,
        *,
        open_threshold: float = 0.25,
        close_threshold: float = 0.0,
        debounce_samples: int = 3,
    ) -> None:
        if open_threshold <= close_threshold:
            raise ValueError("open threshold must exceed close threshold")
        if debounce_samples < 1:
            raise ValueError("debounce_samples must be at least one")
        self._open_threshold = float(open_threshold)
        self._close_threshold = float(close_threshold)
        self._debounce_samples = int(debounce_samples)
        self.reset()

    @property
    def state(self) -> RearmState:
        return self._state

    def reset(self) -> None:
        self._state = RearmState.WAITING_FOR_OPEN
        self._stable_samples = 0

    def update(self, positions: Mapping[str, float]) -> bool:
        if not positions:
            raise ValueError("at least one leader gripper is required")
        values = tuple(float(value) for value in positions.values())
        if self._state is RearmState.READY:
            return True

        if self._state is RearmState.WAITING_FOR_OPEN:
            condition = all(
                value >= self._open_threshold for value in values
            )
            if condition:
                self._stable_samples += 1
                if self._stable_samples >= self._debounce_samples:
                    self._state = RearmState.WAITING_FOR_CLOSE
                    self._stable_samples = 0
            else:
                self._stable_samples = 0
            return False

        condition = all(
            value <= self._close_threshold for value in values
        )
        if condition:
            self._stable_samples += 1
            if self._stable_samples >= self._debounce_samples:
                self._state = RearmState.READY
                return True
        else:
            self._stable_samples = 0
        return False


@dataclass(frozen=True)
class JointAlignmentReport:
    safe: bool
    max_error_rad: float
    pair_errors_rad: dict[str, float]


def hold_leader_arms_at_current_pose(
    leaders: Mapping[str, object],
    *,
    gravity_compensation: bool,
    read_positions: Callable[[object], Sequence[float]],
    disable_gravity_compensation: Callable[[object], None],
    set_position_mode: Callable[[object], None],
    command_positions: Callable[[object, Sequence[float]], bool | None],
    torque_enable: Callable[[object, str, str, bool], None],
) -> None:
    """Hold every leader arm at its measured pose while its gripper stays free."""
    for name in sorted(leaders):
        robot = leaders[name]
        positions = tuple(
            float(value) for value in read_positions(robot)
        )
        if not positions:
            raise RuntimeError(f"{name} returned no arm positions")
        if gravity_compensation:
            disable_gravity_compensation(robot)
        set_position_mode(robot)
        if command_positions(robot, positions) is False:
            raise RuntimeError(
                f"{name} refused current position goal; "
                f"positions={positions!r}"
            )
        torque_enable(robot, "group", "arm", True)
        torque_enable(robot, "single", "gripper", False)


def _joint_distance(
    leader: float,
    follower: float,
    *,
    continuous: bool,
) -> float:
    delta = float(leader) - float(follower)
    if continuous:
        delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
    return abs(delta)


def evaluate_joint_alignment(
    leader_positions: Mapping[str, Sequence[float]],
    follower_positions: Mapping[str, Sequence[float]],
    *,
    max_joint_error_rad: float,
    continuous_joint_indices: Sequence[int] = (3, 5),
) -> JointAlignmentReport:
    """Compare paired arm poses before enabling follower motion."""
    if max_joint_error_rad <= 0:
        raise ValueError("max_joint_error_rad must be positive")
    leader_suffixes = set(leader_positions)
    follower_suffixes = set(follower_positions)
    if leader_suffixes != follower_suffixes:
        unmatched = sorted(leader_suffixes ^ follower_suffixes)
        raise ValueError(
            f"unmatched robot suffixes: {', '.join(unmatched)}"
        )
    if not leader_suffixes:
        raise ValueError("no leader/follower pairs were provided")

    continuous = set(int(index) for index in continuous_joint_indices)
    pair_errors: dict[str, float] = {}
    for suffix in sorted(leader_suffixes):
        leader = tuple(leader_positions[suffix])
        follower = tuple(follower_positions[suffix])
        if len(leader) != len(follower):
            raise ValueError(
                f"{suffix} leader/follower joint counts do not match"
            )
        if not leader:
            raise ValueError(f"{suffix} joint positions are empty")
        pair_errors[suffix] = max(
            _joint_distance(
                leader_value,
                follower_value,
                continuous=index in continuous,
            )
            for index, (leader_value, follower_value) in enumerate(
                zip(leader, follower)
            )
        )

    maximum = max(pair_errors.values())
    return JointAlignmentReport(
        safe=maximum <= float(max_joint_error_rad),
        max_error_rad=maximum,
        pair_errors_rad=pair_errors,
    )


def wait_for_safe_current_pose_rearm(
    *,
    read_grippers: Callable[[], Mapping[str, float]],
    read_leader_positions: Callable[[], Mapping[str, Sequence[float]]],
    read_follower_positions: Callable[[], Mapping[str, Sequence[float]]],
    restore_teleop: Callable[[], None],
    stop_requested: Callable[[], bool],
    max_joint_error_rad: float,
    debounce_samples: int = 3,
    poll_interval: float = 0.01,
    sleep: Callable[[float], None],
    health_check: Callable[[], None] = lambda: None,
    logger: Callable[[str], None] = print,
) -> bool:
    """Wait for a safe gesture and restore following without a HOME move."""
    detector = DualGripperRearmDetector(
        debounce_samples=debounce_samples,
    )
    logger(
        "[current-pose rearm] open both leader grippers, then close both "
        "to enable follower following"
    )
    while not stop_requested():
        health_check()
        if detector.update(read_grippers()):
            report = evaluate_joint_alignment(
                read_leader_positions(),
                read_follower_positions(),
                max_joint_error_rad=max_joint_error_rad,
            )
            if not report.safe:
                logger(
                    "[current-pose rearm] warning; maximum arm joint "
                    f"error={report.max_error_rad:.4f} rad exceeds "
                    f"diagnostic threshold {max_joint_error_rad:.4f} rad; "
                    "continuing because the stop-pose hold owns the "
                    "transition"
                )
            health_check()
            restore_teleop()
            logger(
                "[current-pose rearm] accepted; maximum arm joint "
                f"error={report.max_error_rad:.4f} rad"
            )
            return True
        sleep(max(0.0, float(poll_interval)))
    return False
