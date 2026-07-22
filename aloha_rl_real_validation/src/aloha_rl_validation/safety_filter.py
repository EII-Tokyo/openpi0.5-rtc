from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import time
from typing import Iterable

import numpy as np

from .schema import CANONICAL_JOINT_NAMES


@dataclass
class SafetyConfig:
    allow_real_actuation: bool = False
    joint_lower: np.ndarray | None = None
    joint_upper: np.ndarray | None = None
    max_step_delta: np.ndarray | None = None
    max_velocity: np.ndarray | None = None
    max_acceleration: np.ndarray | None = None
    max_image_age_s: float | None = None
    max_joint_state_age_s: float | None = None
    max_policy_latency_s: float | None = None
    consecutive_exception_limit: int = 1
    joint_names: tuple[str, ...] = CANONICAL_JOINT_NAMES


@dataclass
class SafetyResult:
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    filtered_action: np.ndarray | None = None


class SafetyFilter:
    """Reject unsafe actions. It never silently clamps."""

    def __init__(self, config: SafetyConfig):
        self.config = config
        self._last_action: np.ndarray | None = None
        self._last_time: float | None = None
        self._last_velocity: np.ndarray | None = None
        env_flag = os.environ.get("ALLOW_REAL_ACTUATION", "false").lower()
        self._env_allows_motion = env_flag in {"1", "true", "yes"}

    def validate_joint_names(self, names: Iterable[str]) -> list[str]:
        names = tuple(names)
        if names != self.config.joint_names:
            return [f"joint name/order mismatch: got={names}, expected={self.config.joint_names}"]
        return []

    def check(
        self,
        action: np.ndarray,
        *,
        current_qpos: np.ndarray | None = None,
        joint_names: Iterable[str] | None = None,
        image_age_s: float | None = None,
        joint_state_age_s: float | None = None,
        policy_latency_s: float | None = None,
        now_s: float | None = None,
    ) -> SafetyResult:
        now_s = time.monotonic() if now_s is None else now_s
        action = np.asarray(action, dtype=np.float64)
        reasons: list[str] = []

        if not self.config.allow_real_actuation or not self._env_allows_motion:
            reasons.append("dry-run: real actuation disabled")

        if action.shape != (len(self.config.joint_names),):
            reasons.append(f"action shape {action.shape} != {(len(self.config.joint_names),)}")
            return SafetyResult(False, reasons, None)

        if not np.all(np.isfinite(action)):
            reasons.append("action contains NaN or Inf")

        if joint_names is not None:
            reasons.extend(self.validate_joint_names(joint_names))

        if self.config.joint_lower is not None:
            low = np.asarray(self.config.joint_lower, dtype=np.float64)
            if np.any(action < low):
                idx = np.where(action < low)[0].tolist()
                reasons.append(f"joint lower limit violation at indices {idx}")

        if self.config.joint_upper is not None:
            high = np.asarray(self.config.joint_upper, dtype=np.float64)
            if np.any(action > high):
                idx = np.where(action > high)[0].tolist()
                reasons.append(f"joint upper limit violation at indices {idx}")

        reference = current_qpos if self._last_action is None else self._last_action
        if reference is not None and self.config.max_step_delta is not None:
            max_delta = np.asarray(self.config.max_step_delta, dtype=np.float64)
            delta = np.abs(action - np.asarray(reference, dtype=np.float64))
            if np.any(delta > max_delta):
                idx = np.where(delta > max_delta)[0].tolist()
                reasons.append(f"single-step delta violation at indices {idx}")

        dt = None if self._last_time is None else max(now_s - self._last_time, 1e-9)
        if dt is not None and self._last_action is not None:
            vel = (action - self._last_action) / dt
            if self.config.max_velocity is not None:
                max_vel = np.asarray(self.config.max_velocity, dtype=np.float64)
                if np.any(np.abs(vel) > max_vel):
                    idx = np.where(np.abs(vel) > max_vel)[0].tolist()
                    reasons.append(f"velocity violation at indices {idx}")
            if self.config.max_acceleration is not None and self._last_velocity is not None:
                acc = (vel - self._last_velocity) / dt
                max_acc = np.asarray(self.config.max_acceleration, dtype=np.float64)
                if np.any(np.abs(acc) > max_acc):
                    idx = np.where(np.abs(acc) > max_acc)[0].tolist()
                    reasons.append(f"acceleration violation at indices {idx}")
            self._last_velocity = vel

        if image_age_s is not None and self.config.max_image_age_s is not None:
            if image_age_s > self.config.max_image_age_s:
                reasons.append(f"stale image: {image_age_s:.3f}s")
        if joint_state_age_s is not None and self.config.max_joint_state_age_s is not None:
            if joint_state_age_s > self.config.max_joint_state_age_s:
                reasons.append(f"stale joint state: {joint_state_age_s:.3f}s")
        if policy_latency_s is not None and self.config.max_policy_latency_s is not None:
            if policy_latency_s > self.config.max_policy_latency_s:
                reasons.append(f"policy timeout: {policy_latency_s:.3f}s")

        accepted = not reasons
        if accepted:
            self._last_action = action.copy()
            self._last_time = now_s
            return SafetyResult(True, [], action.copy())
        return SafetyResult(False, reasons, None)


def finite_or_raise(array: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")


def assert_no_publish_allowed() -> None:
    if os.environ.get("ALLOW_REAL_ACTUATION", "false").lower() in {"1", "true", "yes"}:
        return
    raise PermissionError("ALLOW_REAL_ACTUATION is false; publishing commands is forbidden")

