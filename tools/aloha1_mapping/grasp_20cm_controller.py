"""Pure state and acceptance gates for the ALOHA1 20 cm grasp demo."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import math
from typing import Any


class Phase(StrEnum):
    """Ordered runtime phases exposed by the Isaac diagnostic window."""

    IDLE = "IDLE"
    VALIDATE = "VALIDATE"
    SETUP_KINEMATIC = "SETUP_KINEMATIC"
    RELEASE_DYNAMIC = "RELEASE_DYNAMIC"
    SETTLE = "SETTLE"
    OPEN_PREGRASP = "OPEN_PREGRASP"
    VERTICAL_DESCENT = "VERTICAL_DESCENT"
    BILATERAL_CONTACT = "BILATERAL_CONTACT"
    CLOSE_PRELOAD = "CLOSE_PRELOAD"
    VERTICAL_LIFT = "VERTICAL_LIFT"
    HEIGHT_REACHED = "HEIGHT_REACHED"
    HOLD = "HOLD"
    PASS = "PASS"
    FAIL = "FAIL"
    ABORTED = "ABORTED"


ACTIVE_PHASES = (
    Phase.VALIDATE,
    Phase.SETUP_KINEMATIC,
    Phase.RELEASE_DYNAMIC,
    Phase.SETTLE,
    Phase.OPEN_PREGRASP,
    Phase.VERTICAL_DESCENT,
    Phase.BILATERAL_CONTACT,
    Phase.CLOSE_PRELOAD,
    Phase.VERTICAL_LIFT,
    Phase.HEIGHT_REACHED,
    Phase.HOLD,
)
TERMINAL_PHASES = (Phase.PASS, Phase.FAIL, Phase.ABORTED)


@dataclass(frozen=True)
class Grasp20cmThresholds:
    """Engineering diagnostic gates; these are not hardware calibration."""

    target_clearance_m: float = 0.200
    hold_duration_s: float = 2.0
    hold_drop_gate_m: float = 0.010
    settle_linear_speed_gate_m_s: float = 0.005
    settle_angular_speed_gate_rad_s: float = 0.05
    settle_consecutive_frames: int = 10
    false_lift_ee_displacement_m: float = 0.050
    false_lift_clearance_m: float = 0.010

    def __post_init__(self) -> None:
        finite_positive = (
            self.target_clearance_m,
            self.hold_duration_s,
            self.hold_drop_gate_m,
            self.settle_linear_speed_gate_m_s,
            self.settle_angular_speed_gate_rad_s,
            self.false_lift_ee_displacement_m,
            self.false_lift_clearance_m,
        )
        if not all(
            math.isfinite(value) and value > 0.0
            for value in finite_positive
        ):
            raise ValueError("threshold values must be finite and positive")
        if self.settle_consecutive_frames <= 0:
            raise ValueError("settle_consecutive_frames must be positive")


@dataclass(frozen=True)
class RunObservation:
    """One physics-frame observation consumed by the pure state machine."""

    frame: int
    time_s: float
    clearance_m: float
    bottle_dynamic: bool
    support_contact: bool
    bottle_linear_speed_m_s: float
    bottle_angular_speed_rad_s: float
    stage_contract_valid: bool
    setup_complete: bool
    open_target_reached: bool
    descent_complete: bool
    bilateral_contact: bool
    preload_complete: bool
    lift_waypoint_exhausted: bool
    hold_drop_m: float
    finite_state: bool
    persistent_penetration: bool
    numerical_ejection: bool
    forbidden_constraint: bool
    phase_timed_out: bool
    ee_vertical_displacement_m: float


@dataclass(frozen=True)
class TransitionRecord:
    """One state transition or same-state observation."""

    frame: int | None
    previous: Phase
    current: Phase
    reason: str


def measured_clearance_m(
    *,
    bottle_collision_min_world_z_m: float,
    table_top_world_z_m: float,
) -> float:
    """Return bottle collider clearance above the table in world metres."""
    values = (
        float(bottle_collision_min_world_z_m),
        float(table_top_world_z_m),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("clearance inputs must be finite")
    return values[0] - values[1]


def evaluate_terminal_run(
    metrics: dict[str, Any],
    thresholds: Grasp20cmThresholds,
) -> dict[str, str]:
    """Classify one completed run using a fail-closed precedence."""
    if bool(metrics.get("aborted", False)):
        failure_mode = "aborted"
    elif bool(metrics.get("forbidden_constraint", True)):
        failure_mode = "forbidden_constraint"
    elif not bool(metrics.get("finite_state", False)):
        failure_mode = "non_finite_state"
    elif bool(metrics.get("persistent_penetration", True)) or bool(
        metrics.get("numerical_ejection", True)
    ):
        failure_mode = "numerical_penetration_or_ejection"
    elif not bool(metrics.get("dynamic_during_formal_phases", False)):
        failure_mode = "bottle_not_dynamic_during_formal_phases"
    elif not bool(metrics.get("bilateral_contact_before_lift", False)):
        failure_mode = "bilateral_contact_not_established"
    elif (
        _finite_at_least(
            metrics.get("ee_vertical_displacement_m"),
            thresholds.false_lift_ee_displacement_m,
        )
        and _finite_below(
            metrics.get("maximum_clearance_m"),
            thresholds.false_lift_clearance_m,
        )
    ):
        failure_mode = "gripper_moved_without_bottle_lift"
    elif not bool(metrics.get("height_reached", False)):
        failure_mode = "height_target_not_reached"
    elif not bool(metrics.get("bilateral_contact_through_hold", False)):
        failure_mode = "bilateral_contact_lost"
    elif not _finite_at_least(
        metrics.get("hold_duration_s"),
        thresholds.hold_duration_s,
    ):
        failure_mode = "hold_interval_incomplete"
    elif not _finite_at_most(
        metrics.get("hold_drop_m"),
        thresholds.hold_drop_gate_m,
    ):
        failure_mode = "hold_drop_exceeded"
    else:
        failure_mode = "stable_20cm_hold"
    return {
        "status": "PASS" if failure_mode == "stable_20cm_hold" else "FAIL",
        "failure_mode": failure_mode,
        "task8": "NOT_RUN",
    }


def _finite_at_least(value: Any, gate: float) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric >= gate


def _finite_at_most(value: Any, gate: float) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric <= gate


def _finite_below(value: Any, gate: float) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric < gate


class Grasp20cmController:
    """Pure phase controller advanced once per physics observation."""

    def __init__(
        self,
        thresholds: Grasp20cmThresholds | None = None,
    ) -> None:
        self.thresholds = thresholds or Grasp20cmThresholds()
        self.phase = Phase.IDLE
        self.maximum_clearance_m = -math.inf
        self._settled_frames = 0
        self._hold_start_time_s: float | None = None
        self._hold_bilateral_continuous = True

    def start(self) -> TransitionRecord:
        if self.phase is not Phase.IDLE:
            raise RuntimeError(f"cannot start from {self.phase}")
        return self._transition(Phase.VALIDATE, "run_requested")

    def observe(self, observation: RunObservation) -> TransitionRecord:
        if self.phase in (Phase.IDLE, *TERMINAL_PHASES):
            raise RuntimeError(
                f"cannot observe physics frame while phase is {self.phase}"
            )
        self.maximum_clearance_m = max(
            self.maximum_clearance_m,
            observation.clearance_m,
        )
        immediate_failure = self._observation_failure(observation)
        if immediate_failure is not None:
            return self._transition(
                Phase.FAIL,
                immediate_failure,
                frame=observation.frame,
            )
        if self.phase is Phase.VALIDATE:
            return self._transition(
                Phase.SETUP_KINEMATIC,
                "stage_contract_verified",
                frame=observation.frame,
            )
        if self.phase is Phase.SETUP_KINEMATIC:
            if observation.setup_complete:
                return self._transition(
                    Phase.RELEASE_DYNAMIC,
                    "session_setup_complete",
                    frame=observation.frame,
                )
        elif self.phase is Phase.RELEASE_DYNAMIC:
            if observation.bottle_dynamic:
                return self._transition(
                    Phase.SETTLE,
                    "bottle_released_dynamic",
                    frame=observation.frame,
                )
        elif self.phase is Phase.SETTLE:
            settled = (
                observation.bottle_dynamic
                and observation.support_contact
                and observation.bottle_linear_speed_m_s
                <= self.thresholds.settle_linear_speed_gate_m_s
                and observation.bottle_angular_speed_rad_s
                <= self.thresholds.settle_angular_speed_gate_rad_s
            )
            self._settled_frames = (
                self._settled_frames + 1 if settled else 0
            )
            if (
                self._settled_frames
                >= self.thresholds.settle_consecutive_frames
            ):
                return self._transition(
                    Phase.OPEN_PREGRASP,
                    "dynamic_support_settled",
                    frame=observation.frame,
                )
        elif self.phase is Phase.OPEN_PREGRASP:
            if observation.open_target_reached:
                return self._transition(
                    Phase.VERTICAL_DESCENT,
                    "open_target_reached",
                    frame=observation.frame,
                )
        elif self.phase is Phase.VERTICAL_DESCENT:
            if observation.descent_complete:
                return self._transition(
                    Phase.BILATERAL_CONTACT,
                    "vertical_descent_complete",
                    frame=observation.frame,
                )
        elif self.phase is Phase.BILATERAL_CONTACT:
            if observation.bilateral_contact:
                return self._transition(
                    Phase.CLOSE_PRELOAD,
                    "bilateral_contact_confirmed",
                    frame=observation.frame,
                )
        elif self.phase is Phase.CLOSE_PRELOAD:
            if observation.preload_complete:
                return self._transition(
                    Phase.VERTICAL_LIFT,
                    "close_preload_complete",
                    frame=observation.frame,
                )
        elif self.phase is Phase.VERTICAL_LIFT:
            if observation.clearance_m >= self.thresholds.target_clearance_m:
                return self._transition(
                    Phase.HEIGHT_REACHED,
                    "measured_clearance_target_reached",
                    frame=observation.frame,
                )
            bottle_settled = (
                observation.bottle_linear_speed_m_s
                <= self.thresholds.settle_linear_speed_gate_m_s
                and observation.bottle_angular_speed_rad_s
                <= self.thresholds.settle_angular_speed_gate_rad_s
            )
            if observation.lift_waypoint_exhausted and bottle_settled:
                reason = (
                    "gripper_moved_without_bottle_lift"
                    if (
                        observation.ee_vertical_displacement_m
                        >= self.thresholds.false_lift_ee_displacement_m
                        and self.maximum_clearance_m
                        < self.thresholds.false_lift_clearance_m
                    )
                    else "height_target_not_reached"
                )
                return self._transition(
                    Phase.FAIL,
                    reason,
                    frame=observation.frame,
                )
        elif self.phase is Phase.HEIGHT_REACHED:
            self._hold_start_time_s = observation.time_s
            self._hold_bilateral_continuous = (
                observation.bilateral_contact
            )
            return self._transition(
                Phase.HOLD,
                "hold_started",
                frame=observation.frame,
            )
        elif self.phase is Phase.HOLD:
            self._hold_bilateral_continuous &= (
                observation.bilateral_contact
            )
            if observation.hold_drop_m > self.thresholds.hold_drop_gate_m:
                return self._transition(
                    Phase.FAIL,
                    "hold_drop_exceeded",
                    frame=observation.frame,
                )
            if (
                self._hold_start_time_s is not None
                and observation.time_s - self._hold_start_time_s
                >= self.thresholds.hold_duration_s
            ):
                if not self._hold_bilateral_continuous:
                    return self._transition(
                        Phase.FAIL,
                        "bilateral_contact_lost",
                        frame=observation.frame,
                    )
                return self._transition(
                    Phase.PASS,
                    "stable_20cm_hold",
                    frame=observation.frame,
                )
        return TransitionRecord(
            frame=observation.frame,
            previous=self.phase,
            current=self.phase,
            reason="phase_in_progress",
        )

    def request_abort(self) -> TransitionRecord:
        if self.phase not in ACTIVE_PHASES:
            raise RuntimeError(f"cannot abort from {self.phase}")
        return self._transition(Phase.ABORTED, "user_abort")

    def reset(self) -> TransitionRecord:
        if self.phase in ACTIVE_PHASES:
            raise RuntimeError(f"cannot reset active phase {self.phase}")
        previous = self.phase
        self.phase = Phase.IDLE
        self.maximum_clearance_m = -math.inf
        self._settled_frames = 0
        self._hold_start_time_s = None
        self._hold_bilateral_continuous = True
        return TransitionRecord(
            frame=None,
            previous=previous,
            current=Phase.IDLE,
            reason="reset",
        )

    def restore_for_test(self, phase: Phase) -> None:
        """Place the pure controller at a phase without simulating Isaac."""
        self.phase = phase
        self.maximum_clearance_m = -math.inf
        self._settled_frames = 0
        self._hold_start_time_s = None
        self._hold_bilateral_continuous = True

    def _observation_failure(
        self,
        observation: RunObservation,
    ) -> str | None:
        numeric = (
            observation.time_s,
            observation.clearance_m,
            observation.bottle_linear_speed_m_s,
            observation.bottle_angular_speed_rad_s,
            observation.hold_drop_m,
            observation.ee_vertical_displacement_m,
        )
        if not observation.finite_state or not all(
            math.isfinite(value) for value in numeric
        ):
            return "non_finite_state"
        if (
            observation.persistent_penetration
            or observation.numerical_ejection
        ):
            return "numerical_penetration_or_ejection"
        if observation.forbidden_constraint:
            return "forbidden_constraint"
        if observation.phase_timed_out:
            return f"{self.phase.value.lower()}_timeout"
        if self.phase is Phase.VALIDATE and not observation.stage_contract_valid:
            return "stage_contract_invalid"
        return None

    def _transition(
        self,
        current: Phase,
        reason: str,
        *,
        frame: int | None = None,
    ) -> TransitionRecord:
        previous = self.phase
        self.phase = current
        return TransitionRecord(
            frame=frame,
            previous=previous,
            current=current,
            reason=reason,
        )


_SIGNATURE_EXCLUDED_KEYS = {
    "artifact_absolute_path",
    "runtime_seconds",
    "video_sha256",
    "raw_video_sha256",
    "annotated_video_sha256",
}


def canonical_run_signature(
    observations: list[RunObservation],
    terminal_metrics: dict[str, Any],
) -> str:
    """Hash deterministic physics inputs while excluding artifact metadata."""
    payload = {
        "observations": [asdict(observation) for observation in observations],
        "terminal_metrics": {
            key: value
            for key, value in terminal_metrics.items()
            if key not in _SIGNATURE_EXCLUDED_KEYS
        },
    }
    encoded = json.dumps(
        _canonicalize(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _canonicalize(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, list | tuple):
        return [_canonicalize(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NON_FINITE"
        return round(value, 9)
    if isinstance(value, bool) or value is None or isinstance(value, str | int):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "NON_FINITE"
    return round(numeric, 9)
