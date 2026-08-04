"""Event-driven robot health monitoring based only on ROS joint states."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Callable, Iterable, Mapping


@dataclass(frozen=True)
class RobotHealthFault:
    robot_name: str
    phase: str
    reason: str
    message_age: float


@dataclass(frozen=True)
class RobotHealthSnapshot:
    robot_name: str
    sequence: int
    consecutive_valid: int
    message_age: float
    valid: bool
    reason: str | None


@dataclass
class _RobotState:
    expected_joints: frozenset[str]
    sequence: int = 0
    consecutive_valid: int = 0
    last_received_monotonic: float | None = None
    valid: bool = False
    reason: str | None = "no_joint_state"


class RobotHealthUnavailable(RuntimeError):
    """Raised when the requested robots cannot pass a health gate."""


class HealthScope:
    """A temporary watchdog scope for one motion or recovery phase."""

    def __init__(
        self,
        monitor: "RobotHealthMonitor",
        robot_names: frozenset[str],
        phase: str,
        max_age: float,
        latch_global: bool,
    ):
        self._monitor = monitor
        self.robot_names = robot_names
        self.phase = phase
        self.max_age = max_age
        self.latch_global = latch_global
        self.fault_event = threading.Event()
        self.fault: RobotHealthFault | None = None

    def __enter__(self) -> "HealthScope":
        self._monitor._activate_scope(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._monitor._deactivate_scope(self)

    def raise_if_faulted(self) -> None:
        if self.fault is not None:
            raise RobotHealthUnavailable(
                f"{self.fault.robot_name} failed health check during "
                f"{self.fault.phase}: {self.fault.reason} "
                f"(message age={self.fault.message_age:.3f}s)"
            )

    def _set_fault(self, fault: RobotHealthFault) -> None:
        if self.fault is None:
            self.fault = fault
            self.fault_event.set()


class RobotHealthMonitor:
    """Track joint-state validity and watch only the currently armed robots."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        watchdog_rate_hz: float = 10.0,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        if watchdog_rate_hz <= 0:
            raise ValueError("watchdog_rate_hz must be positive")
        self._clock = clock
        self._watchdog_period = 1.0 / watchdog_rate_hz
        self._sleeper = sleeper
        self._states: dict[str, _RobotState] = {}
        self._lock = threading.RLock()
        self._active_scopes: set[HealthScope] = set()
        self._stop_event = threading.Event()
        self._watchdog_thread: threading.Thread | None = None
        self.fault_event = threading.Event()
        self.first_fault: RobotHealthFault | None = None

    def register_robot(
        self,
        robot_name: str,
        expected_joints: Iterable[str],
    ) -> None:
        expected = frozenset(expected_joints)
        if not expected:
            raise ValueError("expected_joints must not be empty")
        with self._lock:
            if robot_name in self._states:
                raise ValueError(f"robot already registered: {robot_name}")
            self._states[robot_name] = _RobotState(expected_joints=expected)

    def unregister_robot(self, robot_name: str) -> None:
        """Roll back a registration that has no usable subscription."""

        with self._lock:
            if self._scope_for_robot_locked(robot_name) is not None:
                raise RuntimeError(
                    f"cannot unregister active robot: {robot_name}"
                )
            if robot_name not in self._states:
                raise KeyError(robot_name)
            del self._states[robot_name]

    def accept(self, robot_name: str, message: object) -> None:
        now = self._clock()
        with self._lock:
            state = self._state(robot_name)
            state.sequence += 1
            state.last_received_monotonic = now
            reason = self._validate_message(state.expected_joints, message)
            state.valid = reason is None
            state.reason = reason
            if reason is None:
                state.consecutive_valid += 1
                return

            state.consecutive_valid = 0
            self._report_fault_locked(
                RobotHealthFault(
                    robot_name=robot_name,
                    phase=self._phase_for(robot_name, "joint_state_callback"),
                    reason=reason,
                    message_age=0.0,
                ),
                malformed_without_scope=True,
            )

    def snapshot(self, robot_name: str) -> RobotHealthSnapshot:
        with self._lock:
            state = self._state(robot_name)
            age = self._message_age(state, self._clock())
            return RobotHealthSnapshot(
                robot_name=robot_name,
                sequence=state.sequence,
                consecutive_valid=state.consecutive_valid,
                message_age=age,
                valid=state.valid,
                reason=state.reason,
            )

    def require_fresh(
        self,
        robot_names: Iterable[str],
        *,
        max_age: float,
        phase: str,
    ) -> Mapping[str, RobotHealthSnapshot]:
        """Return current snapshots or reject stale/invalid robot state."""
        names = frozenset(robot_names)
        if not names:
            raise ValueError("robot_names must not be empty")
        if max_age <= 0:
            raise ValueError("max_age must be positive")

        with self._lock:
            fault = self.first_fault
        if fault is not None:
            raise RobotHealthUnavailable(
                f"{fault.robot_name} failed health check during {phase}: "
                f"{fault.reason} "
                f"(message age={fault.message_age:.3f}s)"
            )

        snapshots = {
            robot_name: self.snapshot(robot_name)
            for robot_name in names
        }
        for robot_name in sorted(snapshots):
            snapshot = snapshots[robot_name]
            if snapshot.valid and snapshot.message_age <= max_age:
                continue
            reason = (
                snapshot.reason
                if not snapshot.valid
                else "joint_state_stale"
            )
            raise RobotHealthUnavailable(
                f"{robot_name} failed health check during {phase}: "
                f"{reason} "
                f"(message age={snapshot.message_age:.3f}s)"
            )
        return snapshots

    def wait_for_fresh(
        self,
        robot_names: Iterable[str],
        *,
        consecutive: int,
        max_age: float,
        timeout: float,
        stop_requested: Callable[[], bool],
    ) -> Mapping[str, RobotHealthSnapshot]:
        names = frozenset(robot_names)
        if not names:
            raise ValueError("robot_names must not be empty")
        if consecutive <= 0 or max_age <= 0 or timeout <= 0:
            raise ValueError("health gate limits must be positive")
        with self._lock:
            for name in names:
                self._state(name)
            starting_sequences = {
                name: self._state(name).sequence for name in names
            }

        deadline = self._clock() + timeout
        while True:
            if stop_requested():
                raise RobotHealthUnavailable("stop requested during health gate")
            snapshots = {name: self.snapshot(name) for name in names}
            if all(
                snapshot.valid
                and snapshot.consecutive_valid >= consecutive
                and (
                    snapshot.sequence - starting_sequences[snapshot.robot_name]
                    >= consecutive
                )
                and snapshot.message_age <= max_age
                for snapshot in snapshots.values()
            ):
                return snapshots
            if self._clock() >= deadline:
                details = ", ".join(
                    f"{name}={snapshot.reason or 'stale'}"
                    f"/{snapshot.message_age:.3f}s"
                    f"/{snapshot.consecutive_valid}"
                    for name, snapshot in sorted(snapshots.items())
                )
                raise RobotHealthUnavailable(
                    f"joint-state health gate timed out: {details}"
                )
            self._sleeper(min(self._watchdog_period, 0.05))

    def arm_scope(
        self,
        robot_names: Iterable[str],
        *,
        phase: str,
        max_age: float,
        latch_global: bool,
    ) -> HealthScope:
        names = frozenset(robot_names)
        if not names:
            raise ValueError("robot_names must not be empty")
        if max_age <= 0:
            raise ValueError("max_age must be positive")
        with self._lock:
            for name in names:
                self._state(name)
        return HealthScope(self, names, phase, max_age, latch_global)

    def check_once(self) -> None:
        with self._lock:
            now = self._clock()
            for scope in tuple(self._active_scopes):
                if scope.fault_event.is_set():
                    continue
                for robot_name in sorted(scope.robot_names):
                    age = self._message_age(self._state(robot_name), now)
                    if age > scope.max_age:
                        self._report_fault_locked(
                            RobotHealthFault(
                                robot_name=robot_name,
                                phase=scope.phase,
                                reason="joint_state_stale",
                                message_age=age,
                            )
                        )
                        break

    def latch_fault(
        self,
        robot_name: str,
        phase: str,
        reason: str,
        message_age: float | None = None,
    ) -> RobotHealthFault:
        with self._lock:
            state = self._state(robot_name)
            fault = RobotHealthFault(
                robot_name=robot_name,
                phase=phase,
                reason=reason,
                message_age=(
                    self._message_age(state, self._clock())
                    if message_age is None
                    else message_age
                ),
            )
            self._latch_global_locked(fault)
            scope = self._scope_for_robot_locked(robot_name)
            if scope is not None:
                scope._set_fault(fault)
            return fault

    def start(self) -> None:
        with self._lock:
            if self._watchdog_thread is not None:
                raise RuntimeError("health watchdog already started")
            self._stop_event.clear()
            thread = threading.Thread(
                target=self._watchdog_loop,
                name="aloha-joint-state-watchdog",
                daemon=True,
            )
            self._watchdog_thread = thread
            thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        with self._lock:
            thread = self._watchdog_thread
        if thread is None:
            return
        self._stop_event.set()
        thread.join(timeout)
        if thread.is_alive():
            raise RuntimeError("health watchdog did not stop")
        with self._lock:
            self._watchdog_thread = None

    def _activate_scope(self, scope: HealthScope) -> None:
        with self._lock:
            overlapping = sorted(
                robot_name
                for active in self._active_scopes
                for robot_name in scope.robot_names & active.robot_names
            )
            if overlapping:
                raise RuntimeError(
                    "overlapping health scopes are not supported: "
                    + ", ".join(overlapping)
                )
            self._active_scopes.add(scope)

    def _deactivate_scope(self, scope: HealthScope) -> None:
        with self._lock:
            self._active_scopes.discard(scope)

    def _scope_for_robot_locked(
        self,
        robot_name: str,
    ) -> HealthScope | None:
        for scope in self._active_scopes:
            if robot_name in scope.robot_names:
                return scope
        return None

    def _watchdog_loop(self) -> None:
        while not self._stop_event.wait(self._watchdog_period):
            self.check_once()

    def _state(self, robot_name: str) -> _RobotState:
        try:
            return self._states[robot_name]
        except KeyError as exc:
            raise KeyError(f"robot is not registered: {robot_name}") from exc

    @staticmethod
    def _validate_message(
        expected_joints: frozenset[str],
        message: object,
    ) -> str | None:
        names = list(getattr(message, "name", ()))
        positions = list(getattr(message, "position", ()))
        if len(names) != len(positions):
            return "joint_name_position_length_mismatch"
        if len(names) != len(set(names)):
            return "duplicate_joint_names"
        positions_by_name = dict(zip(names, positions))
        if not expected_joints.issubset(positions_by_name):
            return "missing_expected_joints"
        expected_positions = [
            float(positions_by_name[name]) for name in expected_joints
        ]
        if not all(math.isfinite(position) for position in expected_positions):
            return "non_finite_position"
        if expected_positions and all(
            math.isclose(position, -math.pi, abs_tol=1e-3)
            for position in expected_positions
        ):
            return "invalid_all_minus_pi"
        return None

    @staticmethod
    def _message_age(state: _RobotState, now: float) -> float:
        if state.last_received_monotonic is None:
            return math.inf
        return max(0.0, now - state.last_received_monotonic)

    def _phase_for(self, robot_name: str, fallback: str) -> str:
        scope = self._scope_for_robot_locked(robot_name)
        if scope is not None:
            return scope.phase
        return fallback

    def _report_fault_locked(
        self,
        fault: RobotHealthFault,
        *,
        malformed_without_scope: bool = False,
    ) -> None:
        scope = self._scope_for_robot_locked(fault.robot_name)
        in_scope = scope is not None
        if in_scope:
            scope._set_fault(fault)
        if (in_scope and scope.latch_global) or (
            malformed_without_scope and not in_scope
        ):
            self._latch_global_locked(fault)

    def _latch_global_locked(self, fault: RobotHealthFault) -> None:
        if self.first_fault is None:
            self.first_fault = fault
        self.fault_event.set()


def attach_joint_state_subscriptions(
    node,
    monitor: RobotHealthMonitor,
    expected_joints_by_robot: Mapping[str, set[str]],
) -> list[object]:
    """Subscribe to existing joint states without touching the servo bus."""

    from sensor_msgs.msg import JointState

    subscriptions = []
    registered_names = []
    try:
        for robot_name, expected in expected_joints_by_robot.items():
            monitor.register_robot(robot_name, expected)
            registered_names.append(robot_name)
            subscriptions.append(
                node.create_subscription(
                    JointState,
                    f"/{robot_name}/joint_states",
                    lambda message, name=robot_name: monitor.accept(
                        name,
                        message,
                    ),
                    10,
                )
            )
    except Exception:
        for subscription in reversed(subscriptions):
            try:
                node.destroy_subscription(subscription)
            except Exception:
                pass
        for robot_name in reversed(registered_names):
            monitor.unregister_robot(robot_name)
        raise
    return subscriptions
