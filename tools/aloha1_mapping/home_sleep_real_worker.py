"""Transport-independent fail-closed core for real ALOHA replay."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Protocol

from tools.aloha1_mapping.home_sleep_correspondence import ARM_JOINT_ORDER
from tools.aloha1_mapping.home_sleep_sync import deadline_ns


@dataclass(frozen=True)
class JointStateRecord:
    names: tuple[str, ...]
    positions: tuple[float, ...]
    velocities: tuple[float, ...] | None
    efforts: tuple[float, ...] | None
    source_stamp_ns: int | None
    receive_monotonic_ns: int
    receive_wall_time_ns: int


class Clock(Protocol):
    def monotonic_ns(self) -> int:
        raise NotImplementedError

    def set_sample_index(self, sample_index: int) -> None:
        raise NotImplementedError

    def wait_until(self, deadline: int) -> None:
        raise NotImplementedError


class JointStateSource(Protocol):
    def latest(self) -> JointStateRecord:
        raise NotImplementedError


class CommandSink(Protocol):
    def publish(self, sample_index: int, q_rad: tuple[float, ...]) -> bool:
        raise NotImplementedError


class StopController(Protocol):
    def hold(self, reason: str) -> bool:
        raise NotImplementedError


def direction_matches(
    *,
    previous_target: Sequence[float],
    target: Sequence[float],
    previous_readback: Sequence[float],
    readback: Sequence[float],
    minimum_motion_rad: float,
) -> bool:
    """Reject an observed joint delta opposite a meaningful target delta."""

    if not (
        len(previous_target)
        == len(target)
        == len(previous_readback)
        == len(readback)
    ):
        raise ValueError("direction vectors must have equal length")
    for target_before, target_now, actual_before, actual_now in zip(
        previous_target, target, previous_readback, readback, strict=True
    ):
        commanded = float(target_now) - float(target_before)
        observed = float(actual_now) - float(actual_before)
        if abs(commanded) < minimum_motion_rad or abs(observed) < minimum_motion_rad:
            continue
        if commanded * observed < 0.0:
            return False
    return True


class RealWorkerCore:
    """Validate and schedule real commands through injected safe interfaces."""

    def __init__(
        self,
        *,
        maximum_readback_age_ns: int,
        expected_joint_names: Sequence[str] = ARM_JOINT_ORDER,
    ) -> None:
        if maximum_readback_age_ns <= 0:
            raise ValueError("maximum_readback_age_ns must be positive")
        self.maximum_readback_age_ns = int(maximum_readback_age_ns)
        self.expected_joint_names = tuple(str(name) for name in expected_joint_names)

    def preflight(
        self,
        state: JointStateRecord,
        *,
        now_monotonic_ns: int,
        camera_ready: bool,
        stop_path_verified: bool,
        hardware_status: Mapping[str, object],
    ) -> dict[str, Any]:
        present_current = hardware_status.get("present_current", "NOT_AVAILABLE")
        if state.names != self.expected_joint_names:
            status = "BLOCKED_JOINT_ORDER"
        elif len(state.positions) != len(self.expected_joint_names):
            status = "BLOCKED_JOINT_COUNT"
        elif not all(math.isfinite(value) for value in state.positions):
            status = "BLOCKED_NONFINITE_READBACK"
        elif now_monotonic_ns - state.receive_monotonic_ns > self.maximum_readback_age_ns:
            status = "BLOCKED_STALE_READBACK"
        elif not camera_ready:
            status = "BLOCKED_CAM_HIGH"
        elif not stop_path_verified:
            status = "BLOCKED_STOP_PATH"
        elif hardware_status.get("hardware_error") not in (None, False, 0):
            status = "BLOCKED_HARDWARE_ERROR"
        else:
            status = "PASS"
        return {
            "status": status,
            "joint_names": list(state.names),
            "readback_age_ns": int(now_monotonic_ns - state.receive_monotonic_ns),
            "camera_ready": bool(camera_ready),
            "stop_path_verified": bool(stop_path_verified),
            "present_current": present_current,
        }

    def run_samples(
        self,
        samples: Sequence[Mapping[str, object]],
        *,
        start_monotonic_ns: int,
        sample_period_ns: int,
        clock: Clock,
        state_source: JointStateSource,
        command_sink: CommandSink,
        stop_controller: StopController,
    ) -> dict[str, Any]:
        records: list[dict[str, object]] = []
        previous_target: tuple[float, ...] | None = None
        previous_readback: tuple[float, ...] | None = None
        status = "PASS"
        for sample in samples:
            sample_index = int(sample["index"])
            clock.set_sample_index(sample_index)
            target_deadline = deadline_ns(
                start_monotonic_ns, sample_index, sample_period_ns
            )
            clock.wait_until(target_deadline)
            applied_at = int(clock.monotonic_ns())
            lateness = applied_at - target_deadline
            if lateness > sample_period_ns:
                status = "ABORTED_DEADLINE_MISS"
                stop_controller.hold(status)
                break
            state = state_source.latest()
            if state.names != self.expected_joint_names:
                status = "ABORTED_JOINT_ORDER"
                stop_controller.hold(status)
                break
            if applied_at - state.receive_monotonic_ns > self.maximum_readback_age_ns:
                status = "ABORTED_STALE_READBACK"
                stop_controller.hold(status)
                break
            target = tuple(float(value) for value in sample["q_rad"])  # type: ignore[arg-type]
            if len(target) != len(self.expected_joint_names) or not all(
                math.isfinite(value) for value in target
            ):
                status = "ABORTED_INVALID_TARGET"
                stop_controller.hold(status)
                break
            if (
                previous_target is not None
                and previous_readback is not None
                and not direction_matches(
                    previous_target=previous_target,
                    target=target,
                    previous_readback=previous_readback,
                    readback=state.positions,
                    minimum_motion_rad=0.001,
                )
            ):
                status = "ABORTED_OPPOSITE_DIRECTION"
                stop_controller.hold(status)
                break
            if not command_sink.publish(sample_index, target):
                status = "ABORTED_COMMAND_REJECTED"
                stop_controller.hold(status)
                break
            records.append(
                {
                    "sample_index": sample_index,
                    "cycle": int(sample["cycle"]),
                    "segment": str(sample["segment"]),
                    "target_q_rad": list(target),
                    "readback_q_rad": list(state.positions),
                    "scheduled_deadline_ns": target_deadline,
                    "applied_monotonic_ns": applied_at,
                    "lateness_ns": lateness,
                }
            )
            previous_target = target
            previous_readback = state.positions
        return {
            "schema_version": 1,
            "status": status,
            "records": records,
            "published_indices": [record["sample_index"] for record in records],
            "published_sample_count": len(records),
            "burst_catchup_used": False,
        }
