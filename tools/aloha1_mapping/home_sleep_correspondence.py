"""Pure Home/Sleep command and comparison primitives for Stationary ALOHA 1."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
import hashlib
import json
import math

ARM_JOINT_ORDER = (
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
)
HOME_ARM = (0.0, -0.96, 1.16, 0.0, -0.3, 0.0)
SLEEP_ARM = (0.0, -2.05, 1.7, 0.0, -2.0, 0.0)


@dataclass(frozen=True)
class CommandSample:
    """One immutable arm-only command sample."""

    index: int
    time_ns: int
    cycle: int
    segment: str
    segment_sample: int
    q_rad: tuple[float, ...]


def _arm_vector(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != 6 or not all(math.isfinite(value) for value in result):
        raise ValueError("expected six finite arm joints")
    return result


def _sample_count(*, frequency_hz: int, duration_seconds: int) -> int:
    if frequency_hz <= 0 or duration_seconds <= 0:
        raise ValueError("frequency and duration must be positive")
    return frequency_hz * duration_seconds


def _linear_segment(
    start: tuple[float, ...], end: tuple[float, ...], count: int
) -> list[tuple[float, ...]]:
    if count < 2:
        raise ValueError("movement segment requires at least two samples")
    denominator = count - 1
    return [
        tuple(
            start[joint] + (end[joint] - start[joint]) * index / denominator
            for joint in range(len(start))
        )
        for index in range(count)
    ]


def build_home_sleep_samples(
    *,
    home: Sequence[float] = HOME_ARM,
    sleep: Sequence[float] = SLEEP_ARM,
    command_hz: int = 50,
    move_seconds: int = 5,
    hold_seconds: int = 1,
    cycles: int = 3,
) -> tuple[CommandSample, ...]:
    """Build the sole three-cycle command authority using integer time."""

    home_q = _arm_vector(home)
    sleep_q = _arm_vector(sleep)
    if cycles <= 0:
        raise ValueError("cycles must be positive")
    if 1_000_000_000 % command_hz != 0:
        raise ValueError("command_hz must divide one second into integer nanoseconds")
    move_count = _sample_count(
        frequency_hz=command_hz, duration_seconds=move_seconds
    )
    hold_count = _sample_count(
        frequency_hz=command_hz, duration_seconds=hold_seconds
    )
    dt_ns = 1_000_000_000 // command_hz
    records: list[CommandSample] = []

    def append_segment(
        *, cycle: int, segment: str, positions: Sequence[tuple[float, ...]]
    ) -> None:
        for segment_sample, q_rad in enumerate(positions):
            index = len(records)
            records.append(
                CommandSample(
                    index=index,
                    time_ns=index * dt_ns,
                    cycle=cycle,
                    segment=segment,
                    segment_sample=segment_sample,
                    q_rad=q_rad,
                )
            )

    append_segment(
        cycle=0,
        segment="initial_home_hold",
        positions=[home_q] * hold_count,
    )
    for cycle in range(1, cycles + 1):
        prefix = f"cycle_{cycle:02d}"
        append_segment(
            cycle=cycle,
            segment=f"{prefix}_home_to_sleep",
            positions=_linear_segment(home_q, sleep_q, move_count),
        )
        append_segment(
            cycle=cycle,
            segment=f"{prefix}_sleep_hold",
            positions=[sleep_q] * hold_count,
        )
        append_segment(
            cycle=cycle,
            segment=f"{prefix}_sleep_to_home",
            positions=_linear_segment(sleep_q, home_q, move_count),
        )
        append_segment(
            cycle=cycle,
            segment=f"{prefix}_home_hold",
            positions=[home_q] * hold_count,
        )
    return tuple(records)


def command_index_for_physics_frame(
    physics_frame: int,
    *,
    physics_hz: int,
    command_hz: int,
    sample_count: int,
) -> int:
    """Map a physics frame to a held command with rational integer arithmetic."""

    if physics_frame < 0:
        raise ValueError("physics_frame must be non-negative")
    if physics_hz <= 0 or command_hz <= 0 or sample_count <= 0:
        raise ValueError("frequencies and sample_count must be positive")
    return min((physics_frame * command_hz) // physics_hz, sample_count - 1)


def command_signature(samples: Sequence[CommandSample]) -> str:
    """Return a canonical SHA-256 for the exact stored command sequence."""

    payload = [asdict(sample) for sample in samples]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
