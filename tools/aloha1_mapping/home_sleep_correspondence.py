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
FOLLOWER_ROOT_PREFIXES = ("/World/follower_left/", "/World/follower_right/")


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


def validate_digital_preflight(contract: dict[str, object]) -> dict[str, object]:
    """Validate the immutable digital-run contract without importing Isaac Sim."""

    boolean_gates = (
        "runtime_versions_match",
        "stage_hash_match",
        "manifest_hash_match",
        "root_prim_valid",
        "required_prims_valid",
        "dof_order_match",
        "finger_limit_hash_match",
        "home_finite_and_legal",
        "first_frame_arm_stable",
        "stationary_scope_declared",
        "source_hashes_immutable",
    )
    failed = [name for name in boolean_gates if contract.get(name) is not True]
    if contract.get("default_prim") != "/World":
        failed.append("default_prim")
    if contract.get("articulation_count") != 2:
        failed.append("articulation_count")
    if contract.get("final_default_asset_modified") is not False:
        failed.append("final_default_asset_modified")
    return {
        "status": "PASS" if not failed else "FAIL",
        "failed_gates": failed,
        "contract": dict(contract),
    }


def digital_runtime_signature(payload: dict[str, object]) -> str:
    """Hash normalized digital evidence while excluding process-local timing."""

    normalized = {
        key: value
        for key, value in payload.items()
        if key not in {"runtime_pid", "wall_time_s", "absolute_output_paths"}
    }
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def count_follower_articulation_roots(paths: Sequence[str]) -> list[str]:
    """Return robot-scoped roots, excluding environment schema roots."""

    return [
        str(path)
        for path in paths
        if any(str(path).startswith(prefix) for prefix in FOLLOWER_ROOT_PREFIXES)
    ]


def values_within_float32_limits(
    values: Sequence[float], lower: Sequence[float], upper: Sequence[float]
) -> bool:
    """Check limits with eight float32 ULP-equivalents of numeric slack."""

    if not (len(values) == len(lower) == len(upper)):
        raise ValueError("limit vectors must have equal lengths")
    relative_slack = 8.0 * 2.0**-23
    return all(
        float(low) - relative_slack * max(1.0, abs(float(low)))
        <= float(value)
        <= float(high) + relative_slack * max(1.0, abs(float(high)))
        for value, low, high in zip(values, lower, upper, strict=True)
    )


def compare_aligned_joint_rows(
    digital_rows: Sequence[dict[str, object]],
    real_rows: Sequence[dict[str, object]],
    *,
    joint_names: Sequence[str] = ARM_JOINT_ORDER,
) -> dict[str, object]:
    """Compare immutable traces by exact command index without rewriting either trace."""

    def by_index(rows: Sequence[dict[str, object]]) -> dict[int, list[float]]:
        result: dict[int, list[float]] = {}
        for row in rows:
            index = int(row["command_index"])
            q = [float(value) for value in row["q"]]  # type: ignore[index]
            if index in result:
                raise ValueError(f"duplicate command index: {index}")
            result[index] = q
        return result

    digital = by_index(digital_rows)
    real = by_index(real_rows)
    matched = sorted(set(digital) & set(real))
    if not matched:
        raise ValueError("no common command indices")
    joint_count = len(joint_names)
    if any(
        len(digital[index]) != joint_count or len(real[index]) != joint_count
        for index in matched
    ):
        raise ValueError("joint vector length does not match joint_names")
    per_joint = []
    for joint_index, joint_name in enumerate(joint_names):
        errors = [real[index][joint_index] - digital[index][joint_index] for index in matched]
        per_joint.append(
            {
                "joint_name": str(joint_name),
                "joint_index": joint_index,
                "signed_mean_error_rad": sum(errors) / len(errors),
                "rmse_rad": math.sqrt(sum(error * error for error in errors) / len(errors)),
                "maximum_abs_error_rad": max(abs(error) for error in errors),
            }
        )
    return {
        "matched_command_count": len(matched),
        "first_command_index": matched[0],
        "last_command_index": matched[-1],
        "per_joint": per_joint,
        "raw_inputs_modified": False,
    }
