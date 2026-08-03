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
SLEEP_ARM = (0.0, -1.8, 1.55, 0.0, -1.57, 0.0)
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


def _linear_segment(start: tuple[float, ...], end: tuple[float, ...], count: int) -> list[tuple[float, ...]]:
    if count < 2:
        raise ValueError("movement segment requires at least two samples")
    denominator = count - 1
    return [
        tuple(start[joint] + (end[joint] - start[joint]) * index / denominator for joint in range(len(start)))
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
    move_count = _sample_count(frequency_hz=command_hz, duration_seconds=move_seconds)
    hold_count = _sample_count(frequency_hz=command_hz, duration_seconds=hold_seconds)
    dt_ns = 1_000_000_000 // command_hz
    records: list[CommandSample] = []

    def append_segment(*, cycle: int, segment: str, positions: Sequence[tuple[float, ...]]) -> None:
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


def build_sleep_home_samples(
    *,
    sleep: Sequence[float],
    home: Sequence[float] = HOME_ARM,
    command_hz: int = 50,
    move_seconds: int = 5,
    hold_seconds: int = 1,
    cycles: int = 3,
) -> tuple[CommandSample, ...]:
    """Build three Sleep-Home-Sleep cycles from an explicit runtime reference."""

    sleep_q = _arm_vector(sleep)
    home_q = _arm_vector(home)
    if cycles <= 0:
        raise ValueError("cycles must be positive")
    if 1_000_000_000 % command_hz != 0:
        raise ValueError("command_hz must divide one second into integer nanoseconds")
    move_count = _sample_count(frequency_hz=command_hz, duration_seconds=move_seconds)
    hold_count = _sample_count(frequency_hz=command_hz, duration_seconds=hold_seconds)
    dt_ns = 1_000_000_000 // command_hz
    records: list[CommandSample] = []

    def append_segment(*, cycle: int, segment: str, positions: Sequence[tuple[float, ...]]) -> None:
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

    append_segment(cycle=0, segment="initial_sleep_hold", positions=[sleep_q] * hold_count)
    for cycle in range(1, cycles + 1):
        prefix = f"cycle_{cycle:02d}"
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
    return tuple(records)


def expand_joint_limits_to_reference(
    reference: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    joint_names: Sequence[str] = ARM_JOINT_ORDER,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[dict[str, object], ...]]:
    """Return the smallest diagnostic-only limit expansion containing a reference."""

    reference_q = _arm_vector(reference)
    lower_q = _arm_vector(lower)
    upper_q = _arm_vector(upper)
    if len(joint_names) != len(reference_q):
        raise ValueError("joint_names must match reference length")
    if any(low > high for low, high in zip(lower_q, upper_q, strict=True)):
        raise ValueError("lower limits must not exceed upper limits")

    expanded_lower = list(lower_q)
    expanded_upper = list(upper_q)
    changes: list[dict[str, object]] = []
    for index, (joint_name, value, low, high) in enumerate(
        zip(joint_names, reference_q, lower_q, upper_q, strict=True)
    ):
        if value < low:
            expanded_lower[index] = value
            changes.append(
                {
                    "joint_name": str(joint_name),
                    "joint_index": index,
                    "bound": "lower",
                    "source_value_rad": low,
                    "diagnostic_value_rad": value,
                    "delta_rad": value - low,
                    "classification": "DIAGNOSTIC_ONLY_RUNTIME_ALIGNMENT",
                }
            )
        elif value > high:
            expanded_upper[index] = value
            changes.append(
                {
                    "joint_name": str(joint_name),
                    "joint_index": index,
                    "bound": "upper",
                    "source_value_rad": high,
                    "diagnostic_value_rad": value,
                    "delta_rad": value - high,
                    "classification": "DIAGNOSTIC_ONLY_RUNTIME_ALIGNMENT",
                }
            )
    return tuple(expanded_lower), tuple(expanded_upper), tuple(changes)


def manifest_initial_terminal_arm(
    manifest: dict[str, object],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Read generalized endpoint fields while preserving legacy Home manifests."""

    if "home_rad" not in manifest:
        raise ValueError("manifest is missing home_rad")
    home = _arm_vector(manifest["home_rad"])  # type: ignore[arg-type]
    initial = _arm_vector(manifest.get("initial_arm_rad", home))  # type: ignore[arg-type]
    terminal = _arm_vector(manifest.get("terminal_arm_rad", home))  # type: ignore[arg-type]
    return initial, terminal


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

    return [str(path) for path in paths if any(str(path).startswith(prefix) for prefix in FOLLOWER_ROOT_PREFIXES)]


def values_within_float32_limits(values: Sequence[float], lower: Sequence[float], upper: Sequence[float]) -> bool:
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


def evaluate_interbotix_group_limit_gate(
    samples: Sequence[CommandSample],
    *,
    lower_rad: Sequence[float],
    upper_rad: Sequence[float],
    moving_time_s: float,
    velocity_limits_rad_s: Sequence[float],
    joint_names: Sequence[str] = ARM_JOINT_ORDER,
) -> dict[str, object]:
    """Reproduce the official Interbotix whole-group Python limit gate.

    ``InterbotixArmXSInterface._check_joint_limits`` truncates each requested
    position toward zero to one milliradian, evaluates every position and
    velocity limit, and publishes the group only when every joint passes.
    This helper intentionally models that command-layer behavior; it does not
    clamp individual joints as a physics engine would.
    """

    joint_count = len(joint_names)
    vectors = (lower_rad, upper_rad, velocity_limits_rad_s)
    if any(len(vector) != joint_count for vector in vectors):
        raise ValueError("limit vectors must match joint_names")
    if moving_time_s <= 0.0 or not math.isfinite(moving_time_s):
        raise ValueError("moving_time_s must be finite and positive")
    if not samples:
        raise ValueError("at least one command sample is required")
    if any(len(sample.q_rad) != joint_count for sample in samples):
        raise ValueError("command sample length must match joint_names")

    joint_commands = list(samples[0].q_rad)
    accepted: list[CommandSample] = []
    rejected: list[dict[str, object]] = []
    for sample in samples:
        truncated = [int(value * 1000) / 1000.0 for value in sample.q_rad]
        speed = [abs(goal - current) / moving_time_s for goal, current in zip(truncated, joint_commands, strict=True)]
        position_failures = [
            str(joint_names[index])
            for index, value in enumerate(truncated)
            if not (float(lower_rad[index]) <= value <= float(upper_rad[index]))
        ]
        velocity_failures = [
            str(joint_names[index]) for index, value in enumerate(speed) if value > float(velocity_limits_rad_s[index])
        ]
        failed_names = list(dict.fromkeys(position_failures + velocity_failures))
        if failed_names:
            rejected.append(
                {
                    "sample_index": sample.index,
                    "segment_sample": sample.segment_sample,
                    "q_rad": list(sample.q_rad),
                    "truncated_q_rad": truncated,
                    "position_failure_joint_names": position_failures,
                    "velocity_failure_joint_names": velocity_failures,
                    "failed_joint_names": failed_names,
                }
            )
            continue
        accepted.append(sample)
        # The official publisher stores the original requested vector, not
        # the milliradian-truncated vector used solely by the limit check.
        joint_commands = list(sample.q_rad)

    first_rejected = rejected[0] if rejected else None
    last_published = list(accepted[-1].q_rad) if accepted else None
    return {
        "command_semantics": "REJECT_WHOLE_GROUP_SAMPLE",
        "sample_count": len(samples),
        "accepted_sample_count": len(accepted),
        "rejected_sample_count": len(rejected),
        "first_rejected_sample_index": (first_rejected["sample_index"] if first_rejected else None),
        "first_rejected_segment_sample": (first_rejected["segment_sample"] if first_rejected else None),
        "first_rejected_joint_names": (first_rejected["failed_joint_names"] if first_rejected else []),
        "first_rejected_q_rad": (first_rejected["q_rad"] if first_rejected else None),
        "first_rejected_truncated_q_rad": (first_rejected["truncated_q_rad"] if first_rejected else None),
        "last_published_q_rad": last_published,
        "individual_joint_clamping_performed": False,
    }


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
    if any(len(digital[index]) != joint_count or len(real[index]) != joint_count for index in matched):
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
