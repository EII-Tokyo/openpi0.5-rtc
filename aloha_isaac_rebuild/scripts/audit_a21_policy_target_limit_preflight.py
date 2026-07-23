"""Pure A21 policy expansion and runtime-limit preflight."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import yaml

from aloha_isaac_rebuild.scripts.a20_policy_runtime_order_adapter import SCHEMA_VERSION as A20_SCHEMA_VERSION
from aloha_isaac_rebuild.scripts.a20_policy_runtime_order_adapter import policy_to_runtime
from aloha_isaac_rebuild.scripts.a20_policy_runtime_order_adapter import runtime_to_policy
from aloha_isaac_rebuild.scripts.run_a20_runtime_articulation_discovery import is_exact_runtime_pass

SCHEMA_VERSION = "a21-policy-target-limit-v1"
PASS_STATUS = "PASS_A21_POLICY_TARGET_LIMIT_PREFLIGHT"
FAIL_STATUS = "FAIL_A21_POLICY_TARGET_LIMIT_PREFLIGHT"
ARM_DELTA_RAD = math.radians(0.25)
GRIPPER_POLICY_INDICES = {6, 13}
DEFAULT_CONFIG = Path("aloha_isaac_rebuild/configs/physical_reconstruction/stationary_style_rebuild.yaml")

_POLICY_DIMENSION = 14
_RUNTIME_DIMENSION = 16
_LIMIT_TOLERANCE = 1e-9
_ROUND_TRIP_TOLERANCE = 1e-12
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RIGHT_FINGER_OVERRIDE_PATHS = (
    "/aloha/joints/left_right_finger",
    "/aloha/joints/right_right_finger",
)
_SAFETY_FLAGS = {
    "physics_stepped": False,
    "actions_applied": False,
    "targets_written": False,
    "targets_restored": False,
    "stage_saved": False,
}


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def build_reviewed_policy_samples() -> list[dict[str, object]]:
    """Return fresh copies of the four reviewed 14D policy samples."""
    samples: list[dict[str, object]] = []
    for label, gripper_value in (
        ("grippers_closed", 0.0),
        ("grippers_mid", 0.5),
        ("grippers_open", 1.0),
    ):
        values = [0.0] * _POLICY_DIMENSION
        for index in GRIPPER_POLICY_INDICES:
            values[index] = gripper_value
        samples.append({"label": label, "policy_values": values})

    signed_values = [ARM_DELTA_RAD if index % 2 == 0 else -ARM_DELTA_RAD for index in range(_POLICY_DIMENSION)]
    for index in GRIPPER_POLICY_INDICES:
        signed_values[index] = 0.5
    samples.append({"label": "signed_arm_micro_targets", "policy_values": signed_values})
    return samples


def runtime_bounds(record: dict[str, object]) -> tuple[float, float]:
    """Convert authored runtime limits to tensor target units."""
    if not isinstance(record, dict):
        raise ValueError("runtime record must be an object")
    lower = _finite_float(record.get("lower_limit"), field="lower_limit")
    upper = _finite_float(record.get("upper_limit"), field="upper_limit")
    if lower >= upper:
        raise ValueError("runtime lower_limit must be less than upper_limit")
    joint_type = record.get("joint_type")
    if joint_type == "PhysicsRevoluteJoint":
        return math.radians(lower), math.radians(upper)
    if joint_type == "PhysicsPrismaticJoint":
        return lower, upper
    raise ValueError(f"unsupported runtime joint_type: {joint_type!r}")


def _error(code: str, message: str, **fields: object) -> dict[str, object]:
    return {"code": code, "message": message, **fields}


def _validate_adapter_shape(adapter: object) -> list[dict[str, Any]]:
    if not isinstance(adapter, dict):
        raise ValueError("adapter must be an object")
    if adapter.get("schema_version") != A20_SCHEMA_VERSION:
        raise ValueError("invalid A20 adapter schema_version")
    if adapter.get("policy_dimension") != _POLICY_DIMENSION:
        raise ValueError("invalid A20 adapter policy_dimension")
    if adapter.get("runtime_dimension") != _RUNTIME_DIMENSION:
        raise ValueError("invalid A20 adapter runtime_dimension")
    entries = adapter.get("policy_to_runtime")
    if not isinstance(entries, list) or len(entries) != _POLICY_DIMENSION:
        raise ValueError("invalid A20 adapter policy entry count")
    if not all(isinstance(entry, dict) for entry in entries):
        raise ValueError("invalid A20 adapter policy entry")
    if adapter.get("mapping_complete") is not True:
        raise ValueError("A20 adapter mapping_complete must be exactly true")
    canonical_to_runtime = _adapter_index_vector(adapter, "canonical_to_runtime_indices")
    runtime_to_canonical = _adapter_index_vector(adapter, "runtime_to_canonical_indices")
    for canonical_index, runtime_index in enumerate(canonical_to_runtime):
        if runtime_to_canonical[runtime_index] != canonical_index:
            raise ValueError("A20 adapter index vectors must be mutual inverses")
    _validate_recorded_round_trip(adapter.get("round_trip_check"))
    return entries


def _adapter_index_vector(adapter: dict[str, object], field: str) -> list[int]:
    values = adapter.get(field)
    if not isinstance(values, list) or len(values) != _RUNTIME_DIMENSION:
        raise ValueError(f"invalid A20 adapter {field}")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError(f"invalid A20 adapter {field} integer")
    if sorted(values) != list(range(_RUNTIME_DIMENSION)):
        raise ValueError(f"A20 adapter {field} inventory must be exactly 0..15")
    return values


def _validate_recorded_round_trip(proof: object) -> None:
    expected_fields = {
        "status",
        "sample_count",
        "gripper_values",
        "max_abs_error",
        "error",
    }
    if not isinstance(proof, dict) or set(proof) != expected_fields:
        raise ValueError("invalid A20 adapter round_trip_check shape")
    sample_count = proof.get("sample_count")
    if (
        proof.get("status") != "PASS"
        or isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count != 3
        or proof.get("error") is not None
    ):
        raise ValueError("A20 adapter round_trip_check is not an exact PASS")
    gripper_values = proof.get("gripper_values")
    if (
        not isinstance(gripper_values, list)
        or len(gripper_values) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value))
            for value in gripper_values
        )
        or [float(value) for value in gripper_values] != [0.0, 0.5, 1.0]
    ):
        raise ValueError("invalid A20 adapter round_trip_check gripper_values")
    max_abs_error = _finite_float(proof.get("max_abs_error"), field="round_trip_check max_abs_error")
    if not 0.0 <= max_abs_error <= _ROUND_TRIP_TOLERANCE:
        raise ValueError("A20 adapter round_trip_check error exceeds tolerance")


def _validate_runtime_records(
    runtime_records: object,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    if not isinstance(runtime_records, list) or len(runtime_records) != _RUNTIME_DIMENSION:
        raise ValueError("runtime_records must contain exactly 16 objects")
    if not all(isinstance(record, dict) for record in runtime_records):
        raise ValueError("runtime_records must contain exactly 16 objects")

    by_index: dict[int, dict[str, Any]] = {}
    seen_paths: set[str] = set()
    for record in runtime_records:
        index = record.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"invalid runtime index: {index!r}")
        if index in by_index:
            raise ValueError(f"duplicate runtime index: {index}")
        path = record.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError(f"invalid runtime path at index {index}")
        if path in seen_paths:
            raise ValueError(f"duplicate runtime path: {path}")
        by_index[index] = record
        seen_paths.add(path)
    if sorted(by_index) != list(range(_RUNTIME_DIMENSION)):
        raise ValueError("runtime index inventory must be exactly 0..15")

    ordered = [by_index[index] for index in range(_RUNTIME_DIMENSION)]
    policy_by_runtime: dict[int, int] = {}
    return ordered, policy_by_runtime


def _validate_right_finger_provenance(
    adapter: dict[str, object],
    entries: list[dict[str, Any]],
    runtime_records: list[dict[str, Any]],
    canonical_dofs: object,
) -> None:
    if not isinstance(canonical_dofs, list) or len(canonical_dofs) != _RUNTIME_DIMENSION:
        raise ValueError("missing canonical_dofs provenance")
    if not all(isinstance(record, dict) for record in canonical_dofs):
        raise ValueError("invalid canonical_dofs provenance")
    by_path: dict[str, dict[str, Any]] = {}
    policy_counts = dict.fromkeys(range(_POLICY_DIMENSION), 0)
    for expected_canonical_index, record in enumerate(canonical_dofs):
        canonical_index = record.get("canonical_index")
        if (
            isinstance(canonical_index, bool)
            or not isinstance(canonical_index, int)
            or canonical_index != expected_canonical_index
        ):
            raise ValueError("canonical index inventory must be exactly 0..15")
        path = record.get("path")
        if not isinstance(path, str) or not path or path in by_path:
            raise ValueError("invalid canonical_dofs path inventory")
        openpi_index = record.get("openpi_index")
        if (
            isinstance(openpi_index, bool)
            or not isinstance(openpi_index, int)
            or not 0 <= openpi_index < _POLICY_DIMENSION
        ):
            raise ValueError(f"invalid canonical openpi_index for {path}")
        dataset_index = record.get("dataset_index")
        if isinstance(dataset_index, bool) or not isinstance(dataset_index, int) or dataset_index != openpi_index:
            raise ValueError(f"invalid canonical dataset_index for {path}")
        policy_counts[openpi_index] += 1
        by_path[path] = record
    expected_policy_counts = {index: 2 if index in GRIPPER_POLICY_INDICES else 1 for index in range(_POLICY_DIMENSION)}
    if policy_counts != expected_policy_counts:
        raise ValueError("canonical policy index inventory must be exactly 0..13")

    canonical_order = adapter.get("canonical_order")
    if canonical_order != [record["path"] for record in canonical_dofs]:
        raise ValueError("adapter canonical_order does not match Layer1 provenance")
    runtime_order = adapter.get("runtime_order")
    runtime_paths = [record["path"] for record in runtime_records]
    if runtime_order != runtime_paths:
        raise ValueError("adapter runtime_order does not match runtime records")
    expected_canonical_to_runtime = [runtime_paths.index(path) for path in canonical_order]
    if adapter.get("canonical_to_runtime_indices") != expected_canonical_to_runtime:
        raise ValueError("adapter canonical_to_runtime_indices do not match path join")
    expected_runtime_to_canonical = [0] * _RUNTIME_DIMENSION
    for canonical_index, runtime_index in enumerate(expected_canonical_to_runtime):
        expected_runtime_to_canonical[runtime_index] = canonical_index
    if adapter.get("runtime_to_canonical_indices") != expected_runtime_to_canonical:
        raise ValueError("adapter runtime_to_canonical_indices do not match path join")

    transform_paths: list[str] = []
    runtime_indices_seen: set[int] = set()
    transforms_by_path: dict[str, dict[str, Any]] = {}
    for policy_index, entry in enumerate(entries):
        entry_index = entry.get("openpi_index")
        if isinstance(entry_index, bool) or not isinstance(entry_index, int) or entry_index != policy_index:
            raise ValueError(f"invalid adapter openpi_index at policy index {policy_index}")
        runtime_indices = entry.get("runtime_indices")
        transforms = entry.get("transforms")
        if not isinstance(runtime_indices, list) or not isinstance(transforms, list):
            raise ValueError(f"invalid transforms at policy index {policy_index}")
        expected_cardinality = 2 if policy_index in GRIPPER_POLICY_INDICES else 1
        if len(runtime_indices) != expected_cardinality or len(transforms) != expected_cardinality:
            raise ValueError(f"invalid adapter cardinality at policy index {policy_index}")
        for runtime_index, transform in zip(runtime_indices, transforms, strict=True):
            if (
                isinstance(runtime_index, bool)
                or not isinstance(runtime_index, int)
                or not 0 <= runtime_index < _RUNTIME_DIMENSION
                or runtime_index in runtime_indices_seen
            ):
                raise ValueError(f"invalid or duplicate runtime index at policy index {policy_index}")
            if not isinstance(transform, dict):
                raise ValueError(f"invalid transform at policy index {policy_index}")
            path = transform.get("path")
            record = by_path.get(path) if isinstance(path, str) else None
            if record is None:
                raise ValueError(f"missing canonical provenance for path {path!r}")
            runtime_record = runtime_records[runtime_index]
            if path != runtime_record.get("path"):
                raise ValueError(f"adapter transform path does not match runtime index {runtime_index}")
            if record.get("openpi_index") != policy_index:
                raise ValueError(f"Layer1 canonical policy index mismatch for path {path}")
            expected_unit = {
                "PhysicsRevoluteJoint": "rad",
                "PhysicsPrismaticJoint": "m",
            }.get(runtime_record.get("joint_type"))
            if expected_unit is None or record.get("unit") != expected_unit:
                raise ValueError(f"canonical unit does not match runtime joint type for {path}")
            effective = record.get("effective_transform")
            if not isinstance(effective, dict):
                raise ValueError(f"missing effective transform provenance for {path}")
            transform_values = tuple(
                _finite_float(transform.get(field), field=f"adapter {field} for {path}")
                for field in ("sign", "offset", "scale")
            )
            effective_values = tuple(
                _finite_float(effective.get(field), field=f"effective {field} for {path}")
                for field in ("sign", "offset", "scale")
            )
            if transform_values != effective_values:
                raise ValueError(f"adapter/effective transform mismatch for {path}")
            runtime_indices_seen.add(runtime_index)
            transform_paths.append(path)
            transforms_by_path[path] = transform

    if runtime_indices_seen != set(range(_RUNTIME_DIMENSION)):
        raise ValueError("adapter runtime index inventory must be exactly 0..15")
    if (
        len(transform_paths) != _RUNTIME_DIMENSION
        or len(set(transform_paths)) != _RUNTIME_DIMENSION
        or set(transform_paths) != set(by_path)
    ):
        raise ValueError("adapter transform paths do not match Layer1 canonical_dofs")

    for path in _RIGHT_FINGER_OVERRIDE_PATHS:
        record = by_path.get(path)
        transform = transforms_by_path.get(path)
        if record is None or transform is None:
            raise ValueError(f"missing reviewed right-finger identity: {path}")
        source = record.get("source_transform")
        effective = record.get("effective_transform")
        override = record.get("clean_runtime_mapping_override")
        if (
            not isinstance(source, dict)
            or not isinstance(effective, dict)
            or not isinstance(override, dict)
            or not override
        ):
            raise ValueError(f"missing effective override provenance for {path}")
        source_values = tuple(
            _finite_float(source.get(field), field=f"source {field} for {path}")
            for field in ("sign", "offset", "scale")
        )
        effective_values = tuple(
            _finite_float(effective.get(field), field=f"effective {field} for {path}")
            for field in ("sign", "offset", "scale")
        )
        override_values = tuple(
            _finite_float(override.get(field), field=f"override {field} for {path}")
            for field in ("sign", "offset", "scale")
        )
        transform_values = tuple(
            _finite_float(transform.get(field), field=f"adapter {field} for {path}")
            for field in ("sign", "offset", "scale")
        )
        if not all(value < 0.0 for value in source_values):
            raise ValueError(f"source transform is not negative for {path}")
        if not all(value > 0.0 for value in effective_values):
            raise ValueError(f"effective transform is not positive for {path}")
        if override_values != effective_values or transform_values != effective_values:
            raise ValueError(f"override/effective adapter mismatch for {path}")
        if record.get("unit") != "m" or override.get("unit") != record.get("unit"):
            raise ValueError(f"override unit mismatch for {path}")
        for field in ("rationale", "source"):
            value = override.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"missing override {field} provenance for {path}")
    for path, record in by_path.items():
        if path in _RIGHT_FINGER_OVERRIDE_PATHS or record.get("openpi_index") not in GRIPPER_POLICY_INDICES:
            continue
        if record.get("clean_runtime_mapping_override") is not None:
            raise ValueError(f"unexpected gripper override identity for {path}")


def _validated_samples(samples: object) -> list[tuple[str, list[float]]]:
    if not isinstance(samples, list):
        raise ValueError("samples must be a list")
    result: list[tuple[str, list[float]]] = []
    labels: set[str] = set()
    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"invalid sample at index {sample_index}")
        label = sample.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError(f"invalid sample label at index {sample_index}")
        if label in labels:
            raise ValueError(f"duplicate sample label: {label}")
        labels.add(label)
        values = sample.get("policy_values")
        if not isinstance(values, list) or len(values) != _POLICY_DIMENSION:
            raise ValueError(f"invalid policy_values for sample {label}")
        result.append(
            (
                label,
                [
                    _finite_float(value, field=f"policy value {index} for sample {label}")
                    for index, value in enumerate(values)
                ],
            )
        )
    return result


def evaluate_policy_samples(
    adapter: dict[str, object],
    runtime_records: list[dict[str, object]],
    samples: list[dict[str, object]],
    *,
    canonical_dofs: list[dict[str, object]],
) -> dict[str, object]:
    """Expand policy samples, validate live-unit limits, and invert the mapping."""
    sample_snapshot = deepcopy(samples) if isinstance(samples, list) else []
    mismatches: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    try:
        entries = _validate_adapter_shape(adapter)
        ordered_records, policy_by_runtime = _validate_runtime_records(runtime_records)
        for policy_index, entry in enumerate(entries):
            runtime_indices = entry.get("runtime_indices")
            if isinstance(runtime_indices, list):
                for runtime_index in runtime_indices:
                    if isinstance(runtime_index, int) and not isinstance(runtime_index, bool):
                        policy_by_runtime[runtime_index] = policy_index
        validated_samples = _validated_samples(samples)
        bounds = [runtime_bounds(record) for record in ordered_records]
    except (TypeError, ValueError) as exc:
        errors.append(_error("invalid_preflight_input", str(exc)))
        return {
            "ok": False,
            "sample_count": len(sample_snapshot),
            "samples": sample_snapshot,
            "mismatches": mismatches,
            "errors": errors,
            "max_arm_delta_rad": ARM_DELTA_RAD,
        }

    try:
        _validate_right_finger_provenance(
            adapter,
            entries,
            ordered_records,
            canonical_dofs,
        )
    except (TypeError, ValueError) as exc:
        errors.append(_error("invalid_right_finger_override_provenance", str(exc)))

    for label, policy_values in validated_samples:
        try:
            runtime_values = policy_to_runtime(policy_values, adapter)
            if not isinstance(runtime_values, list) or len(runtime_values) != (_RUNTIME_DIMENSION):
                raise ValueError("policy expansion did not produce 16 targets")
            finite_targets = [
                _finite_float(value, field=f"runtime target {index}") for index, value in enumerate(runtime_values)
            ]
        except (TypeError, ValueError) as exc:
            errors.append(_error("policy_conversion_error", str(exc), label=label))
            continue

        for runtime_index, (target, (lower, upper)) in enumerate(zip(finite_targets, bounds, strict=True)):
            if target < lower - _LIMIT_TOLERANCE or target > upper + _LIMIT_TOLERANCE:
                record = ordered_records[runtime_index]
                mismatches.append(
                    {
                        "label": label,
                        "policy_index": policy_by_runtime.get(runtime_index),
                        "runtime_index": runtime_index,
                        "path": record["path"],
                        "target": target,
                        "lower": lower,
                        "upper": upper,
                        "code": "target_outside_runtime_limits",
                    }
                )
        try:
            recovered = runtime_to_policy(finite_targets, adapter, tolerance=_ROUND_TRIP_TOLERANCE)
            if not isinstance(recovered, list) or len(recovered) != _POLICY_DIMENSION:
                raise ValueError("inverse conversion did not produce 14 values")
            disagreements = [
                index
                for index, (expected, observed) in enumerate(zip(policy_values, recovered, strict=True))
                if not math.isclose(
                    expected,
                    _finite_float(observed, field=f"recovered policy value {index}"),
                    rel_tol=0.0,
                    abs_tol=_ROUND_TRIP_TOLERANCE,
                )
            ]
            if disagreements:
                raise ValueError(f"inverse round-trip disagreement at policy indices {disagreements}")
        except (TypeError, ValueError) as exc:
            errors.append(_error("round_trip_error", str(exc), label=label))

    return {
        "ok": not errors and not mismatches,
        "sample_count": len(validated_samples),
        "samples": sample_snapshot,
        "mismatches": mismatches,
        "errors": errors,
        "max_arm_delta_rad": ARM_DELTA_RAD,
    }


def evaluate_preflight(
    layer1: object,
    layer2: object,
    *,
    inputs: dict[str, object] | None = None,
) -> dict[str, object]:
    """Validate exact A20 evidence, then run the pure policy-limit checks."""
    bound_inputs = deepcopy(inputs) if isinstance(inputs, dict) else {}
    errors: list[dict[str, object]] = []
    canonical_dofs: list[dict[str, object]] | None = None
    trusted_a20_inputs: dict[str, object] | None = None
    layer1_object: dict[str, object] | None = None
    if inputs is not None and not isinstance(inputs, dict):
        errors.append(_error("invalid_inputs", "inputs must be an object"))
    try:
        layer1_object = _validate_a20_layer(
            layer1,
            expected_status="PASS_A20_USD_DOF_METADATA",
            layer_name="layer1",
        )
        canonical_dofs = _layer1_canonical_dofs(layer1_object)
        trusted_a20_inputs = _layer1_inputs(layer1_object)
    except (TypeError, ValueError) as exc:
        errors.append(_error("invalid_layer1_evidence", str(exc)))

    runtime_records: list[dict[str, object]] | None = None
    adapter: dict[str, object] | None = None
    try:
        layer2_object = _validate_a20_layer(
            layer2,
            expected_status=("PASS_A20_RUNTIME_ARTICULATION_DISCOVERY_NO_STEP"),
            layer_name="layer2",
        )
        if layer1_object is None or not is_exact_runtime_pass(
            layer2_object,
            layer1_object,
        ):
            raise ValueError("layer2 is not a complete exact A20 runtime pass")
        adapter_value = layer2_object.get("order_adapter")
        if not isinstance(adapter_value, dict):
            raise ValueError("layer2 order_adapter must be an object")
        adapter = adapter_value
        if layer2_object.get("run_count") != 3:
            raise ValueError("layer2 run_count must equal 3")
        runs = layer2_object.get("runs")
        if not isinstance(runs, list) or len(runs) != 3:
            raise ValueError("layer2 runs must contain exactly three runs")
        record_sets: list[list[dict[str, object]]] = []
        for run_index, run in enumerate(runs):
            if not isinstance(run, dict):
                raise ValueError(f"layer2 run {run_index} must be an object")
            _require_false_safety_flags(run, label=f"layer2 run {run_index}")
            records = run.get("records")
            if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
                raise ValueError(f"layer2 run {run_index} records must be a list of objects")
            if trusted_a20_inputs is None or run.get("inputs") != trusted_a20_inputs:
                raise ValueError(f"layer2 run {run_index} inputs do not match Layer1 provenance")
            record_sets.append(records)
        if any(records != record_sets[0] for records in record_sets[1:]):
            raise ValueError("layer2 runtime record sets are not deterministic")
        runtime_records = record_sets[0]
    except (TypeError, ValueError) as exc:
        errors.append(_error("invalid_layer2_evidence", str(exc)))

    if errors or adapter is None or runtime_records is None or canonical_dofs is None:
        return _preflight_result(
            inputs=bound_inputs,
            samples=[],
            mismatches=[],
            errors=errors,
        )

    evaluation = evaluate_policy_samples(
        adapter,
        runtime_records,
        build_reviewed_policy_samples(),
        canonical_dofs=canonical_dofs,
    )
    return _preflight_result(
        inputs=bound_inputs,
        samples=evaluation["samples"],
        mismatches=evaluation["mismatches"],
        errors=evaluation["errors"],
    )


def _require_false_safety_flags(payload: dict[str, object], *, label: str) -> None:
    for flag in (
        "physics_stepped",
        "actions_applied",
        "targets_written",
        "stage_saved",
    ):
        if payload.get(flag) is not False:
            raise ValueError(f"{label} {flag} must be exactly false")


def _validate_a20_layer(layer: object, *, expected_status: str, layer_name: str) -> dict[str, object]:
    if not isinstance(layer, dict):
        raise ValueError(f"{layer_name} must be an object")
    if layer.get("status") != expected_status:
        raise ValueError(f"{layer_name} status is not the exact A20 pass status")
    if layer.get("ok") is not True:
        raise ValueError(f"{layer_name} ok must be exactly true")
    if layer.get("errors") != []:
        raise ValueError(f"{layer_name} errors must be an empty list")
    if layer.get("mismatches") != []:
        raise ValueError(f"{layer_name} mismatches must be an empty list")
    _require_false_safety_flags(layer, label=layer_name)
    return layer


def _layer1_canonical_dofs(layer1: dict[str, object]) -> list[dict[str, object]]:
    contract = layer1.get("policy_contract")
    if not isinstance(contract, dict):
        raise ValueError("layer1 policy_contract must be an object")
    if contract.get("schema_version") != A20_SCHEMA_VERSION:
        raise ValueError("invalid layer1 policy_contract schema_version")
    if contract.get("policy_dimension") != _POLICY_DIMENSION:
        raise ValueError("invalid layer1 policy_contract policy_dimension")
    if contract.get("runtime_dimension") != _RUNTIME_DIMENSION:
        raise ValueError("invalid layer1 policy_contract runtime_dimension")
    canonical_order = contract.get("canonical_order")
    if (
        not isinstance(canonical_order, list)
        or len(canonical_order) != _RUNTIME_DIMENSION
        or not all(isinstance(path, str) and path for path in canonical_order)
        or len(set(canonical_order)) != _RUNTIME_DIMENSION
    ):
        raise ValueError("invalid layer1 policy_contract canonical_order")
    canonical_dofs = contract.get("canonical_dofs")
    if not isinstance(canonical_dofs, list) or len(canonical_dofs) != _RUNTIME_DIMENSION:
        raise ValueError("invalid layer1 policy_contract canonical_dofs")
    if not all(isinstance(record, dict) for record in canonical_dofs):
        raise ValueError("invalid layer1 canonical_dofs record")
    observed_indices: list[int] = []
    observed_paths: list[str] = []
    policy_counts = dict.fromkeys(range(_POLICY_DIMENSION), 0)
    for record in canonical_dofs:
        canonical_index = record.get("canonical_index")
        if isinstance(canonical_index, bool) or not isinstance(canonical_index, int):
            raise ValueError("invalid Layer1 canonical_index")
        path = record.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("invalid Layer1 canonical path")
        openpi_index = record.get("openpi_index")
        if (
            isinstance(openpi_index, bool)
            or not isinstance(openpi_index, int)
            or not 0 <= openpi_index < _POLICY_DIMENSION
        ):
            raise ValueError(f"invalid Layer1 openpi_index for {path}")
        dataset_index = record.get("dataset_index")
        if (
            isinstance(dataset_index, bool)
            or not isinstance(dataset_index, int)
            or not 0 <= dataset_index < _POLICY_DIMENSION
            or dataset_index != openpi_index
        ):
            raise ValueError(f"invalid Layer1 dataset_index for {path}")
        unit = record.get("unit")
        if not isinstance(unit, str) or not unit:
            raise ValueError(f"invalid Layer1 canonical unit for {path}")
        for transform_name in ("source_transform", "effective_transform"):
            transform = record.get(transform_name)
            if not isinstance(transform, dict):
                raise ValueError(f"missing Layer1 {transform_name} for {path}")
            for field in ("sign", "offset", "scale"):
                _finite_float(
                    transform.get(field),
                    field=f"Layer1 {transform_name} {field} for {path}",
                )
            if transform.get("scale") == 0.0:
                raise ValueError(f"zero Layer1 {transform_name} scale for {path}")
        override = record.get("clean_runtime_mapping_override")
        if override is not None and not isinstance(override, dict):
            raise ValueError(f"invalid Layer1 override provenance for {path}")
        if isinstance(override, dict) and override.get("unit") != unit:
            raise ValueError(f"Layer1 override unit mismatch for {path}")
        observed_indices.append(canonical_index)
        observed_paths.append(path)
        policy_counts[openpi_index] += 1
    if observed_indices != list(range(_RUNTIME_DIMENSION)):
        raise ValueError("invalid Layer1 canonical index inventory")
    if observed_paths != canonical_order:
        raise ValueError("Layer1 canonical_dofs do not match canonical_order")
    expected_counts = {index: 2 if index in GRIPPER_POLICY_INDICES else 1 for index in range(_POLICY_DIMENSION)}
    if policy_counts != expected_counts:
        raise ValueError("invalid Layer1 OpenPI policy index inventory")
    return canonical_dofs


def _absolute_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f"{field} must be an absolute path")
    return value


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA256")
    return value


def _layer1_inputs(layer1: dict[str, object]) -> dict[str, object]:
    inputs = layer1.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("layer1 inputs must be an object")
    trusted: dict[str, object] = {}
    for name in ("config", "mapping"):
        binding = inputs.get(name)
        if not isinstance(binding, dict):
            raise ValueError(f"layer1 {name} input must be an object")
        trusted[name] = {
            "path": _absolute_path(binding.get("path"), field=f"layer1 {name} input path"),
            "sha256": _sha256(binding.get("sha256"), field=f"layer1 {name} input sha256"),
        }
    stage = inputs.get("stage")
    if not isinstance(stage, dict):
        raise ValueError("layer1 stage input must be an object")
    stage_path = _absolute_path(stage.get("path"), field="layer1 stage input path")
    pre_hash = _sha256(stage.get("pre_sha256"), field="layer1 stage pre_sha256")
    post_hash = _sha256(stage.get("post_sha256"), field="layer1 stage post_sha256")
    if pre_hash != post_hash or stage.get("consistent_during_audit") is not True:
        raise ValueError("layer1 stage hashes must be consistent during audit")
    trusted["stage"] = {"path": stage_path, "sha256": pre_hash}
    return trusted


def _preflight_result(
    *,
    inputs: dict[str, object],
    samples: object,
    mismatches: object,
    errors: object,
) -> dict[str, object]:
    safe_samples = deepcopy(samples) if isinstance(samples, list) else []
    safe_mismatches = deepcopy(mismatches) if isinstance(mismatches, list) else []
    safe_errors = (
        deepcopy(errors) if isinstance(errors, list) else [_error("invalid_internal_result", "errors must be a list")]
    )
    ok = not safe_mismatches and not safe_errors and len(safe_samples) == 4
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": ok,
        "status": PASS_STATUS if ok else FAIL_STATUS,
        "inputs": deepcopy(inputs),
        "sample_count": len(safe_samples),
        "samples": safe_samples,
        "mismatches": safe_mismatches,
        "errors": safe_errors,
        **_SAFETY_FLAGS,
    }


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _configured_path(repo: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing or invalid configured output {field}")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else repo / path).resolve()


def _load_json_with_binding(path: Path, *, input_name: str, inputs: dict[str, object]) -> object:
    inputs[input_name] = {"path": str(path.resolve()), "sha256": None}
    content = path.read_bytes()
    inputs[input_name] = {
        "path": str(path.resolve()),
        "sha256": _sha256_bytes(content),
    }
    return json.loads(content.decode("utf-8"))


def _atomic_write(path: Path, payload: dict[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    previous_content = path.read_bytes() if path.exists() else None
    previous_failure = (
        previous_content if previous_content is not None and _is_serialized_failure(previous_content) else None
    )
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    replaced = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        replaced = True
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        if replaced:
            _restore_failure_or_remove(path, previous_failure)
        elif previous_content is not None and previous_failure is None:
            path.unlink(missing_ok=True)
        raise


def _is_serialized_failure(content: bytes) -> bool:
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("status") == FAIL_STATUS and payload.get("ok") is False


def _restore_failure_or_remove(path: Path, previous_failure: bytes | None) -> None:
    path.unlink(missing_ok=True)
    if previous_failure is None:
        return

    descriptor: int | None = None
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.rollback.",
            suffix=".tmp",
            dir=path.parent,
        )
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(previous_failure)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        path.unlink(missing_ok=True)


def _exact_pass(result: object) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("schema_version") != SCHEMA_VERSION:
        return False
    if result.get("status") != PASS_STATUS or result.get("ok") is not True:
        return False
    if result.get("sample_count") != 4:
        return False
    if not isinstance(result.get("samples"), list) or len(result["samples"]) != 4:
        return False
    if result.get("mismatches") != [] or result.get("errors") != []:
        return False
    return all(result.get(flag) is False for flag in _SAFETY_FLAGS)


def _cli_failure(
    *,
    inputs: dict[str, object],
    code: str,
    message: str,
) -> dict[str, object]:
    return _preflight_result(
        inputs=inputs,
        samples=[],
        mismatches=[],
        errors=[_error(code, message)],
    )


def main() -> int:
    """Load configured A20 evidence, write A21 JSON atomically, and fail closed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    repo = Path.cwd().resolve()
    config_path = args.config.expanduser().resolve()
    inputs: dict[str, object] = {"config": {"path": str(config_path), "sha256": None}}
    output_path: Path | None = None
    try:
        config_bytes = config_path.read_bytes()
        inputs["config"] = {
            "path": str(config_path),
            "sha256": _sha256_bytes(config_bytes),
        }
        config = yaml.safe_load(config_bytes.decode("utf-8"))
        if not isinstance(config, dict):
            raise ValueError("config must be an object")
        outputs = config.get("outputs")
        if not isinstance(outputs, dict):
            raise ValueError("config outputs must be an object")
        output_path = _configured_path(
            repo,
            outputs.get("a21_policy_target_limit_preflight_json"),
            field="a21_policy_target_limit_preflight_json",
        )
        layer1_path = _configured_path(
            repo,
            outputs.get("a20_usd_dof_metadata_json"),
            field="a20_usd_dof_metadata_json",
        )
        layer2_path = _configured_path(
            repo,
            outputs.get("a20_runtime_articulation_discovery_json"),
            field="a20_runtime_articulation_discovery_json",
        )
        layer1 = _load_json_with_binding(layer1_path, input_name="layer1", inputs=inputs)
        layer2 = _load_json_with_binding(layer2_path, input_name="layer2", inputs=inputs)
        result = evaluate_preflight(layer1, layer2, inputs=inputs)
    except Exception as exc:
        result = _cli_failure(
            inputs=inputs,
            code="cli_input_error",
            message=str(exc),
        )

    if output_path is not None:
        try:
            _atomic_write(output_path, result)
        except Exception as exc:
            result = _cli_failure(
                inputs=inputs,
                code="output_write_error",
                message=str(exc),
            )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if _exact_pass(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
