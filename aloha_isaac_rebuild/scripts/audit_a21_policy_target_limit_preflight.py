"""Pure A21 policy expansion and runtime-limit preflight."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import yaml

from aloha_isaac_rebuild.scripts.a20_policy_runtime_order_adapter import SCHEMA_VERSION as A20_SCHEMA_VERSION
from aloha_isaac_rebuild.scripts.a20_policy_runtime_order_adapter import policy_to_runtime
from aloha_isaac_rebuild.scripts.a20_policy_runtime_order_adapter import runtime_to_policy

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
    return entries


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


def _validate_right_finger_provenance(adapter: dict[str, object], entries: list[dict[str, Any]]) -> None:
    canonical_dofs = adapter.get("canonical_dofs")
    if not isinstance(canonical_dofs, list) or len(canonical_dofs) != _RUNTIME_DIMENSION:
        raise ValueError("missing adapter canonical_dofs provenance")
    if not all(isinstance(record, dict) for record in canonical_dofs):
        raise ValueError("invalid adapter canonical_dofs provenance")
    by_path: dict[str, dict[str, Any]] = {}
    for record in canonical_dofs:
        path = record.get("path")
        if not isinstance(path, str) or not path or path in by_path:
            raise ValueError("invalid adapter canonical_dofs path inventory")
        by_path[path] = record

    candidates: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for policy_index in sorted(GRIPPER_POLICY_INDICES):
        entry = entries[policy_index]
        transforms = entry.get("transforms")
        if not isinstance(transforms, list):
            raise ValueError(f"invalid gripper transforms at policy index {policy_index}")
        for transform in transforms:
            if not isinstance(transform, dict):
                raise ValueError(f"invalid gripper transform at policy index {policy_index}")
            path = transform.get("path")
            record = by_path.get(path) if isinstance(path, str) else None
            if record is None:
                raise ValueError(f"missing canonical provenance for path {path!r}")
            source = record.get("source_transform")
            if not isinstance(source, dict):
                raise ValueError(f"missing source transform provenance for {path}")
            source_scale = _finite_float(source.get("scale"), field=f"source scale for {path}")
            if source_scale < 0.0:
                candidates.append((path, transform, record))

    if len(candidates) != 2:
        raise ValueError("right-finger provenance must identify exactly two negative source transforms")
    for path, transform, record in candidates:
        source = record["source_transform"]
        effective = record.get("effective_transform")
        override = record.get("clean_runtime_mapping_override")
        if not isinstance(effective, dict) or not isinstance(override, dict) or not override:
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
        for field in ("rationale", "source"):
            value = override.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"missing override {field} provenance for {path}")


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
        _validate_right_finger_provenance(adapter, entries)
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
    if inputs is not None and not isinstance(inputs, dict):
        errors.append(_error("invalid_inputs", "inputs must be an object"))
    try:
        _validate_a20_layer(
            layer1,
            expected_status="PASS_A20_USD_DOF_METADATA",
            layer_name="layer1",
        )
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
            record_sets.append(records)
        if any(records != record_sets[0] for records in record_sets[1:]):
            raise ValueError("layer2 runtime record sets are not deterministic")
        runtime_records = record_sets[0]
    except (TypeError, ValueError) as exc:
        errors.append(_error("invalid_layer2_evidence", str(exc)))

    if errors or adapter is None or runtime_records is None:
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
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


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
