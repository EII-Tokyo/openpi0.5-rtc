"""Pure, fail-closed ALOHA 14D policy to Isaac 16-DOF order mapping."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any

SCHEMA_VERSION = "a20-policy-runtime-order-v1"
POLICY_DIMENSION = 14
RUNTIME_DIMENSION = 16
GRIPPER_POLICY_INDICES = {6, 13}


def _required_list(container: dict[str, Any], field: str) -> list[Any]:
    value = container.get(field)
    if not isinstance(value, list):
        raise ValueError(f"missing or invalid {field}")
    return value


def _unique_records_by_field(
    records: list[Any], field: str, *, label: str
) -> dict[Any, dict[str, Any]]:
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"invalid {label} record")
    values = [record.get(field) for record in records]
    duplicates = sorted(
        (value for value, count in Counter(values).items() if count > 1),
        key=repr,
    )
    if duplicates:
        raise ValueError(f"duplicate {label} {field}: {duplicates}")
    return {record[field]: record for record in records}


def _finite_float(value: Any, *, field: str, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"invalid {field} for {path}: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {field} for {path}: {value!r}")
    return result


def build_policy_contract(mapping: dict[str, object]) -> dict[str, object]:
    """Extract the trusted canonical 16-DOF/14D contract from the A17 artifact."""
    if not isinstance(mapping, dict):
        raise ValueError("mapping must be an object")
    order = _required_list(mapping, "proposed_canonical_dof_order")
    joint_records = _required_list(mapping, "joint_records")
    if len(order) != RUNTIME_DIMENSION:
        raise ValueError(
            f"invalid canonical DOF count: expected {RUNTIME_DIMENSION}, got {len(order)}"
        )

    dof_records = [
        record
        for record in joint_records
        if isinstance(record, dict) and record.get("is_dof_joint") is True
    ]
    if len(dof_records) != RUNTIME_DIMENSION:
        raise ValueError(
            f"invalid mapped DOF count: expected {RUNTIME_DIMENSION}, "
            f"got {len(dof_records)}"
        )
    by_path = _unique_records_by_field(
        dof_records, "proposed_clean_joint_path", label="canonical DOF"
    )

    canonical_dofs: list[dict[str, object]] = []
    for canonical_index, order_record in enumerate(order):
        if not isinstance(order_record, dict):
            raise ValueError("invalid canonical order record")
        path = order_record.get("clean_joint_path")
        if not isinstance(path, str) or not path:
            raise ValueError(f"invalid canonical path at index {canonical_index}")
        source = by_path.get(path)
        if source is None:
            raise ValueError(f"missing canonical DOF path: {path}")
        raw_mapping = source.get("canonical_mapping")
        if not isinstance(raw_mapping, dict):
            raise ValueError(f"missing canonical mapping for {path}")

        openpi_index = raw_mapping.get("openpi_index")
        dataset_index = raw_mapping.get("dataset_index")
        if isinstance(openpi_index, bool) or not isinstance(openpi_index, int):
            raise ValueError(f"invalid openpi_index for {path}: {openpi_index!r}")
        if isinstance(dataset_index, bool) or not isinstance(dataset_index, int):
            raise ValueError(f"invalid dataset_index for {path}: {dataset_index!r}")
        if dataset_index != openpi_index:
            raise ValueError(
                f"dataset/openpi index mismatch for {path}: "
                f"{dataset_index} != {openpi_index}"
            )
        if order_record.get("openpi_index") != openpi_index:
            raise ValueError(f"canonical order openpi_index mismatch for {path}")
        if order_record.get("dataset_index") != dataset_index:
            raise ValueError(f"canonical order dataset_index mismatch for {path}")

        transform = {
            "sign": _finite_float(raw_mapping.get("sign"), field="sign", path=path),
            "offset": _finite_float(
                raw_mapping.get("offset"), field="offset", path=path
            ),
            "scale": _finite_float(raw_mapping.get("scale"), field="scale", path=path),
        }
        if transform["scale"] == 0.0:
            raise ValueError(f"zero scale for {path}")
        canonical_name = raw_mapping.get("canonical_name")
        if not isinstance(canonical_name, str) or not canonical_name:
            raise ValueError(f"invalid canonical_name for {path}")
        canonical_dofs.append(
            {
                "canonical_index": canonical_index,
                "path": path,
                "canonical_name": canonical_name,
                "openpi_index": openpi_index,
                "dataset_index": dataset_index,
                **transform,
                "unit": raw_mapping.get("unit"),
                "source": raw_mapping.get("source"),
                "isaac_dof_name": raw_mapping.get("isaac_dof_name"),
            }
        )

    canonical_paths = [record["path"] for record in canonical_dofs]
    if len(set(canonical_paths)) != RUNTIME_DIMENSION:
        raise ValueError("duplicate canonical path in canonical order")

    grouped: dict[int, list[dict[str, object]]] = {}
    for record in canonical_dofs:
        grouped.setdefault(int(record["openpi_index"]), []).append(record)
    observed_indices = sorted(grouped)
    if observed_indices != list(range(POLICY_DIMENSION)):
        raise ValueError(
            "invalid OpenPI index inventory: "
            f"expected {list(range(POLICY_DIMENSION))}, got {observed_indices}"
        )

    policy_entries: list[dict[str, object]] = []
    for openpi_index in range(POLICY_DIMENSION):
        records = grouped[openpi_index]
        expected_cardinality = 2 if openpi_index in GRIPPER_POLICY_INDICES else 1
        if len(records) != expected_cardinality:
            raise ValueError(
                f"invalid OpenPI index {openpi_index} cardinality: "
                f"expected {expected_cardinality}, got {len(records)}"
            )
        policy_entries.append(
            {
                "openpi_index": openpi_index,
                "dataset_index": openpi_index,
                "canonical_indices": [record["canonical_index"] for record in records],
                "canonical_paths": [record["path"] for record in records],
                "transforms": [
                    {
                        "path": record["path"],
                        "sign": record["sign"],
                        "offset": record["offset"],
                        "scale": record["scale"],
                    }
                    for record in records
                ],
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "policy_dimension": POLICY_DIMENSION,
        "runtime_dimension": RUNTIME_DIMENSION,
        "canonical_order": canonical_paths,
        "canonical_dofs": canonical_dofs,
        "policy_entries": policy_entries,
    }


def build_order_adapter(
    policy_contract: dict[str, object], runtime_records: list[dict[str, object]]
) -> dict[str, object]:
    """Join a raw runtime DOF order to the trusted canonical contract by path."""
    if not isinstance(policy_contract, dict):
        raise ValueError("policy contract must be an object")
    if policy_contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("invalid policy contract schema_version")
    if policy_contract.get("policy_dimension") != POLICY_DIMENSION:
        raise ValueError("invalid policy contract policy_dimension")
    if policy_contract.get("runtime_dimension") != RUNTIME_DIMENSION:
        raise ValueError("invalid policy contract runtime_dimension")

    canonical_dofs = _required_list(policy_contract, "canonical_dofs")
    policy_entries = _required_list(policy_contract, "policy_entries")
    if len(canonical_dofs) != RUNTIME_DIMENSION:
        raise ValueError("invalid policy contract canonical DOF count")
    if len(policy_entries) != POLICY_DIMENSION:
        raise ValueError("invalid policy contract policy entry count")
    canonical_by_path = _unique_records_by_field(
        canonical_dofs, "path", label="canonical contract"
    )

    if not isinstance(runtime_records, list) or len(runtime_records) != RUNTIME_DIMENSION:
        observed = len(runtime_records) if isinstance(runtime_records, list) else None
        raise ValueError(
            f"invalid runtime DOF count: expected {RUNTIME_DIMENSION}, got {observed}"
        )
    runtime_by_path = _unique_records_by_field(runtime_records, "path", label="runtime")
    _unique_records_by_field(runtime_records, "index", label="runtime")
    runtime_indices = [record.get("index") for record in runtime_records]
    if runtime_indices != list(range(RUNTIME_DIMENSION)):
        raise ValueError(
            "invalid runtime index order: "
            f"expected {list(range(RUNTIME_DIMENSION))}, got {runtime_indices}"
        )

    canonical_paths = list(canonical_by_path)
    runtime_paths = [record["path"] for record in runtime_records]
    missing = sorted(set(canonical_paths) - set(runtime_by_path))
    unexpected = sorted(set(runtime_by_path) - set(canonical_paths))
    if missing or unexpected:
        raise ValueError(
            "runtime path inventory mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    canonical_to_runtime_indices = [
        int(runtime_by_path[path]["index"]) for path in canonical_paths
    ]
    runtime_to_canonical_indices = [0] * RUNTIME_DIMENSION
    for canonical_index, runtime_index in enumerate(canonical_to_runtime_indices):
        runtime_to_canonical_indices[runtime_index] = canonical_index

    policy_to_runtime: list[dict[str, object]] = []
    for expected_index, entry in enumerate(policy_entries):
        if not isinstance(entry, dict) or entry.get("openpi_index") != expected_index:
            raise ValueError(f"invalid policy entry at index {expected_index}")
        paths = _required_list(entry, "canonical_paths")
        transforms = _required_list(entry, "transforms")
        if len(paths) != len(transforms):
            raise ValueError(f"policy entry transform count mismatch at {expected_index}")
        policy_to_runtime.append(
            {
                "openpi_index": expected_index,
                "runtime_indices": [int(runtime_by_path[path]["index"]) for path in paths],
                "transforms": transforms,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "policy_dimension": POLICY_DIMENSION,
        "runtime_dimension": RUNTIME_DIMENSION,
        "runtime_order": runtime_paths,
        "canonical_order": canonical_paths,
        "canonical_to_runtime_indices": canonical_to_runtime_indices,
        "runtime_to_canonical_indices": runtime_to_canonical_indices,
        "policy_to_runtime": policy_to_runtime,
        "mapping_complete": True,
    }
