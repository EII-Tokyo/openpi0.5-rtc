from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from aloha_isaac_rebuild.scripts.a20_policy_runtime_order_adapter import build_order_adapter
from aloha_isaac_rebuild.scripts.a20_policy_runtime_order_adapter import build_policy_contract

ROOT = Path(__file__).resolve().parents[2]
MAPPING = (
    ROOT
    / "aloha_isaac_rebuild/artifacts/validation/a17_clean_articulation_mapping_plan.json"
)
INTERLEAVED_PATHS = [
    "/aloha/joints/left_waist",
    "/aloha/joints/right_waist",
    "/aloha/joints/left_shoulder",
    "/aloha/joints/right_shoulder",
    "/aloha/joints/left_elbow",
    "/aloha/joints/right_elbow",
    "/aloha/joints/left_forearm_roll",
    "/aloha/joints/right_forearm_roll",
    "/aloha/joints/left_wrist_angle",
    "/aloha/joints/right_wrist_angle",
    "/aloha/joints/left_wrist_rotate",
    "/aloha/joints/right_wrist_rotate",
    "/aloha/joints/left_left_finger",
    "/aloha/joints/right_left_finger",
    "/aloha/joints/left_right_finger",
    "/aloha/joints/right_right_finger",
]


def _mapping() -> dict[str, object]:
    return json.loads(MAPPING.read_text(encoding="utf-8"))


def _runtime_records() -> list[dict[str, object]]:
    return [
        {"index": index, "path": path, "name": path.rsplit("/", 1)[-1]}
        for index, path in enumerate(INTERLEAVED_PATHS)
    ]


def test_build_policy_contract_from_real_a17_mapping() -> None:
    contract = build_policy_contract(_mapping())

    assert contract["schema_version"] == "a20-policy-runtime-order-v1"
    assert contract["policy_dimension"] == 14
    assert contract["runtime_dimension"] == 16
    assert [entry["openpi_index"] for entry in contract["policy_entries"]] == list(
        range(14)
    )


def test_build_order_adapter_preserves_interleaved_runtime_order() -> None:
    contract = build_policy_contract(_mapping())
    adapter = build_order_adapter(contract, _runtime_records())

    assert adapter["schema_version"] == "a20-policy-runtime-order-v1"
    assert adapter["runtime_order"] == INTERLEAVED_PATHS
    assert adapter["canonical_to_runtime_indices"] == [
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        1,
        3,
        5,
        7,
        9,
        11,
        13,
        15,
    ]
    assert len(adapter["policy_to_runtime"]) == 14
    assert adapter["policy_to_runtime"][6]["runtime_indices"] == [12, 14]
    assert adapter["policy_to_runtime"][13]["runtime_indices"] == [13, 15]
    assert adapter["mapping_complete"] is True


def test_build_order_adapter_rejects_duplicate_runtime_path() -> None:
    records = _runtime_records()
    records[1]["path"] = records[0]["path"]

    with pytest.raises(ValueError, match="duplicate runtime path"):
        build_order_adapter(build_policy_contract(_mapping()), records)


def test_build_order_adapter_rejects_missing_and_unexpected_runtime_paths() -> None:
    records = _runtime_records()
    records[-1]["path"] = "/aloha/joints/unexpected"

    with pytest.raises(ValueError, match="runtime path inventory mismatch"):
        build_order_adapter(build_policy_contract(_mapping()), records)


def test_build_order_adapter_rejects_duplicate_runtime_index() -> None:
    records = _runtime_records()
    records[1]["index"] = records[0]["index"]

    with pytest.raises(ValueError, match="duplicate runtime index"):
        build_order_adapter(build_policy_contract(_mapping()), records)


def test_build_policy_contract_rejects_missing_openpi_index() -> None:
    mapping = _mapping()
    for record in mapping["joint_records"]:
        canonical_mapping = record.get("canonical_mapping")
        if isinstance(canonical_mapping, dict) and canonical_mapping.get(
            "openpi_index"
        ) == 13:
            canonical_mapping["openpi_index"] = 12
            canonical_mapping["dataset_index"] = 12
    for record in mapping["proposed_canonical_dof_order"]:
        if record.get("openpi_index") == 13:
            record["openpi_index"] = 12
            record["dataset_index"] = 12

    with pytest.raises(ValueError, match="invalid OpenPI index inventory"):
        build_policy_contract(mapping)


def test_build_policy_contract_rejects_wrong_arm_cardinality() -> None:
    mapping = _mapping()
    source = next(
        record
        for record in mapping["joint_records"]
        if record.get("proposed_clean_joint_path")
        == "/aloha/joints/left_left_finger"
    )
    source["canonical_mapping"]["openpi_index"] = 5
    source["canonical_mapping"]["dataset_index"] = 5
    order = next(
        record
        for record in mapping["proposed_canonical_dof_order"]
        if record.get("clean_joint_path") == "/aloha/joints/left_left_finger"
    )
    order["openpi_index"] = 5
    order["dataset_index"] = 5

    with pytest.raises(ValueError, match="invalid OpenPI index 5 cardinality"):
        build_policy_contract(mapping)


@pytest.mark.parametrize("field", ["sign", "offset", "scale"])
def test_build_policy_contract_rejects_non_finite_affine_values(field: str) -> None:
    mapping = deepcopy(_mapping())
    record = next(
        record
        for record in mapping["joint_records"]
        if record.get("canonical_mapping") is not None
    )
    record["canonical_mapping"][field] = float("nan")

    with pytest.raises(ValueError, match=f"non-finite {field}"):
        build_policy_contract(mapping)
