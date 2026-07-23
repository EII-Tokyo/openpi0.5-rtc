from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from aloha_isaac_rebuild.scripts.a20_policy_runtime_order_adapter import build_order_adapter
from aloha_isaac_rebuild.scripts.a20_policy_runtime_order_adapter import build_policy_contract
from aloha_isaac_rebuild.scripts.a20_policy_runtime_order_adapter import policy_to_runtime
from aloha_isaac_rebuild.scripts.a20_policy_runtime_order_adapter import round_trip_check
from aloha_isaac_rebuild.scripts.a20_policy_runtime_order_adapter import runtime_to_policy

ROOT = Path(__file__).resolve().parents[2]
MAPPING = (
    ROOT
    / "aloha_isaac_rebuild/artifacts/validation/a17_clean_articulation_mapping_plan.json"
)
OBSERVED_RUNTIME_PATHS = [
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
    "/aloha/joints/left_right_finger",
    "/aloha/joints/right_left_finger",
    "/aloha/joints/right_right_finger",
]


def _mapping() -> dict[str, object]:
    return json.loads(MAPPING.read_text(encoding="utf-8"))


def _runtime_records() -> list[dict[str, object]]:
    return [
        {"index": index, "path": path, "name": path.rsplit("/", 1)[-1]}
        for index, path in enumerate(OBSERVED_RUNTIME_PATHS)
    ]


def _adapter() -> dict[str, object]:
    return build_order_adapter(build_policy_contract(_mapping()), _runtime_records())


def test_build_policy_contract_from_real_a17_mapping() -> None:
    contract = build_policy_contract(_mapping())

    assert contract["schema_version"] == "a20-policy-runtime-order-v1"
    assert contract["policy_dimension"] == 14
    assert contract["runtime_dimension"] == 16
    assert [entry["openpi_index"] for entry in contract["policy_entries"]] == list(
        range(14)
    )


def test_contract_preserves_source_and_effective_right_finger_transforms() -> None:
    contract = build_policy_contract(_mapping())
    right_finger_paths = (
        "/aloha/joints/left_right_finger",
        "/aloha/joints/right_right_finger",
    )
    dofs_by_path = {
        dof["path"]: dof
        for dof in contract["canonical_dofs"]
        if dof["path"] in right_finger_paths
    }

    assert set(dofs_by_path) == set(right_finger_paths)
    for path in right_finger_paths:
        dof = dofs_by_path[path]
        assert dof["source_transform"] == {
            "sign": -1.0,
            "offset": -0.021,
            "scale": -0.036,
        }
        assert dof["effective_transform"] == {
            "sign": 1.0,
            "offset": 0.021,
            "scale": 0.036,
        }
        assert dof["clean_runtime_mapping_override"] == {
            "sign": 1.0,
            "offset": 0.021,
            "scale": 0.036,
            "unit": "m",
            "rationale": "clean Isaac DOF coordinate already mirrors through its joint frame",
            "source": "A19 authored and A20 runtime positive prismatic limits",
        }
        assert dof["clean_runtime_mapping_override"]["rationale"]
        assert (dof["sign"], dof["offset"], dof["scale"]) == (1.0, 0.021, 0.036)


def test_build_order_adapter_preserves_observed_runtime_order() -> None:
    contract = build_policy_contract(_mapping())
    adapter = build_order_adapter(contract, _runtime_records())

    assert adapter["schema_version"] == "a20-policy-runtime-order-v1"
    assert adapter["runtime_order"] == OBSERVED_RUNTIME_PATHS
    assert adapter["canonical_to_runtime_indices"] == [
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        13,
        1,
        3,
        5,
        7,
        9,
        11,
        14,
        15,
    ]
    assert len(adapter["policy_to_runtime"]) == 14
    assert adapter["policy_to_runtime"][6]["runtime_indices"] == [12, 13]
    assert adapter["policy_to_runtime"][13]["runtime_indices"] == [14, 15]
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


@pytest.mark.parametrize(
    ("source_mapping", "error"),
    [
        (None, "invalid source canonical mapping"),
        ([], "invalid source canonical mapping"),
        (
            {"sign": float("nan"), "offset": -0.021, "scale": -0.036},
            "non-finite source sign",
        ),
        (
            {"sign": -1.0, "offset": float("inf"), "scale": -0.036},
            "non-finite source offset",
        ),
        (
            {"sign": -1.0, "offset": -0.021, "scale": float("nan")},
            "non-finite source scale",
        ),
        (
            {"sign": -1.0, "offset": -0.021, "scale": 0.0},
            "zero source scale",
        ),
    ],
)
def test_build_policy_contract_rejects_invalid_source_canonical_mapping(
    source_mapping: object, error: str
) -> None:
    mapping = deepcopy(_mapping())
    record = next(
        record
        for record in mapping["joint_records"]
        if record.get("proposed_clean_joint_path")
        == "/aloha/joints/left_right_finger"
    )
    record["source_canonical_mapping"] = source_mapping

    with pytest.raises(ValueError, match=error):
        build_policy_contract(mapping)


def test_contract_rejects_unoverridden_source_effective_transform_mismatch() -> None:
    mapping = deepcopy(_mapping())
    record = next(
        record
        for record in mapping["joint_records"]
        if record.get("proposed_clean_joint_path")
        == "/aloha/joints/left_right_finger"
    )
    record["clean_runtime_mapping_override"] = None

    with pytest.raises(
        ValueError, match="unoverridden source/effective transform mismatch"
    ):
        build_policy_contract(mapping)


@pytest.mark.parametrize(
    ("field", "value"),
    [("sign", -1.0), ("offset", -0.021), ("scale", -0.036)],
)
def test_contract_rejects_override_effective_transform_mismatch(
    field: str, value: float
) -> None:
    mapping = deepcopy(_mapping())
    record = next(
        record
        for record in mapping["joint_records"]
        if record.get("proposed_clean_joint_path")
        == "/aloha/joints/left_right_finger"
    )
    record["clean_runtime_mapping_override"][field] = value

    with pytest.raises(ValueError, match="override/effective transform mismatch"):
        build_policy_contract(mapping)


@pytest.mark.parametrize("override", [[], "invalid"])
def test_contract_rejects_invalid_clean_runtime_mapping_override_type(
    override: object,
) -> None:
    mapping = deepcopy(_mapping())
    record = next(
        record
        for record in mapping["joint_records"]
        if record.get("proposed_clean_joint_path")
        == "/aloha/joints/left_right_finger"
    )
    record["clean_runtime_mapping_override"] = override

    with pytest.raises(ValueError, match="invalid clean runtime mapping override"):
        build_policy_contract(mapping)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("rationale", None, "invalid override rationale"),
        ("rationale", "", "invalid override rationale"),
        ("source", None, "invalid override source"),
        ("source", "", "invalid override source"),
    ],
)
def test_contract_rejects_missing_or_empty_override_provenance(
    field: str, value: object, error: str
) -> None:
    mapping = deepcopy(_mapping())
    record = next(
        record
        for record in mapping["joint_records"]
        if record.get("proposed_clean_joint_path")
        == "/aloha/joints/left_right_finger"
    )
    if value is None:
        record["clean_runtime_mapping_override"].pop(field)
    else:
        record["clean_runtime_mapping_override"][field] = value

    with pytest.raises(ValueError, match=error):
        build_policy_contract(mapping)


def test_contract_rejects_override_unit_mismatch() -> None:
    mapping = deepcopy(_mapping())
    record = next(
        record
        for record in mapping["joint_records"]
        if record.get("proposed_clean_joint_path")
        == "/aloha/joints/left_right_finger"
    )
    record["clean_runtime_mapping_override"]["unit"] = "rad"

    with pytest.raises(ValueError, match="override unit mismatch"):
        build_policy_contract(mapping)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("sign", float("nan"), "non-finite override sign"),
        ("offset", float("inf"), "non-finite override offset"),
        ("scale", float("nan"), "non-finite override scale"),
        ("scale", 0.0, "zero override scale"),
        ("sign", "1.0", "invalid override sign"),
    ],
)
def test_contract_rejects_invalid_override_affine_values(
    field: str, value: object, error: str
) -> None:
    mapping = deepcopy(_mapping())
    record = next(
        record
        for record in mapping["joint_records"]
        if record.get("proposed_clean_joint_path")
        == "/aloha/joints/left_right_finger"
    )
    record["clean_runtime_mapping_override"][field] = value

    with pytest.raises(ValueError, match=error):
        build_policy_contract(mapping)


def test_contract_uses_equal_fallback_transforms_for_unoverridden_records() -> None:
    mapping = deepcopy(_mapping())
    arm_path = "/aloha/joints/left_waist"
    left_finger_path = "/aloha/joints/left_left_finger"
    arm_record = next(
        record
        for record in mapping["joint_records"]
        if record.get("proposed_clean_joint_path") == arm_path
    )
    arm_record.pop("source_canonical_mapping")

    contract = build_policy_contract(mapping)
    dofs_by_path = {dof["path"]: dof for dof in contract["canonical_dofs"]}

    for path in (arm_path, left_finger_path):
        dof = dofs_by_path[path]
        assert dof["source_transform"] == dof["effective_transform"]
        assert dof["clean_runtime_mapping_override"] is None


def test_contract_deep_copies_clean_runtime_mapping_override_provenance() -> None:
    mapping = _mapping()
    source_record = next(
        record
        for record in mapping["joint_records"]
        if record.get("proposed_clean_joint_path")
        == "/aloha/joints/left_right_finger"
    )
    source_record["clean_runtime_mapping_override"]["metadata"] = {
        "nested": {"coordinate": "source"}
    }
    contract = build_policy_contract(mapping)
    dof = next(
        dof
        for dof in contract["canonical_dofs"]
        if dof["path"] == "/aloha/joints/left_right_finger"
    )

    dof["clean_runtime_mapping_override"]["metadata"]["nested"]["coordinate"] = (
        "mutated contract provenance"
    )

    assert (
        source_record["clean_runtime_mapping_override"]["metadata"]["nested"][
            "coordinate"
        ]
        == "source"
    )


@pytest.mark.parametrize("gripper_value", [0.0, 0.5, 1.0])
def test_policy_to_runtime_expands_both_grippers(gripper_value: float) -> None:
    policy = [float(index) / 10.0 for index in range(14)]
    policy[6] = gripper_value
    policy[13] = gripper_value

    runtime = policy_to_runtime(policy, _adapter())
    expected_gripper_position = 0.021 + 0.036 * gripper_value

    assert len(runtime) == 16
    assert runtime[12] == pytest.approx(expected_gripper_position)
    assert runtime[13] == pytest.approx(expected_gripper_position)
    assert runtime[14] == pytest.approx(expected_gripper_position)
    assert runtime[15] == pytest.approx(expected_gripper_position)
    assert runtime[0] == pytest.approx(policy[0])
    assert runtime[1] == pytest.approx(policy[7])
    assert runtime_to_policy(runtime, _adapter()) == pytest.approx(policy)


def test_policy_to_runtime_rejects_wrong_length_and_non_finite_values() -> None:
    with pytest.raises(ValueError, match="policy vector length"):
        policy_to_runtime([0.0] * 13, _adapter())
    policy = [0.0] * 14
    policy[3] = float("inf")
    with pytest.raises(ValueError, match="non-finite policy value"):
        policy_to_runtime(policy, _adapter())


def test_runtime_to_policy_round_trips_arms_and_grippers() -> None:
    policy = [
        -0.6,
        -0.4,
        -0.2,
        0.0,
        0.2,
        0.4,
        0.25,
        0.6,
        0.4,
        0.2,
        0.0,
        -0.2,
        -0.4,
        0.75,
    ]

    recovered = runtime_to_policy(policy_to_runtime(policy, _adapter()), _adapter())

    assert recovered == pytest.approx(policy)


def test_runtime_to_policy_rejects_inconsistent_gripper_readback() -> None:
    adapter = _adapter()
    runtime = policy_to_runtime([0.0] * 14, adapter)
    runtime[13] -= 0.001

    with pytest.raises(ValueError, match="inconsistent gripper readback"):
        runtime_to_policy(runtime, adapter)


def test_runtime_to_policy_rejects_wrong_length_and_non_finite_values() -> None:
    with pytest.raises(ValueError, match="runtime vector length"):
        runtime_to_policy([0.0] * 15, _adapter())
    runtime = [0.0] * 16
    runtime[2] = float("nan")
    with pytest.raises(ValueError, match="non-finite runtime value"):
        runtime_to_policy(runtime, _adapter())


def test_round_trip_check_covers_gripper_calibration_range() -> None:
    result = round_trip_check(_adapter())

    assert result["status"] == "PASS"
    assert result["sample_count"] == 3
    assert result["gripper_values"] == [0.0, 0.5, 1.0]
    assert result["max_abs_error"] <= 1e-12
