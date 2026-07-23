from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

import pytest
import yaml

from aloha_isaac_rebuild.scripts import audit_a21_policy_target_limit_preflight as audit
from aloha_isaac_rebuild.scripts.audit_a21_policy_target_limit_preflight import ARM_DELTA_RAD
from aloha_isaac_rebuild.scripts.audit_a21_policy_target_limit_preflight import FAIL_STATUS
from aloha_isaac_rebuild.scripts.audit_a21_policy_target_limit_preflight import PASS_STATUS
from aloha_isaac_rebuild.scripts.audit_a21_policy_target_limit_preflight import SCHEMA_VERSION
from aloha_isaac_rebuild.scripts.audit_a21_policy_target_limit_preflight import build_reviewed_policy_samples
from aloha_isaac_rebuild.scripts.audit_a21_policy_target_limit_preflight import evaluate_policy_samples
from aloha_isaac_rebuild.scripts.audit_a21_policy_target_limit_preflight import evaluate_preflight
from aloha_isaac_rebuild.scripts.audit_a21_policy_target_limit_preflight import runtime_bounds

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "aloha_isaac_rebuild/scripts/audit_a21_policy_target_limit_preflight.py"

CANONICAL_PATHS = [
    "/aloha/joints/left_waist",
    "/aloha/joints/left_shoulder",
    "/aloha/joints/left_elbow",
    "/aloha/joints/left_forearm_roll",
    "/aloha/joints/left_wrist_angle",
    "/aloha/joints/left_wrist_rotate",
    "/aloha/joints/left_left_finger",
    "/aloha/joints/left_right_finger",
    "/aloha/joints/right_waist",
    "/aloha/joints/right_shoulder",
    "/aloha/joints/right_elbow",
    "/aloha/joints/right_forearm_roll",
    "/aloha/joints/right_wrist_angle",
    "/aloha/joints/right_wrist_rotate",
    "/aloha/joints/right_left_finger",
    "/aloha/joints/right_right_finger",
]
RUNTIME_PATHS = [CANONICAL_PATHS[index] for index in (0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 7, 14, 15)]
POLICY_CANONICAL_GROUPS = [[index] for index in range(6)] + [[6, 7]] + [[index] for index in range(8, 14)] + [[14, 15]]
MIRRORED_FINGER_PATHS = {
    "/aloha/joints/left_right_finger",
    "/aloha/joints/right_right_finger",
}
LAYER_INPUTS = {
    "config": {"path": "/evidence/config.yaml", "sha256": "a" * 64},
    "mapping": {"path": "/evidence/mapping.json", "sha256": "b" * 64},
    "stage": {
        "path": "/evidence/stage.usda",
        "pre_sha256": "c" * 64,
        "post_sha256": "c" * 64,
        "consistent_during_audit": True,
    },
}


def _transform(path: str, *, corrected: bool) -> dict[str, object]:
    if "finger" not in path:
        return {"path": path, "sign": 1.0, "offset": 0.0, "scale": 1.0}
    if path in MIRRORED_FINGER_PATHS and not corrected:
        return {"path": path, "sign": -1.0, "offset": -0.021, "scale": -0.036}
    return {"path": path, "sign": 1.0, "offset": 0.021, "scale": 0.036}


def _adapter(*, corrected: bool = True) -> dict[str, object]:
    runtime_index_by_path = {path: runtime_index for runtime_index, path in enumerate(RUNTIME_PATHS)}
    entries = []
    for policy_index, canonical_indices in enumerate(POLICY_CANONICAL_GROUPS):
        paths = [CANONICAL_PATHS[index] for index in canonical_indices]
        entries.append(
            {
                "openpi_index": policy_index,
                "runtime_indices": [runtime_index_by_path[path] for path in paths],
                "transforms": [_transform(path, corrected=corrected) for path in paths],
            }
        )

    return {
        "schema_version": "a20-policy-runtime-order-v1",
        "policy_dimension": 14,
        "runtime_dimension": 16,
        "canonical_order": deepcopy(CANONICAL_PATHS),
        "runtime_order": deepcopy(RUNTIME_PATHS),
        "canonical_to_runtime_indices": [runtime_index_by_path[path] for path in CANONICAL_PATHS],
        "runtime_to_canonical_indices": [CANONICAL_PATHS.index(path) for path in RUNTIME_PATHS],
        "policy_to_runtime": entries,
        "mapping_complete": True,
    }


def _canonical_dofs(*, corrected: bool = True) -> list[dict[str, object]]:
    canonical_dofs = []
    for canonical_index, path in enumerate(CANONICAL_PATHS):
        effective = _transform(path, corrected=corrected)
        if path in MIRRORED_FINGER_PATHS:
            source = {
                "sign": -1.0,
                "offset": -0.021,
                "scale": -0.036,
            }
            override = (
                {
                    "sign": 1.0,
                    "offset": 0.021,
                    "scale": 0.036,
                    "unit": "m",
                    "rationale": "clean coordinate is positive",
                    "source": "synthetic A20 evidence",
                }
                if corrected
                else None
            )
        else:
            source = {field: effective[field] for field in ("sign", "offset", "scale")}
            override = None
        canonical_dofs.append(
            {
                "canonical_index": canonical_index,
                "path": path,
                "openpi_index": next(
                    policy_index
                    for policy_index, indices in enumerate(POLICY_CANONICAL_GROUPS)
                    if canonical_index in indices
                ),
                "source_transform": source,
                "effective_transform": {field: effective[field] for field in ("sign", "offset", "scale")},
                "clean_runtime_mapping_override": override,
                "unit": "m" if "finger" in path else "rad",
            }
        )
    return canonical_dofs


def _policy_contract(*, corrected: bool = True) -> dict[str, object]:
    canonical_dofs = _canonical_dofs(corrected=corrected)
    policy_entries = []
    for policy_index, canonical_indices in enumerate(POLICY_CANONICAL_GROUPS):
        paths = [CANONICAL_PATHS[index] for index in canonical_indices]
        policy_entries.append(
            {
                "openpi_index": policy_index,
                "dataset_index": policy_index,
                "canonical_indices": deepcopy(canonical_indices),
                "canonical_paths": paths,
                "transforms": [_transform(path, corrected=corrected) for path in paths],
            }
        )
    return {
        "schema_version": "a20-policy-runtime-order-v1",
        "policy_dimension": 14,
        "runtime_dimension": 16,
        "canonical_order": deepcopy(CANONICAL_PATHS),
        "canonical_dofs": canonical_dofs,
        "policy_entries": policy_entries,
    }


def _runtime_records() -> list[dict[str, object]]:
    records = []
    for index, path in enumerate(RUNTIME_PATHS):
        is_finger = "finger" in path
        records.append(
            {
                "index": index,
                "path": path,
                "name": path.rsplit("/", 1)[-1],
                "joint_type": ("PhysicsPrismaticJoint" if is_finger else "PhysicsRevoluteJoint"),
                "lower_limit": 0.018 if is_finger else -180.0,
                "upper_limit": 0.058 if is_finger else 180.0,
            }
        )
    return records


def _evaluate(
    adapter: dict[str, object] | None = None,
    records: list[dict[str, object]] | None = None,
    samples: list[dict[str, object]] | None = None,
    canonical_dofs: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return evaluate_policy_samples(
        adapter if adapter is not None else _adapter(),
        records if records is not None else _runtime_records(),
        samples if samples is not None else build_reviewed_policy_samples(),
        canonical_dofs=(canonical_dofs if canonical_dofs is not None else _canonical_dofs()),
    )


def test_negative_right_finger_expansion_fails_positive_runtime_limits() -> None:
    result = evaluate_policy_samples(
        _adapter(corrected=False),
        _runtime_records(),
        build_reviewed_policy_samples(),
        canonical_dofs=_canonical_dofs(corrected=False),
    )

    expected_indices = {RUNTIME_PATHS.index(path) for path in MIRRORED_FINGER_PATHS}
    assert result["ok"] is False
    assert {mismatch["runtime_index"] for mismatch in result["mismatches"]} == (expected_indices)
    assert {mismatch["code"] for mismatch in result["mismatches"]} == {"target_outside_runtime_limits"}


def test_corrected_effective_mapping_passes_all_reviewed_samples() -> None:
    result = evaluate_policy_samples(
        _adapter(),
        _runtime_records(),
        build_reviewed_policy_samples(),
        canonical_dofs=_canonical_dofs(),
    )

    assert result["ok"] is True
    assert result["sample_count"] == 4
    assert result["mismatches"] == []
    assert result["errors"] == []
    assert result["max_arm_delta_rad"] == ARM_DELTA_RAD == math.radians(0.25)


def _layer1(*, corrected: bool = True) -> dict[str, object]:
    return {
        "status": "PASS_A20_USD_DOF_METADATA",
        "ok": True,
        "policy_contract": _policy_contract(corrected=corrected),
        "mismatches": [],
        "errors": [],
        "inputs": deepcopy(LAYER_INPUTS),
        "physics_stepped": False,
        "actions_applied": False,
        "targets_written": False,
        "stage_saved": False,
    }


def _layer2() -> dict[str, object]:
    records = _runtime_records()
    runs = [
        {
            "invocation_id": f"run-{index}",
            "records": deepcopy(records),
            "physics_stepped": False,
            "actions_applied": False,
            "targets_written": False,
            "stage_saved": False,
            "inputs": {
                "config": deepcopy(LAYER_INPUTS["config"]),
                "mapping": deepcopy(LAYER_INPUTS["mapping"]),
                "stage": {
                    "path": LAYER_INPUTS["stage"]["path"],
                    "sha256": LAYER_INPUTS["stage"]["pre_sha256"],
                },
            },
        }
        for index in range(3)
    ]
    return {
        "status": "PASS_A20_RUNTIME_ARTICULATION_DISCOVERY_NO_STEP",
        "ok": True,
        "mismatches": [],
        "errors": [],
        "physics_stepped": False,
        "actions_applied": False,
        "targets_written": False,
        "stage_saved": False,
        "run_count": 3,
        "runs": runs,
        "order_adapter": _adapter(),
    }


def test_reviewed_samples_are_exact_and_fresh() -> None:
    samples = build_reviewed_policy_samples()

    assert [sample["label"] for sample in samples] == [
        "grippers_closed",
        "grippers_mid",
        "grippers_open",
        "signed_arm_micro_targets",
    ]
    for sample, gripper_value in zip(samples[:3], (0.0, 0.5, 1.0), strict=True):
        values = sample["policy_values"]
        assert len(values) == 14
        assert values[6] == values[13] == gripper_value
        assert all(value == 0.0 for index, value in enumerate(values) if index not in {6, 13})
    signed = samples[3]["policy_values"]
    for index, value in enumerate(signed):
        expected = 0.5 if index in {6, 13} else (ARM_DELTA_RAD if index % 2 == 0 else -ARM_DELTA_RAD)
        assert value == expected

    samples[0]["policy_values"][0] = 99.0
    assert build_reviewed_policy_samples()[0]["policy_values"][0] == 0.0


def test_runtime_bounds_converts_revolute_degrees_to_radians() -> None:
    assert runtime_bounds(
        {
            "joint_type": "PhysicsRevoluteJoint",
            "lower_limit": -180.0,
            "upper_limit": 90.0,
        }
    ) == pytest.approx((-math.pi, math.pi / 2.0))


def test_runtime_bounds_preserves_prismatic_metres() -> None:
    assert runtime_bounds(
        {
            "joint_type": "PhysicsPrismaticJoint",
            "lower_limit": 0.018,
            "upper_limit": 0.058,
        }
    ) == (0.018, 0.058)


@pytest.mark.parametrize(
    "record",
    [
        None,
        {
            "joint_type": "PhysicsRevoluteJoint",
            "lower_limit": True,
            "upper_limit": 1.0,
        },
        {
            "joint_type": "PhysicsRevoluteJoint",
            "lower_limit": float("nan"),
            "upper_limit": 1.0,
        },
        {
            "joint_type": "PhysicsPrismaticJoint",
            "lower_limit": 1.0,
            "upper_limit": 1.0,
        },
        {
            "joint_type": "PhysicsPrismaticJoint",
            "lower_limit": 2.0,
            "upper_limit": 1.0,
        },
        {"joint_type": "unsupported", "lower_limit": 0.0, "upper_limit": 1.0},
    ],
)
def test_runtime_bounds_rejects_invalid_records(record: object) -> None:
    with pytest.raises(
        ValueError,
        match="runtime record|finite number|less than|unsupported",
    ):
        runtime_bounds(record)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda adapter: adapter.update(schema_version="stale"),
        lambda adapter: adapter.update(policy_dimension=13),
        lambda adapter: adapter.update(runtime_dimension=15),
        lambda adapter: adapter["policy_to_runtime"].pop(),
    ],
)
def test_evaluation_rejects_wrong_adapter_contract(mutate) -> None:
    adapter = _adapter()
    mutate(adapter)

    result = _evaluate(adapter=adapter)

    assert result["ok"] is False
    assert result["errors"][0]["code"] == "invalid_preflight_input"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda records: records[1].update(index=records[0]["index"]),
        lambda records: records[15].update(index=16),
        lambda records: records[1].update(path=records[0]["path"]),
        lambda records: records[1].update(path=""),
        lambda records: records[1].update(index=True),
    ],
)
def test_evaluation_rejects_bad_runtime_index_or_path_inventory(mutate) -> None:
    records = _runtime_records()
    mutate(records)

    result = _evaluate(records=records)

    assert result["ok"] is False
    assert result["errors"][0]["code"] == "invalid_preflight_input"


@pytest.mark.parametrize("bad_value", [True, float("nan"), float("inf")])
def test_evaluation_rejects_bool_or_non_finite_policy_input(
    bad_value: object,
) -> None:
    samples = build_reviewed_policy_samples()
    samples[0]["policy_values"][0] = bad_value

    result = _evaluate(samples=samples)

    assert result["ok"] is False
    assert result["errors"][0]["code"] == "invalid_preflight_input"


def test_evaluation_rejects_duplicate_sample_labels() -> None:
    samples = build_reviewed_policy_samples()
    samples[1]["label"] = samples[0]["label"]

    result = _evaluate(samples=samples)

    assert result["ok"] is False
    assert "duplicate sample label" in result["errors"][0]["message"]


def test_non_finite_expanded_target_is_a_deterministic_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        audit,
        "policy_to_runtime",
        lambda *_: [float("nan")] + [0.0] * 15,
    )

    result = _evaluate()

    assert result["ok"] is False
    assert result["mismatches"] == []
    assert {error["code"] for error in result["errors"]} == {"policy_conversion_error"}


def test_conversion_exceptions_are_errors_not_crashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_conversion(*_args, **_kwargs):
        raise ValueError("synthetic conversion failure")

    monkeypatch.setattr(audit, "policy_to_runtime", fail_conversion)
    result = _evaluate()

    assert result["ok"] is False
    assert len(result["errors"]) == 4
    assert all(error["code"] == "policy_conversion_error" for error in result["errors"])


def test_inverse_round_trip_disagreement_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        audit,
        "runtime_to_policy",
        lambda *_args, **_kwargs: [0.0] * 14,
    )

    result = _evaluate()

    assert result["ok"] is False
    assert any(error["code"] == "round_trip_error" for error in result["errors"])
    assert any("disagreement" in error["message"] for error in result["errors"])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda dofs: dofs[7].update(clean_runtime_mapping_override=None),
        lambda dofs: dofs[7]["effective_transform"].update(scale=-0.036),
        lambda dofs: dofs[7]["clean_runtime_mapping_override"].update(scale=0.04),
        lambda dofs: dofs[7]["source_transform"].update(scale=0.036),
        lambda dofs: dofs[7]["clean_runtime_mapping_override"].update(rationale=""),
    ],
)
def test_missing_malformed_or_inconsistent_right_finger_provenance_fails(
    mutate,
) -> None:
    canonical_dofs = _canonical_dofs()
    mutate(canonical_dofs)

    result = _evaluate(canonical_dofs=canonical_dofs)

    assert result["ok"] is False
    assert any(error["code"] == "invalid_right_finger_override_provenance" for error in result["errors"])


def test_direct_api_keeps_legacy_embedded_provenance_compatibility() -> None:
    adapter = _adapter()
    adapter["canonical_dofs"] = _canonical_dofs()

    result = evaluate_policy_samples(
        adapter,
        _runtime_records(),
        build_reviewed_policy_samples(),
    )

    assert result["ok"] is True
    assert result["errors"] == []


def test_adapter_transform_path_must_match_its_raw_runtime_record() -> None:
    adapter = _adapter()
    adapter["policy_to_runtime"][0]["transforms"][0]["path"] = CANONICAL_PATHS[8]

    result = _evaluate(adapter=adapter)

    assert result["ok"] is False
    assert any(error["code"] == "invalid_right_finger_override_provenance" for error in result["errors"])


def _set_bool_adapter_runtime_index(adapter: dict[str, object]) -> None:
    adapter["policy_to_runtime"][0]["runtime_indices"][0] = True


@pytest.mark.parametrize(
    "mutate",
    [
        lambda adapter: adapter["policy_to_runtime"][0].update(openpi_index=True),
        _set_bool_adapter_runtime_index,
        lambda adapter: adapter["policy_to_runtime"][1]["runtime_indices"].__setitem__(
            0, adapter["policy_to_runtime"][0]["runtime_indices"][0]
        ),
    ],
)
def test_direct_api_rejects_bool_or_duplicate_adapter_indices(mutate) -> None:
    adapter = _adapter()
    mutate(adapter)

    result = _evaluate(adapter=adapter)

    assert result["ok"] is False
    assert any(error["code"] == "invalid_right_finger_override_provenance" for error in result["errors"])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda dofs: dofs[0].update(canonical_index=True),
        lambda dofs: dofs[1].update(openpi_index=True),
        lambda dofs: dofs[15].update(openpi_index=12),
    ],
)
def test_direct_api_rejects_bool_or_incomplete_canonical_indices(mutate) -> None:
    canonical_dofs = _canonical_dofs()
    mutate(canonical_dofs)

    result = _evaluate(canonical_dofs=canonical_dofs)

    assert result["ok"] is False
    assert any(error["code"] == "invalid_right_finger_override_provenance" for error in result["errors"])


def test_exact_runtime_limit_is_accepted() -> None:
    records = _runtime_records()
    for record in records:
        if record["joint_type"] == "PhysicsPrismaticJoint":
            record["upper_limit"] = 0.057

    result = _evaluate(records=records)

    assert result["ok"] is True
    assert result["mismatches"] == []


def test_target_beyond_explicit_limit_tolerance_is_rejected() -> None:
    records = _runtime_records()
    target_path = "/aloha/joints/right_left_finger"
    target_record = next(record for record in records if record["path"] == target_path)
    target_record["upper_limit"] = 0.057 - 1.1e-9

    result = _evaluate(records=records)

    mismatch = next(
        item for item in result["mismatches"] if item["path"] == target_path and item["label"] == "grippers_open"
    )
    assert mismatch["runtime_index"] == RUNTIME_PATHS.index(target_path)
    assert mismatch["policy_index"] == 13
    assert mismatch["code"] == "target_outside_runtime_limits"


def test_evaluation_does_not_mutate_inputs() -> None:
    adapter = _adapter()
    records = _runtime_records()
    samples = build_reviewed_policy_samples()
    canonical_dofs = _canonical_dofs()
    before = deepcopy((adapter, records, samples, canonical_dofs))

    evaluate_policy_samples(
        adapter,
        records,
        samples,
        canonical_dofs=canonical_dofs,
    )

    assert (adapter, records, samples, canonical_dofs) == before


def test_preflight_pass_has_exact_structured_contract() -> None:
    inputs = {
        "layer1": {"path": "/evidence/layer1.json", "sha256": "a" * 64},
        "layer2": {"path": "/evidence/layer2.json", "sha256": "b" * 64},
    }
    result = evaluate_preflight(_layer1(), _layer2(), inputs=inputs)

    assert set(result) == {
        "schema_version",
        "ok",
        "status",
        "inputs",
        "sample_count",
        "samples",
        "mismatches",
        "errors",
        "physics_stepped",
        "actions_applied",
        "targets_written",
        "targets_restored",
        "stage_saved",
    }
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["status"] == PASS_STATUS
    assert result["ok"] is True
    assert result["inputs"] == inputs
    assert result["sample_count"] == 4
    assert result["mismatches"] == []
    assert result["errors"] == []
    for flag in (
        "physics_stepped",
        "actions_applied",
        "targets_written",
        "targets_restored",
        "stage_saved",
    ):
        assert result[flag] is False


def test_preflight_reads_canonical_provenance_from_layer1_policy_contract() -> None:
    layer1 = _layer1()
    layer2 = _layer2()
    adapter = layer2["order_adapter"]

    result = evaluate_preflight(layer1, layer2)

    assert "canonical_dofs" not in adapter
    assert result["status"] == PASS_STATUS
    assert result["ok"] is True
    assert result["errors"] == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda layer: layer.pop("policy_contract"),
        lambda layer: layer.update(policy_contract=[]),
        lambda layer: layer["policy_contract"].pop("canonical_dofs"),
        lambda layer: layer["policy_contract"]["canonical_dofs"].pop(),
        lambda layer: layer["policy_contract"]["canonical_dofs"][0].update(canonical_index=True),
        lambda layer: layer["policy_contract"]["canonical_dofs"][0].update(openpi_index=True),
        lambda layer: layer["policy_contract"]["canonical_dofs"][13].update(openpi_index=11),
    ],
)
def test_preflight_rejects_missing_or_malformed_layer1_canonical_contract(
    mutate,
) -> None:
    layer1 = _layer1()
    mutate(layer1)

    result = evaluate_preflight(layer1, _layer2())

    assert result["status"] == FAIL_STATUS
    assert result["ok"] is False
    assert any(error["code"] == "invalid_layer1_evidence" for error in result["errors"])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda layer: layer["policy_contract"]["canonical_dofs"][0].update(unit="cm"),
        lambda layer: layer["policy_contract"]["canonical_dofs"][7]["clean_runtime_mapping_override"].update(unit="cm"),
    ],
)
def test_preflight_rejects_wrong_canonical_or_override_units(mutate) -> None:
    layer1 = _layer1()
    mutate(layer1)

    result = evaluate_preflight(layer1, _layer2())

    assert result["status"] == FAIL_STATUS
    assert result["ok"] is False


def test_right_finger_override_provenance_cannot_be_swapped_to_left_finger() -> None:
    layer1 = _layer1()
    records = layer1["policy_contract"]["canonical_dofs"]
    left = next(record for record in records if record["path"] == "/aloha/joints/left_left_finger")
    right = next(record for record in records if record["path"] == "/aloha/joints/left_right_finger")
    for field in (
        "source_transform",
        "effective_transform",
        "clean_runtime_mapping_override",
    ):
        left[field], right[field] = right[field], left[field]

    result = evaluate_preflight(layer1, _layer2())

    assert result["status"] == FAIL_STATUS
    assert result["ok"] is False


@pytest.mark.parametrize(
    ("target", "mutate"),
    [
        ("layer1", lambda layer: layer.pop("inputs")),
        (
            "layer1",
            lambda layer: layer["inputs"]["config"].update(path="relative.yaml"),
        ),
        (
            "layer1",
            lambda layer: layer["inputs"]["mapping"].update(sha256="BAD"),
        ),
        (
            "layer1",
            lambda layer: layer["inputs"]["stage"].update(post_sha256="d" * 64),
        ),
        (
            "layer1",
            lambda layer: layer["inputs"]["stage"].update(consistent_during_audit=False),
        ),
        ("layer2", lambda layer: layer["runs"][0].pop("inputs")),
        (
            "layer2",
            lambda layer: layer["runs"][1]["inputs"]["config"].update(sha256="d" * 64),
        ),
        (
            "layer2",
            lambda layer: layer["runs"][2]["inputs"]["stage"].update(path="/evidence/other.usda"),
        ),
        (
            "layer2",
            lambda layer: layer["runs"][2]["inputs"].update(mapping=[]),
        ),
    ],
)
def test_preflight_binds_all_run_inputs_to_layer1_provenance(
    target: str,
    mutate,
) -> None:
    layer1 = _layer1()
    layer2 = _layer2()
    mutate(layer1 if target == "layer1" else layer2)

    result = evaluate_preflight(layer1, layer2)

    assert result["status"] == FAIL_STATUS
    assert result["ok"] is False
    assert result["errors"]


@pytest.mark.parametrize(
    ("layer1", "layer2"),
    [
        (None, _layer2()),
        ([], _layer2()),
        ("bad", _layer2()),
        (_layer1(), None),
        (_layer1(), 3),
        (_layer1(), "bad"),
    ],
)
def test_preflight_fails_closed_for_scalar_or_non_dict_layers(layer1: object, layer2: object) -> None:
    result = evaluate_preflight(layer1, layer2)

    assert result["status"] == FAIL_STATUS
    assert result["ok"] is False
    assert result["errors"]


@pytest.mark.parametrize(
    ("target", "mutate"),
    [
        ("layer1", lambda layer: layer.update(status="FAIL_A20_USD_DOF_METADATA")),
        ("layer2", lambda layer: layer.update(status="FAIL_A20_RUNTIME")),
        ("layer1", lambda layer: layer.update(ok=False)),
        ("layer2", lambda layer: layer.update(ok=False)),
        ("layer1", lambda layer: layer["errors"].append({"code": "stale"})),
        ("layer2", lambda layer: layer["mismatches"].append({"field": "path"})),
        ("layer1", lambda layer: layer.update(physics_stepped=True)),
        ("layer2", lambda layer: layer.update(actions_applied=True)),
        ("layer1", lambda layer: layer.pop("targets_written")),
        ("layer2", lambda layer: layer.update(stage_saved=None)),
    ],
)
def test_preflight_requires_exact_a20_pass_and_false_safety_flags(target: str, mutate) -> None:
    layer1 = _layer1()
    layer2 = _layer2()
    mutate(layer1 if target == "layer1" else layer2)

    result = evaluate_preflight(layer1, layer2)

    assert result["status"] == FAIL_STATUS
    assert result["ok"] is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda layer: layer.pop("order_adapter"),
        lambda layer: layer.update(run_count=2),
        lambda layer: layer["runs"].pop(),
        lambda layer: layer["runs"][1]["records"][0].update(upper_limit=179.0),
        lambda layer: layer["runs"][2].update(targets_written=True),
        lambda layer: layer["runs"][0].pop("records"),
    ],
)
def test_preflight_requires_three_deterministic_no_step_runtime_runs(mutate) -> None:
    layer2 = _layer2()
    mutate(layer2)

    result = evaluate_preflight(_layer1(), layer2)

    assert result["status"] == FAIL_STATUS
    assert result["ok"] is False
    assert result["errors"]


def test_preflight_does_not_mutate_layers_or_inputs() -> None:
    layer1 = _layer1()
    layer2 = _layer2()
    inputs = {"config": {"path": "/config.yaml", "sha256": "c" * 64}}
    before = deepcopy((layer1, layer2, inputs))

    evaluate_preflight(layer1, layer2, inputs=inputs)

    assert (layer1, layer2, inputs) == before


def _write_cli_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    layer1_path = tmp_path / "evidence/layer1.json"
    layer2_path = tmp_path / "evidence/layer2.json"
    output_path = tmp_path / "out/a21.json"
    config_path = tmp_path / "config.yaml"
    layer1_path.parent.mkdir()
    layer1_path.write_text(json.dumps(_layer1()), encoding="utf-8")
    layer2_path.write_text(json.dumps(_layer2()), encoding="utf-8")
    config_path.write_text(
        yaml.safe_dump(
            {
                "outputs": {
                    "a20_usd_dof_metadata_json": str(layer1_path.relative_to(tmp_path)),
                    "a20_runtime_articulation_discovery_json": str(layer2_path.relative_to(tmp_path)),
                    "a21_policy_target_limit_preflight_json": str(output_path.relative_to(tmp_path)),
                }
            }
        ),
        encoding="utf-8",
    )
    return config_path, layer1_path, layer2_path, output_path


def test_cli_hashes_absolute_inputs_writes_pass_and_exits_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path, layer1_path, layer2_path, output_path = _write_cli_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(MODULE), "--config", str(config_path)])

    assert audit.main() == 0
    written = json.loads(output_path.read_text(encoding="utf-8"))
    printed = json.loads(capsys.readouterr().out)
    assert written == printed
    assert written["status"] == PASS_STATUS
    for name, path in (
        ("config", config_path),
        ("layer1", layer1_path),
        ("layer2", layer2_path),
    ):
        assert written["inputs"][name] == {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }


def test_cli_artifact_hashes_do_not_replace_a20_cross_layer_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, layer1_path, _layer2_path, output_path = _write_cli_fixture(tmp_path)
    layer1 = json.loads(layer1_path.read_text(encoding="utf-8"))
    layer1.pop("inputs")
    layer1_path.write_text(json.dumps(layer1), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(MODULE), "--config", str(config_path)])

    assert audit.main() == 1
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["inputs"]["layer1"]["sha256"] == hashlib.sha256(layer1_path.read_bytes()).hexdigest()
    assert written["status"] == FAIL_STATUS
    assert any(error["code"] == "invalid_layer1_evidence" for error in written["errors"])


def test_cli_writes_parseable_fail_for_missing_or_malformed_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path, layer1_path, _layer2_path, output_path = _write_cli_fixture(tmp_path)
    layer1_path.write_text("{bad json", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(MODULE), "--config", str(config_path)])

    assert audit.main() == 1
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == written
    assert written["status"] == FAIL_STATUS
    assert written["ok"] is False
    assert written["errors"]


def test_cli_resolves_output_and_fails_when_input_keys_are_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "nested/a21.json"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"outputs": {"a21_policy_target_limit_preflight_json": str(output_path.relative_to(tmp_path))}}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(MODULE), "--config", str(config_path)])

    assert audit.main() == 1
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["status"] == FAIL_STATUS
    assert written["ok"] is False


def test_module_is_pure_and_has_only_atomic_artifact_writes() -> None:
    source = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_import_roots = {"isaacsim", "omni", "pxr"}
    forbidden_calls = {
        "apply_action",
        "save",
        "Save",
        "set_joint_efforts",
        "set_joint_position_targets",
        "set_joint_positions",
        "set_joint_velocities",
        "set_joint_velocity_targets",
        "step",
        "update_simulation",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".", 1)[0] not in forbidden_import_roots for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".", 1)[0] not in forbidden_import_roots
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_calls
            elif isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_calls

    assert "os.replace(" in source
    assert "os.fsync(" in source
