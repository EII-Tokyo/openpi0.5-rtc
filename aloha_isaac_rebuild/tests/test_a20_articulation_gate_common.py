from __future__ import annotations

from copy import deepcopy

import pytest

from aloha_isaac_rebuild.scripts import a20_articulation_gate_common as common
from aloha_isaac_rebuild.scripts.a20_articulation_gate_common import compare_dof_records


def _record(index: int, name: str) -> dict[str, object]:
    return {
        "index": index,
        "path": f"/aloha/joints/{name}",
        "name": name,
        "joint_type": "revolute",
        "axis": "Z",
        "lower_limit": -90.0,
        "upper_limit": 90.0,
        "body0": "/aloha/links/parent",
        "body1": f"/aloha/links/{name}",
    }


def test_compare_dof_records_accepts_exact_ordered_match() -> None:
    expected = [_record(0, "waist"), _record(1, "shoulder")]

    result = compare_dof_records(expected, deepcopy(expected))

    assert result == {
        "ok": True,
        "expected_count": 2,
        "observed_count": 2,
        "mismatches": [],
    }


def test_compare_dof_records_reports_order_change_stably() -> None:
    expected = [_record(0, "waist"), _record(1, "shoulder")]
    observed = [_record(0, "shoulder"), _record(1, "waist")]

    result = compare_dof_records(expected, observed)

    assert result["ok"] is False
    assert result["mismatches"][0] == {
        "field": "path",
        "index": 0,
        "expected": "/aloha/joints/waist",
        "observed": "/aloha/joints/shoulder",
    }


def test_compare_dof_records_reports_missing_and_unexpected_paths() -> None:
    expected = [_record(0, "waist"), _record(1, "shoulder")]
    observed = [_record(0, "waist"), _record(1, "elbow")]

    result = compare_dof_records(expected, observed)

    assert {
        "field": "missing",
        "index": None,
        "expected": "/aloha/joints/shoulder",
        "observed": None,
    } in result["mismatches"]
    assert {
        "field": "unexpected",
        "index": None,
        "expected": None,
        "observed": "/aloha/joints/elbow",
    } in result["mismatches"]


@pytest.mark.parametrize(
    ("field", "error_code"),
    [("path", "duplicate_path"), ("name", "duplicate_name")],
)
def test_validate_dof_records_rejects_duplicates(field: str, error_code: str) -> None:
    records = [_record(0, "waist"), _record(1, "shoulder")]
    records[1][field] = records[0][field]

    result = common.validate_dof_records(records)

    assert result["ok"] is False
    assert result["errors"][0] == {
        "code": error_code,
        "field": field,
        "value": records[0][field],
        "indices": [0, 1],
    }


def test_validate_dof_records_rejects_non_increasing_limits() -> None:
    record = _record(0, "waist")
    record["lower_limit"] = 1.0
    record["upper_limit"] = 1.0

    result = common.validate_dof_records([record])

    assert result["errors"] == [
        {
            "code": "invalid_limit_order",
            "index": 0,
            "lower_limit": 1.0,
            "upper_limit": 1.0,
        }
    ]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_validate_dof_records_rejects_non_finite_limits(value: float) -> None:
    record = _record(0, "waist")
    record["lower_limit"] = value

    result = common.validate_dof_records([record])

    assert result["ok"] is False
    assert result["errors"][0]["code"] == "non_finite_limit"
    assert result["errors"][0]["field"] == "lower_limit"
    assert result["errors"][0]["index"] == 0


@pytest.mark.parametrize(
    "prohibited_flag",
    ["physics_stepped", "actions_applied", "targets_written", "stage_saved"],
)
def test_validate_safety_flags_rejects_prohibited_true_flag(prohibited_flag: str) -> None:
    payload = {
        "physics_stepped": False,
        "actions_applied": False,
        "targets_written": False,
        "stage_saved": False,
    }
    payload[prohibited_flag] = True

    result = common.validate_safety_flags(payload)

    assert result == {
        "ok": False,
        "errors": [
            {
                "code": "prohibited_safety_flag",
                "field": prohibited_flag,
                "observed": True,
            }
        ],
    }
