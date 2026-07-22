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
    assert result["mismatches"] == [
        {
            "field": "path",
            "index": 0,
            "expected": "/aloha/joints/waist",
            "observed": "/aloha/joints/shoulder",
        },
        {
            "field": "name",
            "index": 0,
            "expected": "waist",
            "observed": "shoulder",
        },
        {
            "field": "body1",
            "index": 0,
            "expected": "/aloha/links/waist",
            "observed": "/aloha/links/shoulder",
        },
        {
            "field": "path",
            "index": 1,
            "expected": "/aloha/joints/shoulder",
            "observed": "/aloha/joints/waist",
        },
        {
            "field": "name",
            "index": 1,
            "expected": "shoulder",
            "observed": "waist",
        },
        {
            "field": "body1",
            "index": 1,
            "expected": "/aloha/links/shoulder",
            "observed": "/aloha/links/waist",
        },
    ]


def test_compare_dof_records_reports_count_mismatch_explicitly() -> None:
    result = compare_dof_records([_record(0, "waist")], [])

    assert result["mismatches"][0] == {
        "field": "count",
        "index": None,
        "expected": 1,
        "observed": 0,
    }


def test_compare_dof_records_orders_all_mismatch_categories_stably() -> None:
    expected = [_record(0, "waist"), _record(1, "shoulder")]
    observed = [_record(0, "elbow")]

    result = compare_dof_records(expected, observed)

    assert result["mismatches"] == [
        {
            "field": "count",
            "index": None,
            "expected": 2,
            "observed": 1,
        },
        {
            "field": "path",
            "index": 0,
            "expected": "/aloha/joints/waist",
            "observed": "/aloha/joints/elbow",
        },
        {
            "field": "name",
            "index": 0,
            "expected": "waist",
            "observed": "elbow",
        },
        {
            "field": "body1",
            "index": 0,
            "expected": "/aloha/links/waist",
            "observed": "/aloha/links/elbow",
        },
        {
            "field": "missing",
            "index": None,
            "expected": "/aloha/joints/waist",
            "observed": None,
        },
        {
            "field": "missing",
            "index": None,
            "expected": "/aloha/joints/shoulder",
            "observed": None,
        },
        {
            "field": "unexpected",
            "index": None,
            "expected": None,
            "observed": "/aloha/joints/elbow",
        },
    ]


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


@pytest.mark.parametrize(
    ("lower_limit", "upper_limit"), [(1.0, 1.0), (2.0, 1.0)]
)
def test_validate_dof_records_rejects_non_increasing_limits(
    lower_limit: float, upper_limit: float
) -> None:
    record = _record(0, "waist")
    record["lower_limit"] = lower_limit
    record["upper_limit"] = upper_limit

    result = common.validate_dof_records([record])

    assert result["errors"] == [
        {
            "code": "invalid_limit_order",
            "index": 0,
            "lower_limit": lower_limit,
            "upper_limit": upper_limit,
        }
    ]


@pytest.mark.parametrize("field", ["lower_limit", "upper_limit"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_validate_dof_records_rejects_non_finite_limits(
    field: str, value: float
) -> None:
    record = _record(0, "waist")
    record[field] = value

    result = common.validate_dof_records([record])

    assert result["ok"] is False
    assert result["errors"][0]["code"] == "non_finite_limit"
    assert result["errors"][0]["field"] == field
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


def test_validate_dof_records_rejects_missing_required_field() -> None:
    record = _record(0, "waist")
    del record["axis"]

    result = common.validate_dof_records([record])

    assert result["errors"][0] == {
        "code": "missing_field",
        "index": 0,
        "field": "axis",
    }


@pytest.mark.parametrize(("field", "value"), [("path", ""), ("name", "  ")])
def test_validate_dof_records_rejects_empty_identity(field: str, value: str) -> None:
    record = _record(0, "waist")
    record[field] = value

    result = common.validate_dof_records([record])

    assert result["errors"][0] == {
        "code": "invalid_field_type",
        "index": 0,
        "field": field,
        "expected": "non-empty string",
        "observed_type": "str",
    }


@pytest.mark.parametrize(("field", "value"), [("path", []), ("name", {})])
def test_validate_dof_records_rejects_unhashable_identity(
    field: str, value: object
) -> None:
    record = _record(0, "waist")
    record[field] = value

    result = common.validate_dof_records([record])

    assert result["errors"][0] == {
        "code": "invalid_field_type",
        "index": 0,
        "field": field,
        "expected": "non-empty string",
        "observed_type": type(value).__name__,
    }


@pytest.mark.parametrize("field", ["lower_limit", "upper_limit"])
def test_validate_dof_records_rejects_bool_limit(field: str) -> None:
    record = _record(0, "waist")
    record[field] = True

    result = common.validate_dof_records([record])

    assert result["errors"][0] == {
        "code": "invalid_field_type",
        "index": 0,
        "field": field,
        "expected": "finite int or float",
        "observed_type": "bool",
    }


def test_compare_dof_records_rejects_matching_malformed_records() -> None:
    malformed = _record(0, "waist")
    del malformed["axis"]

    result = compare_dof_records([malformed], [deepcopy(malformed)])

    assert result["ok"] is False
    assert result["validation_errors"] == [
        {
            "side": "expected",
            "code": "missing_field",
            "index": 0,
            "field": "axis",
        },
        {
            "side": "observed",
            "code": "missing_field",
            "index": 0,
            "field": "axis",
        },
    ]


def test_validate_safety_flags_rejects_missing_flag() -> None:
    payload = {
        "physics_stepped": False,
        "actions_applied": False,
        "targets_written": False,
    }

    result = common.validate_safety_flags(payload)

    assert result["errors"] == [
        {"code": "missing_field", "field": "stage_saved"}
    ]


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_validate_safety_flags_rejects_non_bool_flag(value: object) -> None:
    payload = {
        "physics_stepped": value,
        "actions_applied": False,
        "targets_written": False,
        "stage_saved": False,
    }

    result = common.validate_safety_flags(payload)

    assert result["errors"] == [
        {
            "code": "invalid_field_type",
            "field": "physics_stepped",
            "expected": "bool",
            "observed_type": type(value).__name__,
        }
    ]
