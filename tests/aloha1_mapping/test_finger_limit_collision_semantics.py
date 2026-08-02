from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.probe_aloha1_finger_limit_collision_semantics import aggregate_report
from tools.probe_aloha1_finger_limit_collision_semantics import validate_session_layer_probe

ROOT = Path(__file__).resolve().parents[2]
REPORT = (
    ROOT
    / "reports/aloha1_mapping/aloha1_finger_limit_collision_semantics.json"
)


def _runtime_record(*, signature: str = "same") -> dict[str, object]:
    return {
        "status": "PASS",
        "stage": {"sha256": "a" * 64},
        "runtime_readback": {
            "dof_order": [
                "waist",
                "shoulder",
                "elbow",
                "forearm_roll",
                "wrist_angle",
                "wrist_rotate",
                "gripper",
                "left_finger",
                "right_finger",
            ],
            "dof_limits": {
                "left_finger": {"lower": 0.021, "upper": 0.057},
                "right_finger": {"lower": -0.0642, "upper": -0.0138},
            },
            "self_collision": False,
        },
        "composed_usd": {
            "authored_limits": {
                "left_finger": {"lower": 0.021, "upper": 0.057},
                "right_finger": {"lower": -0.0642, "upper": -0.0138},
            },
            "mimic_api": {
                "effective_multiplier": -1.0,
                "effective_offset": 0.0,
            },
        },
        "filtered_pair_inventory": [],
        "deterministic_signature": signature,
    }


def test_aggregate_detects_composed_right_finger_limit_defect() -> None:
    source = {
        "limits": {
            "left_finger": {"lower": 0.021, "upper": 0.057},
            "right_finger": {"lower": -0.057, "upper": -0.021},
        },
        "mimic": {
            "joint": "left_finger",
            "multiplier": -1.0,
            "offset": 0.0,
        },
    }

    report = aggregate_report(
        source_urdf=source,
        runtime_records=[_runtime_record(), _runtime_record()],
        candidate=None,
    )

    assert report["limit_semantics_status"] == "VERIFIED_USD_LIMIT_DEFECT"
    assert report["candidate_created"] is False
    assert report["pair_collision_support_status"] == "INCONCLUSIVE"
    assert report["fresh_process_determinism"]["status"] == "PASS"


def test_aggregate_rejects_non_deterministic_fresh_processes() -> None:
    source = {
        "limits": {
            "left_finger": {"lower": 0.021, "upper": 0.057},
            "right_finger": {"lower": -0.057, "upper": -0.021},
        },
        "mimic": {
            "joint": "left_finger",
            "multiplier": -1.0,
            "offset": 0.0,
        },
    }

    with pytest.raises(ValueError, match="deterministic"):
        aggregate_report(
            source_urdf=source,
            runtime_records=[
                _runtime_record(signature="first"),
                _runtime_record(signature="second"),
            ],
            candidate=None,
        )


def test_generated_report_has_required_machine_schema() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["source_urdf"]["limits"]
    assert report["source_urdf"]["mimic"]
    assert report["composed_usd"]["authored_limits"]
    assert report["composed_usd"]["mimic_api"]
    assert report["runtime_readback"]["dof_limits"]
    assert report["runtime_readback"]["self_collision"] is False
    assert report["limit_semantics_status"] in {
        "VERIFIED_EQUIVALENT",
        "VERIFIED_USD_LIMIT_DEFECT",
        "INCONCLUSIVE",
    }
    assert report["pair_collision_support_status"] in {
        "SUPPORTED_LOCAL_5_1",
        "NOT_SUPPORTED_LOCAL_5_1",
        "INCONCLUSIVE",
    }
    assert isinstance(report["candidate_created"], bool)
    assert report["task8"] == "NOT_RUN"


def test_session_layer_probe_requires_source_limits_and_immutable_root() -> None:
    source_limits = {
        "left_finger": {"lower": 0.021, "upper": 0.057},
        "right_finger": {"lower": -0.057, "upper": -0.021},
    }
    layer_path = "/tmp/finger_source_limits.usda"
    record = {
        "status": "PASS",
        "stage": {
            "sha256_before": "a" * 64,
            "sha256_after": "a" * 64,
            "root_sublayers_before": ["geometry.usda"],
            "root_sublayers_after": ["geometry.usda"],
        },
        "session_sublayer_application": {
            "status": "PASS",
            "inserted_paths": [layer_path],
            "after": [layer_path],
            "root_layer_saved": False,
        },
        "runtime_readback": {"dof_limits": source_limits},
        "composed_usd": {"authored_limits": source_limits},
    }

    result = validate_session_layer_probe(
        record=record,
        source_limits=source_limits,
        expected_stage_sha256="a" * 64,
        expected_layer_path=layer_path,
    )

    assert result["status"] == "PASS"
    assert all(result["gates"].values())
