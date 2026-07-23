"""Regression coverage for A21 clean-runtime finger mapping overrides."""

from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest
import yaml

try:
    import pxr  # noqa: F401
except ModuleNotFoundError:
    pxr_stub = types.ModuleType("pxr")
    pxr_stub.Usd = types.SimpleNamespace(Stage=object)
    pxr_stub.Gf = types.SimpleNamespace()
    pxr_stub.Sdf = types.SimpleNamespace()
    pxr_stub.UsdGeom = types.SimpleNamespace()
    pxr_stub.UsdPhysics = types.SimpleNamespace(
        CollisionAPI=object,
        RigidBodyAPI=object,
        MassAPI=object,
        ArticulationRootAPI=object,
        FilteredPairsAPI=object,
    )
    sys.modules["pxr"] = pxr_stub

from aloha_isaac_rebuild.scripts.create_aloha_clean_articulation_mapping_plan import (
    apply_clean_runtime_mapping_override,
)
from aloha_isaac_rebuild.scripts.create_aloha_clean_articulation_mapping_plan import (
    validate_clean_runtime_mapping_override_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "aloha_isaac_rebuild/configs/physical_reconstruction/stationary_style_rebuild.yaml"
ORIGINAL_MAPPING_PATH = REPO_ROOT / "configs/aloha/original_stationary_aloha_mapping.yaml"
RIGHT_FINGER_PATHS = (
    "/aloha/joints/left_right_finger",
    "/aloha/joints/right_right_finger",
)


def _source_record(path: str = RIGHT_FINGER_PATHS[0]) -> dict:
    return {
        "proposed_clean_joint_path": path,
        "lower_limit": "0.01844",
        "upper_limit": "0.058",
        "canonical_mapping": {
            "canonical_name": "left_gripper",
            "dataset_index": 6,
            "openpi_index": 6,
            "sign": -1.0,
            "offset": -0.021,
            "scale": -0.036,
            "unit": "m",
            "source": "robot_description vx300s right_finger mimic multiplier -1",
            "confidence": 0.85,
        },
    }


def _override(**changes: object) -> dict:
    override = {
        "sign": 1.0,
        "offset": 0.021,
        "scale": 0.036,
        "unit": "m",
        "rationale": "clean Isaac DOF coordinate already mirrors through its joint frame",
        "source": "A19 authored and A20 runtime positive prismatic limits",
    }
    override.update(changes)
    return override


def test_override_preserves_source_mapping_and_replaces_effective_mapping() -> None:
    record = _source_record()
    record["canonical_mapping"]["metadata"] = {"nested": {"coordinate": "source"}}
    override = _override()
    result = apply_clean_runtime_mapping_override(
        record,
        {RIGHT_FINGER_PATHS[0]: override},
    )

    assert result["source_canonical_mapping"] == record["canonical_mapping"]
    assert result["source_canonical_mapping"] is not record["canonical_mapping"]
    assert result["canonical_mapping"] == {
        **record["canonical_mapping"],
        "sign": 1.0,
        "offset": 0.021,
        "scale": 0.036,
        "unit": "m",
        "source": "A19 authored and A20 runtime positive prismatic limits",
    }
    assert result["clean_runtime_mapping_override"] == override
    assert result["clean_runtime_mapping_override"] is not override
    assert record["canonical_mapping"]["scale"] == -0.036
    result["canonical_mapping"]["metadata"]["nested"]["coordinate"] = "effective"
    assert record["canonical_mapping"]["metadata"]["nested"]["coordinate"] == "source"
    assert result["source_canonical_mapping"]["metadata"]["nested"]["coordinate"] == "source"


def test_no_override_preserves_source_and_effective_mapping() -> None:
    record = _source_record()
    result = apply_clean_runtime_mapping_override(record, {})

    assert result["source_canonical_mapping"] == record["canonical_mapping"]
    assert result["canonical_mapping"] == record["canonical_mapping"]
    assert result["clean_runtime_mapping_override"] is None


def test_record_without_canonical_mapping_is_returned_unchanged() -> None:
    record = {"proposed_clean_joint_path": RIGHT_FINGER_PATHS[0], "canonical_mapping": None}

    assert apply_clean_runtime_mapping_override(record, {}) is record


@pytest.mark.parametrize("path", RIGHT_FINGER_PATHS)
def test_both_approved_right_finger_paths_support_the_override(path: str) -> None:
    result = apply_clean_runtime_mapping_override(_source_record(path), {path: _override()})

    assert result["canonical_mapping"]["scale"] == 0.036
    assert result["canonical_mapping"]["offset"] == 0.021


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"offset": -0.021}, "outside clean joint limits"),
        ({"offset": 0.021, "scale": -0.036}, "positive scale"),
        ({"rationale": ""}, "rationale"),
        ({"sign": float("nan")}, "finite numeric sign"),
        ({"offset": float("inf")}, "finite numeric offset"),
        ({"scale": float("nan")}, "finite numeric scale"),
        ({"unit": "rad"}, "unit mismatch"),
    ],
)
def test_override_rejects_invalid_clean_runtime_transform(
    changes: dict[str, object], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        apply_clean_runtime_mapping_override(
            _source_record(),
            {RIGHT_FINGER_PATHS[0]: _override(**changes)},
        )


def test_override_requires_all_provenance_and_transform_fields() -> None:
    for field in ("sign", "offset", "scale", "unit", "rationale", "source"):
        override = _override()
        override.pop(field)
        with pytest.raises(ValueError, match=field):
            apply_clean_runtime_mapping_override(
                _source_record(), {RIGHT_FINGER_PATHS[0]: override}
            )


def test_unknown_configured_override_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="not consumed exactly once"):
        validate_clean_runtime_mapping_override_paths(
            [_source_record()],
            {"/aloha/joints/not_a_real_joint": _override()},
        )


@pytest.mark.parametrize(
    "records",
    [
        [_source_record(), _source_record()],
        [
            _source_record(),
            {
                "proposed_clean_joint_path": RIGHT_FINGER_PATHS[0],
                "canonical_mapping": None,
            },
        ],
    ],
    ids=["two-mapped-duplicates", "mapped-and-unmapped-duplicates"],
)
def test_configured_override_requires_one_raw_path_match_and_one_application(
    records: list[dict],
) -> None:
    with pytest.raises(ValueError, match="raw match count"):
        validate_clean_runtime_mapping_override_paths(
            records,
            {RIGHT_FINGER_PATHS[0]: _override()},
        )


def test_stationary_config_declares_the_two_approved_overrides_and_a21_outputs() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["outputs"] == {
        **config["outputs"],
        "a21_policy_target_limit_preflight_json": "aloha_isaac_rebuild/artifacts/validation/a21_policy_target_limit_preflight.json",
        "a21_runtime_target_readback_json": "aloha_isaac_rebuild/artifacts/validation/a21_runtime_target_readback.json",
        "a21_target_limit_and_readback_md": "aloha_isaac_rebuild/reports/a21_target_limit_and_readback.md",
    }
    assert config["clean_runtime_mapping_overrides"] == {
        path: _override() for path in RIGHT_FINGER_PATHS
    }


def test_original_mapping_retains_negative_right_finger_mimic_semantics() -> None:
    mapping = yaml.safe_load(ORIGINAL_MAPPING_PATH.read_text(encoding="utf-8"))
    right_fingers = [
        record
        for record in mapping["dof_mapping"]
        if record["isaac_dof_name"].endswith("/right_finger")
    ]

    assert len(right_fingers) == 2
    assert all(record["sign"] == -1.0 for record in right_fingers)
    assert all(record["offset"] == -0.021 for record in right_fingers)
    assert all(record["scale"] == -0.036 for record in right_fingers)
    assert all("mimic multiplier -1" in record["source"] for record in right_fingers)
