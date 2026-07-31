from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/validate_aloha1_gripper_coupling_ab.py"
STAGE_SHA256 = (
    "2b3f76365ed67532f478d995ae859a88b5639975ac07cb7ac8a53ac679e8205c"
)


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "validate_aloha1_gripper_coupling_ab",
        TOOL,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trial(
    variant: str,
    index: int,
    residual_m: float,
    *,
    maximum_impulse_ns: float = 0.0005,
    minimum_separation_m: float = -0.0001,
) -> dict[str, object]:
    return {
        "status": "PASS",
        "variant": variant,
        "run_index": index,
        "fresh_process": True,
        "stage_sha256": STAGE_SHA256,
        "mimic_residual_abs_m": residual_m,
        "bilateral_contact": True,
        "maximum_impulse_ns": maximum_impulse_ns,
        "minimum_separation_m": minimum_separation_m,
        "source_stage_unchanged": True,
        "single_variable_contract": True,
    }


def test_variant_contract_changes_only_coupling_representation() -> None:
    module = _load_tool()
    baseline = module.coupling_variant_contract("current_physx_mimic")
    symmetric = module.coupling_variant_contract(
        "official_symmetric_adapter"
    )

    assert baseline["stage_sha256"] == STAGE_SHA256
    assert symmetric["stage_sha256"] == STAGE_SHA256
    assert baseline["changed_variable"] == "NONE_BASELINE"
    assert symmetric["changed_variable"] == "COUPLING_REPRESENTATION_ONLY"
    assert symmetric["classification"] == (
        "DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING"
    )
    assert symmetric["right_target_formula"] == "q_right=-q_left"
    assert symmetric["collider_changed"] is False
    assert symmetric["friction_changed"] is False
    assert symmetric["drive_magnitude_changed"] is False
    assert symmetric["bottle_changed"] is False
    assert symmetric["timestep_changed"] is False
    assert symmetric["solver_changed"] is False
    assert symmetric["initial_pose_changed"] is False
    assert symmetric["task8"] == "NOT_RUN"


def test_symmetric_targets_preserve_exact_sign_relation() -> None:
    module = _load_tool()
    targets = module.build_coupling_targets(
        "official_symmetric_adapter",
        left_target_m=0.048316874538855845,
        left_index=7,
        right_index=8,
    )

    assert targets["joint_indices"] == [7, 8]
    assert targets["joint_positions_m"] == pytest.approx(
        [0.048316874538855845, -0.048316874538855845]
    )


def test_ab_requires_five_fresh_runs_and_classifies_mimic_primary() -> None:
    module = _load_tool()
    trials = [
        _trial("current_physx_mimic", index, 0.0017794594168663025)
        for index in range(5)
    ]
    trials.extend(
        _trial("official_symmetric_adapter", index, 0.0002)
        for index in range(5)
    )

    result = module.classify_ab_trials(trials)
    assert result["status"] == "PASS"
    assert result["classification"] == "PHYSX_MIMIC_PRIMARY"
    assert result["variants"]["current_physx_mimic"]["mimic_gate"] == "FAIL"
    assert (
        result["variants"]["official_symmetric_adapter"]["mimic_gate"]
        == "PASS"
    )
    assert result["promotion_authorized"] is False
    assert result["next_gate"] == "GRASP_EDITOR_DIAGNOSTIC_ON_PASSING_PATH"
    assert result["task8"] == "NOT_RUN"


def test_ab_fails_closed_when_a_fresh_run_is_missing() -> None:
    module = _load_tool()
    trials = [
        _trial("current_physx_mimic", index, 0.0017)
        for index in range(5)
    ]
    trials.extend(
        _trial("official_symmetric_adapter", index, 0.0002)
        for index in range(4)
    )

    result = module.classify_ab_trials(trials)
    assert result["status"] == "FAIL"
    assert result["classification"] == "INCONCLUSIVE"
    assert "INSUFFICIENT_FRESH_RUNS:official_symmetric_adapter" in (
        result["failure_reasons"]
    )


def test_ab_rejects_a_zero_residual_created_by_deep_penetration() -> None:
    module = _load_tool()
    trials = [
        _trial("current_physx_mimic", index, 0.0017)
        for index in range(5)
    ]
    trials.extend(
        _trial(
            "official_symmetric_adapter",
            index,
            0.0,
            maximum_impulse_ns=0.645,
            minimum_separation_m=-0.0094,
        )
        for index in range(5)
    )

    result = module.classify_ab_trials(trials)
    assert result["status"] == "FAIL"
    assert result["classification"] == "INCONCLUSIVE"
    assert "CONTACT_EQUIVALENCE_FAILED" in result["failure_reasons"]
    assert result["promotion_authorized"] is False


def test_native_gui_report_normalization_preserves_expected_baseline_fail() -> None:
    module = _load_tool()
    report = {
        "status": "FAIL",
        "cleanup_errors": [],
        "inputs": {
            "stage": {
                "sha256": STAGE_SHA256,
            }
        },
        "runtime": {
            "coupling_variant": module.coupling_variant_contract(
                "current_physx_mimic"
            )
        },
        "result": {
            "gate": {
                "bilateral_contact": "PASS",
                "raw_export": "PASS",
                "derived_export": "PASS",
                "failure_reasons": ["MIMIC_ACCURACY_FAILED"],
            },
            "joint_readback": {
                "mimic_error_abs_m": 0.0017794594168663025,
            },
            "contacts": {
                "summary": {
                    "status": "PASS",
                    "bilateral_finger_contact": True,
                    "maximum_impulse_ns": 0.000547,
                    "minimum_separation_m": -0.000122,
                }
            },
        },
    }

    normalized = module.normalize_gui_trial(
        report,
        run_index=2,
        report_path=Path("/tmp/run02/report.json"),
    )
    assert normalized["status"] == "PASS"
    assert normalized["variant"] == "current_physx_mimic"
    assert normalized["mimic_residual_abs_m"] == pytest.approx(
        0.0017794594168663025
    )
    assert normalized["source_stage_unchanged"] is True
