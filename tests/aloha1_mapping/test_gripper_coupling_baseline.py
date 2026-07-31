from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/reproduce_aloha1_gripper_coupling_baseline.py"
STAGE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/table_support_alignment/1.0"
    / "aloha1_table_support_aligned_workcell.usda"
)
STAGE_SHA256 = "2b3f76365ed67532f478d995ae859a88b5639975ac07cb7ac8a53ac679e8205c"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "reproduce_aloha1_gripper_coupling_baseline",
        TOOL,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trial(
    index: int,
    *,
    load_case: str,
    residual_m: float,
    contact: bool,
) -> dict[str, object]:
    left = 0.04995
    right = -left - residual_m
    return {
        "schema_version": 1,
        "status": "PASS",
        "run_index": index,
        "load_case": load_case,
        "fresh_process": True,
        "runtime": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
        },
        "inputs": {
            "stage": {
                "absolute_path": str(STAGE),
                "sha256": STAGE_SHA256,
            }
        },
        "controls": {
            "physics_frequency_hz": 60,
            "solve_articulation_contact_last": True,
            "left_finger_target_m": 0.048316874538855845,
            "right_finger_commanded": False,
            "mimic_authored_unchanged": True,
        },
        "final_readback": {
            "left_finger_m": left,
            "right_finger_m": right,
            "ideal_right_finger_m": -left,
            "mimic_residual_abs_m": residual_m,
            "left_velocity_m_s": 0.0,
            "right_velocity_m_s": 0.0,
        },
        "contacts": {
            "bilateral_finger_contact": contact,
            "maximum_impulse_ns": 0.0005 if contact else None,
            "minimum_separation_m": -0.0001 if contact else None,
        },
        "source_stage_unchanged": True,
    }


def test_runtime_contract_preserves_the_frozen_baseline() -> None:
    module = _load_tool()
    contract = module.runtime_contract()

    assert contract["stage"]["absolute_path"] == str(STAGE)
    assert contract["stage"]["sha256"] == STAGE_SHA256
    assert contract["articulation"] == ("/World/follower_left/vx300s_left/root_joint")
    assert contract["dof_order"][-2:] == ["left_finger", "right_finger"]
    assert contract["physics_frequency_hz"] == 60
    assert contract["solve_articulation_contact_last"] is True
    assert contract["mimic_policy"] == "UNCHANGED_PHYSX_MIMIC"
    assert contract["right_finger_commanded"] is False
    assert contract["left_close_target_m"] == pytest.approx(0.048316874538855845)
    assert contract["task8"] == "NOT_RUN"


def test_classification_requires_five_fresh_runs_per_load_case() -> None:
    module = _load_tool()
    trials = [_trial(index, load_case="bottle_contact", residual_m=0.00178, contact=True) for index in range(5)]
    trials.extend(
        _trial(
            index,
            load_case="no_object_contact",
            residual_m=0.00142,
            contact=False,
        )
        for index in range(5)
    )

    report = module.aggregate_trials(trials)
    assert report["status"] == "PASS"
    assert report["classification"] == "PHYSX_MIMIC_COUPLING_RESIDUAL"
    assert report["run_count"] == 10
    assert report["fresh_process_count"] == 10
    assert report["load_cases"]["bottle_contact"]["run_count"] == 5
    assert report["load_cases"]["no_object_contact"]["run_count"] == 5
    assert report["load_cases"]["bottle_contact"]["deterministic"] is True
    assert report["load_cases"]["no_object_contact"]["deterministic"] is True
    assert report["contact_dependency"]["status"] == ("CONTACT_AMPLIFIES_BUT_DOES_NOT_CREATE_RESIDUAL")
    assert report["mimic_gate"]["status"] == "FAIL"
    assert report["next_gate"] == "ISOLATED_COUPLING_AB"
    assert report["task8"] == "NOT_RUN"


def test_report_fails_closed_on_missing_or_nonfresh_runs() -> None:
    module = _load_tool()
    trials = [_trial(index, load_case="bottle_contact", residual_m=0.00178, contact=True) for index in range(5)]
    trials.extend(
        _trial(
            index,
            load_case="no_object_contact",
            residual_m=0.00142,
            contact=False,
        )
        for index in range(4)
    )
    trials[0]["fresh_process"] = False

    report = module.aggregate_trials(trials)
    assert report["status"] == "FAIL"
    assert "INSUFFICIENT_FRESH_RUNS:no_object_contact" in report["failure_reasons"]
    assert "NONFRESH_PROCESS_RECORD" in report["failure_reasons"]
    assert report["classification"] == "INCONCLUSIVE"


def test_signature_ignores_runtime_but_not_physical_readback() -> None:
    module = _load_tool()
    first = _trial(
        0,
        load_case="bottle_contact",
        residual_m=0.00178,
        contact=True,
    )
    second = _trial(
        1,
        load_case="bottle_contact",
        residual_m=0.00178,
        contact=True,
    )
    first["runtime_seconds"] = 10.0
    second["runtime_seconds"] = 20.0
    assert module.deterministic_signature(first) == module.deterministic_signature(second)

    second["final_readback"]["right_finger_m"] = -0.06
    assert module.deterministic_signature(first) != module.deterministic_signature(second)


def _native_gui_report(residual_m: float) -> dict[str, object]:
    left = 0.04995
    right = -left - residual_m
    return {
        "status": "FAIL",
        "cleanup_errors": [],
        "inputs": {
            "stage": {
                "path": str(STAGE),
                "sha256": STAGE_SHA256,
            }
        },
        "runtime": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
        },
        "result": {
            "execution_mode": "external_contact_skip_sim",
            "gate": {
                "status": "FAIL",
                "bilateral_contact": "PASS",
                "mimic_accuracy": "FAIL",
                "raw_export": "PASS",
                "derived_export": "PASS",
                "failure_reasons": ["MIMIC_ACCURACY_FAILED"],
            },
            "joint_readback": {
                "left_finger_after_m": left,
                "right_finger_after_m": right,
                "mimic_error_abs_m": residual_m,
                "after_test": [0.0] * 7 + [left, right],
            },
            "external_close_trace": [
                {
                    "readback_left_finger_m": left,
                    "readback_right_finger_m": right,
                }
            ],
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


def test_native_gui_normalization_treats_expected_mimic_gate_as_evidence() -> None:
    module = _load_tool()
    normalized = module.normalize_native_grasp_editor_trial(
        _native_gui_report(0.0017794594168663025),
        run_index=3,
        report_path=Path("/tmp/native_run03/report.json"),
    )

    assert normalized["status"] == "PASS"
    assert normalized["run_index"] == 3
    assert normalized["execution_boundary"] == (
        "NATIVE_GRASP_EDITOR_GUI_EXTERNAL_CONTACT_SKIP_SIM"
    )
    assert normalized["final_readback"]["mimic_residual_abs_m"] == pytest.approx(
        0.0017794594168663025
    )
    assert normalized["contacts"]["bilateral_finger_contact"] is True
    assert normalized["source_stage_unchanged"] is True


def test_execution_boundary_matrix_classifies_reset_dependency() -> None:
    module = _load_tool()
    native = [
        module.normalize_native_grasp_editor_trial(
            _native_gui_report(0.0017794594168663025),
            run_index=index,
            report_path=Path(f"/tmp/native_run{index:02d}/report.json"),
        )
        for index in range(5)
    ]
    fresh_reset = [
        {
            **_trial(
                index,
                load_case="bottle_contact",
                residual_m=0.00082,
                contact=True,
            ),
            "execution_boundary": "FRESH_WORLD_RESET_HEADLESS",
        }
        for index in range(5)
    ]

    report = module.compare_execution_boundaries(
        native_trials=native,
        reset_trials=fresh_reset,
    )
    assert report["status"] == "PASS"
    assert report["classification"] == "RESET_DEPENDENT"
    assert report["native_grasp_editor"]["run_count"] == 5
    assert report["fresh_world_reset"]["run_count"] == 5
    assert report["native_grasp_editor"]["mimic_gate"] == "FAIL"
    assert report["fresh_world_reset"]["mimic_gate"] == "PASS"
    assert report["next_gate"] == "ISOLATED_COUPLING_AB"
    assert report["task8"] == "NOT_RUN"
