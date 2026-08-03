from __future__ import annotations

from pathlib import Path

from tools.build_aloha1_official_model_first_closure import build_closure

ROOT = Path(__file__).resolve().parents[2]


def test_closure_keeps_model_gate_partial_and_candidate_unpromoted() -> None:
    closure = build_closure(ROOT)

    assert closure["status"] == "PARTIAL_MODEL_PROOF"
    assert closure["compound_contact_region_status"] == (
        "PASS_RUNTIME_COOKED_CONTACT_REGION_GEOMETRY_NOT_PROMOTED"
    )
    assert closure["compound_task_contact_band_status"] == (
        "FAIL_CENTRAL_TANGENCY_OUTSIDE_COMPOUND_PATCH"
    )
    assert closure["compound_task_contact_band_decision"] == (
        "REJECTED_TASK_CONTACT_BAND_NOT_PROMOTED"
    )
    assert closure["compound_usd_status"] == "PASS_GEOMETRY_ONLY_DIAGNOSTIC_USD"
    assert closure["collider_contract_status"] == "PARTIAL"
    assert closure["official_model_candidate_status"] == "NOT_BUILT_BLOCKED"
    assert closure["numerical_convergence_status"] == "PARTIAL"
    assert closure["numerical_convergence_reason"] == (
        "TIMESTEP_NOT_CONVERGED_SOLVER_SWEEPS_NOT_RUN"
    )
    assert closure["force_drive_candidate_status"] == "HARD_BLOCKER"
    assert closure["material_thermal_status"] == "HARD_BLOCKER"
    assert closure["evidence_boundaries"]["aggregate_scope"] == (
        "STATIC_GEOMETRY_AND_DYNAMIC_NUMERICAL_DIAGNOSTICS"
    )
    assert closure["evidence_boundaries"][
        "numerical_convergence_timeline_started"
    ] is True
    assert closure["task8_status"] == "AUTHORIZED_PAUSED_AT_MODEL_PROOF_GATE"
    assert closure["final_or_default_asset_modified"] is False
