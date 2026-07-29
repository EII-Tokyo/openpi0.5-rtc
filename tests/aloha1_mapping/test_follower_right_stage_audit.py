from __future__ import annotations

import json
from pathlib import Path

from tools.aloha1_mapping.follower_right_stage_audit import classify_candidate
from tools.aloha1_mapping.follower_right_stage_audit import summarize_audit

SUPPLIER_HASHES = {
    "left": "left-supplier-v2",
    "right": "right-supplier-v2",
}
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/audit_aloha1_follower_right_candidates.py"
REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_follower_right_stage_audit.json"
)


def _candidate(**overrides: object) -> dict[str, object]:
    candidate: dict[str, object] = {
        "absolute_path": "/repo/candidate.usda",
        "open_status": "PASS",
        "sha256_before": "stage-hash",
        "sha256_after": "stage-hash",
        "articulation_roots": ["/follower_right/root_joint"],
        "right_prim_count": 10,
        "finger_source_hashes": {
            "left-supplier-v2",
            "right-supplier-v2",
        },
        "finger_mesh_signatures": [
            {"point_count": 831, "face_count": 1662},
            {"point_count": 831, "face_count": 1662},
        ],
        "used_layers": ["/repo/current_supplier_right.usda"],
    }
    candidate.update(overrides)
    return candidate


def test_current_supplier_pair_requires_independent_right_articulation() -> None:
    result = classify_candidate(_candidate(), SUPPLIER_HASHES)
    assert result["classification"] == "ELIGIBLE_CURRENT_SUPPLIER_CAD"
    assert result["eligible"] is True
    assert result["gates"]["supplier_hash_pair"] is True
    assert result["gates"]["independent_right_articulation"] is True


def test_rejected_phantom_branch_cannot_become_eligible_from_hashes() -> None:
    result = classify_candidate(
        _candidate(
            used_layers=[
                "/repo/cad_finger_task5_convex_hull_"
                "rejected_phantom_right_branch/root.usda"
            ]
        ),
        SUPPLIER_HASHES,
    )
    assert result["classification"] == "REJECTED_PHANTOM_RIGHT_BRANCH"
    assert result["eligible"] is False


def test_old_generic_and_historical_fingers_are_not_supplier_v2() -> None:
    generic = classify_candidate(
        _candidate(
            finger_source_hashes=set(),
            finger_mesh_signatures=[
                {"point_count": 2568, "face_count": 856},
                {"point_count": 2568, "face_count": 856},
            ],
        ),
        SUPPLIER_HASHES,
    )
    assert generic["classification"] == "REJECTED_GENERIC_FINGER"

    historical = classify_candidate(
        _candidate(
            finger_source_hashes=set(),
            finger_mesh_signatures=[
                {"point_count": 4998, "face_count": 1666},
                {"point_count": 4998, "face_count": 1666},
            ],
        ),
        SUPPLIER_HASHES,
    )
    assert historical["classification"] == (
        "HISTORICAL_GYM_ALOHA_NOT_CURRENT_SUPPLIER_CAD"
    )


def test_aloha2_or_combined_articulation_is_not_a_right_follower_asset() -> None:
    aloha2 = classify_candidate(
        _candidate(
            used_layers=[
                "/repo/aloha2_menagerie_scene_base.usd",
            ]
        ),
        SUPPLIER_HASHES,
    )
    assert aloha2["classification"] == "REJECTED_ALOHA2_OR_LEGACY_REBUILD"

    combined = classify_candidate(
        _candidate(
            articulation_roots=["/aloha/root_joint"],
        ),
        SUPPLIER_HASHES,
    )
    assert combined["classification"] == (
        "REJECTED_NOT_INDEPENDENT_FOLLOWER_RIGHT_ARTICULATION"
    )


def test_summary_records_hard_blocker_when_no_candidate_is_eligible() -> None:
    records = [
        classify_candidate(
            _candidate(
                finger_source_hashes=set(),
                finger_mesh_signatures=[
                    {"point_count": 2568, "face_count": 856}
                ],
            ),
            SUPPLIER_HASHES,
        )
    ]
    summary = summarize_audit(
        records,
        cad_identity_classification=(
            "VERIFIED_SINGLE_REUSABLE_ROBOT_PRODUCT"
        ),
        workcell_placement_verified=False,
    )
    assert summary["status"] == "PARTIAL"
    assert summary["eligible_count"] == 0
    assert summary["cad_availability"] == (
        "VERIFIED_SINGLE_REUSABLE_ROBOT_PRODUCT"
    )
    assert summary["next_action"] == (
        "GENERATE_ROBOT_LOCAL_FOLLOWER_RIGHT_DIAGNOSTIC_STAGE"
    )
    assert summary["hard_blockers"] == [
        "HARD_BLOCKER_NO_CURRENT_SUPPLIER_CAD_FOLLOWER_RIGHT_STAGE",
        "HARD_BLOCKER_FOLLOWER_RIGHT_WORKCELL_INSTALL_TRANSFORM",
    ]
    assert summary["blocker_definitions"][
        "HARD_BLOCKER_NO_CURRENT_SUPPLIER_CAD_FOLLOWER_RIGHT_STAGE"
    ]["meaning"] == (
        "No already-generated and validated supplier-CAD follower_right "
        "Stage exists yet."
    )
    assert summary["blocker_definitions"][
        "HARD_BLOCKER_NO_CURRENT_SUPPLIER_CAD_FOLLOWER_RIGHT_STAGE"
    ]["does_not_mean"] == "Supplier CAD lacks the right-arm robot product."


def test_audit_script_is_bounded_and_uses_supplier_mesh_hashes() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "local_eval_assets/aloha_isaac" in source
    assert "assets/Trossen/ALOHA1" in source
    assert "aloha_isaac_rebuild" in source
    assert "c6710d0fe5b2030a" in source
    assert "b0979c5d55fee448" in source
    assert "Usd.Stage.Open" in source
    assert "GetCustomData" in source


def test_generated_audit_promotes_only_the_new_supplier_right_stage() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["eligible_count"] == 1
    assert report["approved_stage"]["follower_right_present"] is False
    assert report["protected_inputs_unchanged"] is True
    assert report["task8"] == "NOT_RUN"
    assert report["cad_availability"] == (
        "VERIFIED_SINGLE_REUSABLE_ROBOT_PRODUCT"
    )
    assert report["next_action"] == (
        "VALIDATE_EXISTING_FOLLOWER_RIGHT_STAGE"
    )
    assert report["approved_stage"]["absence_scope"] == (
        "APPROVED_LEFT_REVIEW_STAGE_ONLY"
    )
    assert report["hard_blockers"] == [
        "HARD_BLOCKER_APPROVED_STAGE_MISSING_FOLLOWER_RIGHT",
        "HARD_BLOCKER_FOLLOWER_RIGHT_WORKCELL_INSTALL_TRANSFORM",
    ]
    assert report["blocker_definitions"][
        "HARD_BLOCKER_APPROVED_STAGE_MISSING_FOLLOWER_RIGHT"
    ]["does_not_mean"] == "Supplier CAD lacks the right-arm robot product."
    eligible = [item for item in report["candidates"] if item["eligible"]]
    assert len(eligible) == 1
    assert eligible[0]["absolute_path"].endswith(
        "supplier_cad_follower_right/1.0/"
        "supplier_cad_follower_right.usda"
    )
    assert eligible[0]["articulation_roots"] == [
        "/follower_right/vx300s_right/root_joint"
    ]
