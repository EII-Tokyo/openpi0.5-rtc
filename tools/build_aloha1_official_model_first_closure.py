#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = Path("reports/aloha1_mapping")
DEFAULT_JSON = ROOT / REPORT_DIR / "aloha1_official_model_first_closure.json"
DEFAULT_MARKDOWN = ROOT / REPORT_DIR / "aloha1_official_model_first_closure.md"

INPUT_REPORTS = {
    "official_parameter_source_audit": "aloha1_official_parameter_source_audit.json",
    "official_parameter_matrix": "aloha1_official_parameter_matrix.json",
    "kinematic_contract": "aloha1_kinematic_contract.json",
    "dynamics_contract": "aloha1_dynamics_contract.json",
    "gripper_geometry_contract": "aloha1_gripper_geometry_contract.json",
    "collider_geometry_contract": "aloha1_collider_geometry_contract.json",
    "compound_runtime_cooking": "aloha1_supplier_cad_compound_runtime_cooking_certificate.json",
    "compound_geometry_usd": "aloha1_supplier_cad_compound_contact_usd.json",
    "official_model_candidate": "aloha1_official_model_candidate.json",
    "official_model_runtime": "aloha1_official_model_runtime.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_inputs(root: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, object]]]:
    reports: dict[str, dict[str, Any]] = {}
    manifest: list[dict[str, object]] = []
    for report_id, filename in INPUT_REPORTS.items():
        path = root / REPORT_DIR / filename
        report = json.loads(path.read_text(encoding="utf-8"))
        reports[report_id] = report
        manifest.append(
            {
                "id": report_id,
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "status": report["status"],
                "deterministic_signature": report.get("deterministic_signature"),
            }
        )
    return reports, manifest


def build_closure(root: Path) -> dict[str, object]:
    reports, manifest = _load_inputs(root)
    matrix = reports["official_parameter_matrix"]
    dynamics = reports["dynamics_contract"]
    collider = reports["collider_geometry_contract"]
    cooking = reports["compound_runtime_cooking"]
    compound_usd = reports["compound_geometry_usd"]
    candidate = reports["official_model_candidate"]
    runtime = reports["official_model_runtime"]

    if cooking["coordinate_frame"] != "FINGER_LINK_LOCAL_METRES":
        raise ValueError("compound cooking certificate is not finger-link-local")
    if compound_usd["coordinate_frame"] != "FINGER_LINK_LOCAL_METRES":
        raise ValueError("compound diagnostic USD is not finger-link-local")
    if cooking["fresh_process_determinism"]["status"] != "PASS_DETERMINISTIC_FRESH_PROCESS_COOKING":
        raise ValueError("compound runtime cooking is not deterministic across fresh processes")
    if compound_usd["determinism"]["status"] != "PASS_TWO_FRESH_BUILDS_BYTE_IDENTICAL":
        raise ValueError("compound diagnostic USD is not byte-identical across fresh builds")

    blockers = matrix["formal_parameter_candidate_gate"]["blocking_records"]
    closure: dict[str, object] = {
        "schema_version": 1,
        "status": "PARTIAL_MODEL_PROOF",
        "inputs": manifest,
        "verified_gates": {
            "official_source_chain": reports["official_parameter_source_audit"]["status"],
            "kinematic_contract": reports["kinematic_contract"]["status"],
            "gripper_geometry_contract": reports["gripper_geometry_contract"]["status"],
            "compound_runtime_fresh_process_determinism": cooking["fresh_process_determinism"]["status"],
            "compound_usd_fresh_build_determinism": compound_usd["determinism"]["status"],
        },
        "compound_contact_region_status": cooking["status"],
        "compound_contact_region_coordinate_frame": cooking["coordinate_frame"],
        "compound_full_face_scope": collider["supplier_finger_compound_full_face_scope"],
        "compound_usd_status": compound_usd["status"],
        "compound_articulation_integration_status": compound_usd["articulation_integration_status"],
        "compound_contact_dynamics_status": compound_usd["contact_dynamics_status"],
        "compound_video_status": cooking["video_status"],
        "collider_contract_status": collider["status"],
        "dynamics_contract_status": dynamics["status"],
        "official_model_candidate_status": candidate["status"],
        "official_model_runtime_status": runtime["status"],
        "formal_parameter_blockers": blockers,
        "remaining_blocker_count": len(blockers),
        "task8_status": "AUTHORIZED_PAUSED_AT_MODEL_PROOF_GATE",
        "final_or_default_asset_modified": any(
            (
                collider["final_or_default_asset_modified"],
                cooking["final_or_default_collider_modified"],
                compound_usd["final_or_default_collider_modified"],
                candidate["final_or_default_asset_modified"],
                runtime["final_or_default_asset_modified"],
            )
        ),
        "interpretation": (
            "The supplier-CAD central finger contact region is now proven in finger-link-local "
            "metres through two fresh Isaac 5.1 cooking processes, and the geometry-only USD is "
            "byte-identical across two fresh builds. This does not prove the complete effective "
            "finger contact surface, articulation integration, contact dynamics, calibrated drives, "
            "or material/solver mappings. The diagnostic candidate is therefore not promoted and "
            "the final/default asset remains unchanged."
        ),
        "evidence_boundaries": {
            "static_cooking_only": True,
            "timeline_started": cooking["timeline_started"],
            "video_required": False,
            "video_reason": "NOT_APPLICABLE_STATIC_COOKING_ONLY",
            "diagnostic_asset_redistribution": "UNKNOWN_HARD_BLOCKER_SUPPLIER_CAD_LICENSE",
        },
    }
    closure["deterministic_signature"] = hashlib.sha256(
        json.dumps(closure, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return closure


def _markdown(report: dict[str, Any]) -> str:
    blocker_lines = [
        f"- `{record['blocker']['id']}`: {record['blocker']['missing_definition']}"
        for record in report["formal_parameter_blockers"]
    ]
    return "\n".join(
        [
            "# ALOHA1 official model-first closure",
            "",
            f"- Status: **{report['status']}**",
            f"- Compound central contact region: **{report['compound_contact_region_status']}**",
            f"- Coordinate frame: `{report['compound_contact_region_coordinate_frame']}`",
            f"- Full effective contact surface: **{report['compound_full_face_scope']}**",
            f"- Geometry-only USD: **{report['compound_usd_status']}**",
            f"- Articulation integration: **{report['compound_articulation_integration_status']}**",
            f"- Contact dynamics: **{report['compound_contact_dynamics_status']}**",
            f"- Official model candidate: **{report['official_model_candidate_status']}**",
            f"- Task 8: **{report['task8_status']}**",
            f"- Final/default asset modified: `{report['final_or_default_asset_modified']}`",
            "",
            "## Verified boundary",
            "",
            report["interpretation"],
            "",
            "This was a static collision-cooking and geometry-authoring check. No timeline was "
            "started, so a video would not provide additional physical evidence. The rejected "
            "exact-ray attempt and its annotated numerical-quantization screenshot remain part "
            "of the evidence trail.",
            "",
            "## Remaining formal blockers",
            "",
            *blocker_lines,
            "",
            "The supplier-derived diagnostic geometry remains local-only while redistribution "
            "rights are unknown.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    report = build_closure(args.root)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "remaining_blocker_count": report["remaining_blocker_count"],
                "final_or_default_asset_modified": report["final_or_default_asset_modified"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
