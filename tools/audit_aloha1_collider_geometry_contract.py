#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = DEFAULT_ROOT / "reports/aloha1_mapping/aloha1_collider_geometry_contract.json"
DEFAULT_MARKDOWN = DEFAULT_ROOT / "reports/aloha1_mapping/aloha1_collider_geometry_contract.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(root: Path, relative: str) -> tuple[Path, dict[str, Any]]:
    path = root / relative
    return path, json.loads(path.read_text(encoding="utf-8"))


def build_contract(root: Path) -> dict[str, object]:
    geometry_path, geometry = _load(root, "reports/aloha1_mapping/aloha1_cad_derived_collider_geometry.json")
    semantics_path, semantics = _load(root, "reports/aloha1_mapping/aloha1_cad_link_collision_semantics.json")
    swept_path, swept = _load(root, "reports/aloha1_mapping/aloha1_cad_derived_five_pose_swept_collision.json")
    static_path, static = _load(root, "reports/aloha1_mapping/aloha1_cad_derived_collision_replan_static.json")
    resolution_path, resolution = _load(
        root,
        "reports/aloha1_mapping/aloha1_cad_link_identity_resolution.json",
    )
    certificate_path, certificate = _load(
        root,
        "reports/aloha1_mapping/aloha1_official_collider_surface_certificate.json",
    )
    finger_brep_path, finger_brep = _load(
        root,
        "reports/aloha1_mapping/aloha1_supplier_cad_finger_brep_cooked_certificate.json",
    )
    compound_runtime_path, compound_runtime = _load(
        root,
        "reports/aloha1_mapping/aloha1_supplier_cad_compound_runtime_cooking_certificate.json",
    )
    compound_usd_path, compound_usd = _load(
        root,
        "reports/aloha1_mapping/aloha1_supplier_cad_compound_contact_usd.json",
    )
    if resolution["status"] != "PASS":
        raise ValueError("CAD/link source resolution must pass")
    historical_identity_blockers = geometry["identity_blockers"]
    historical_invalid_brep = geometry["invalid_brep_blockers"]
    unresolved_suffixes: list[str] = []
    input_records = [
        {"path": str(path.resolve()), "sha256": _sha256(path), "status": data["status"]}
        for path, data in (
            (geometry_path, geometry),
            (semantics_path, semantics),
            (swept_path, swept),
            (static_path, static),
            (resolution_path, resolution),
            (certificate_path, certificate),
            (finger_brep_path, finger_brep),
            (compound_runtime_path, compound_runtime),
            (compound_usd_path, compound_usd),
        )
    ]
    contract: dict[str, object] = {
        "schema_version": 1,
        "status": "PARTIAL",
        "source_cad": geometry["source_cad"],
        "toolchain": {
            "freecad_version": geometry["toolchain"]["freecad_version"],
            "opencascade_version": geometry["toolchain"]["opencascade_version"],
            "linear_deflection_mm": geometry["toolchain"]["linear_deflection_mm"],
            "angular_deflection_deg": geometry["toolchain"]["angular_deflection_deg"],
        },
        "input_reports": input_records,
        "two_fresh_directory_determinism": geometry["two_fresh_directory_determinism"],
        "existing_swept_collision_gate": swept["status"],
        "existing_static_collision_gate": static["status"],
        "unresolved_identity_blocker_count": 0,
        "unresolved_link_suffixes": unresolved_suffixes,
        "link_identity_resolution": resolution["status"],
        "resolved_source_boundary_records": resolution["records"],
        "historical_identity_blockers": historical_identity_blockers,
        "historical_invalid_brep_blockers": historical_invalid_brep,
        "identity_blockers": [],
        "invalid_brep_blockers": [],
        "surface_error_certificate": certificate["surface_error_certificate"],
        "surface_error_acceptance": certificate["acceptance_status"],
        "surface_certificate_link_count": certificate["summary"]["link_count"],
        "surface_certificate_summary": certificate["summary"],
        "supplier_finger_exact_brep_gate": finger_brep["comparison"]["exact_surface_status"],
        "supplier_finger_decomposition_comparison": finger_brep["comparison"]["decomposition_comparison"],
        "supplier_finger_exact_asset_decision": finger_brep["comparison"]["asset_decision"],
        "supplier_finger_task_local_acceptance": finger_brep["task_local_approximation_tolerance"],
        "supplier_finger_exact_numeric_tolerance_m": finger_brep["comparison_numeric_tolerance_m"],
        "supplier_finger_compound_runtime_status": compound_runtime["status"],
        "supplier_finger_compound_coordinate_frame": compound_runtime["coordinate_frame"],
        "supplier_finger_compound_contact_region_gates": {
            side: record["contact_region_status"] for side, record in compound_runtime["fingers"].items()
        },
        "supplier_finger_compound_full_face_scope": compound_runtime["fingers"]["left"]["full_face_scope"],
        "supplier_finger_compound_usd_status": compound_usd["status"],
        "supplier_finger_compound_usd_piece_count": compound_usd["readback"]["collision_piece_count"],
        "supplier_finger_compound_asset_decision": compound_usd["asset_decision"],
        "formal_candidate_gate": "BLOCKED",
        "final_or_default_asset_modified": False,
        "interpretation": "The former bar/prop/wrist identity blockers are resolved by explicit source boundaries: the supplier STEP remains fused/invalid where observed, while byte-identical pinned Interbotix link meshes provide robot-description geometry. Every physical link has a deterministic finite-sample convex-hull surface/volume certificate. Exact trimmed B-Rep sampling proves that both the current single hull and default 32-piece decomposition cross the inward CAD face; neither is promoted. A CAD-derived compound candidate now passes two fresh finger-link-local PhysX cooking runs for its central contact rectangle, and its 68-piece geometry-only USD is byte-deterministic. This resolves the local crossing construction without fitting successful grasp data, but the candidate explicitly covers only the central contact region and is not integrated into an articulation. The contract remains PARTIAL because full effective contact-surface/task-local acceptance and remaining physical dynamics mappings are not yet proven.",
    }
    contract["deterministic_signature"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return contract


def _markdown(contract: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# ALOHA1 collider geometry contract",
            "",
            f"- Status: **{contract['status']}**",
            f"- Deterministic tessellation: **{contract['two_fresh_directory_determinism']}**",
            f"- Existing swept collision gate: **{contract['existing_swept_collision_gate']}**",
            f"- Unresolved CAD/link records: `{contract['unresolved_identity_blocker_count']}`",
            f"- Unresolved suffixes: `{contract['unresolved_link_suffixes']}`",
            f"- Formal candidate gate: **{contract['formal_candidate_gate']}**",
            f"- Exact supplier-finger B-Rep gate: **{contract['supplier_finger_exact_brep_gate']}**",
            f"- Default decomposition comparison: **{contract['supplier_finger_decomposition_comparison']}**",
            f"- Compound central contact region: **{contract['supplier_finger_compound_runtime_status']}**",
            f"- Compound geometry-only USD: **{contract['supplier_finger_compound_usd_status']}**",
            "",
            "The supplier STEP remains authoritative for the geometry it exposes, and its fused "
            "gripper/invalid wrist boundaries are preserved. Byte-identical pinned Interbotix "
            "meshes supply the link-level identities. Every physical link now has a numerical "
            "convex-hull surface/volume certificate. Two fresh FreeCAD processes sampled the "
            "exact trimmed finger-pad B-Rep faces, and two fresh Isaac 5.1 processes cooked both "
            "single hull and default decomposition. All four profiles cross the inward CAD face "
            "beyond the derived numerical floor; decomposition is mixed/worse across handed "
            "sides. A CAD-derived compound candidate then removes inward-plane crossing within "
            "a central contact rectangle in two fresh finger-link-local PhysX cooking runs. Its "
            "68-piece geometry-only USD is deterministic, but full-face scope and articulation "
            "integration remain incomplete. Promotion remains blocked because the effective "
            "task contact surface and remaining physical mappings are not fully proven; "
            "successful grasp videos were not used to fit a tolerance. "
            "Existing static/swept tests remain rejection evidence. "
            "No collider is accepted because a grasp happened to pass, and no final/default asset "
            "was changed.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    contract = build_contract(args.root)
    args.json.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(_markdown(contract), encoding="utf-8")
    print(json.dumps({"status": contract["status"], "candidate_gate": contract["formal_candidate_gate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
