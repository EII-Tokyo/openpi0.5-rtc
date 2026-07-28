"""Build the immutable Stage gate for the supplier-CAD finger diagnostic."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_CANDIDATE_SHA256 = (
    "b24afe3678155654892c69517fc58ecd970108d68cec56b02dc6fdcb8bf4493e"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_gate_report(
    *,
    mapping_path: Path,
    tessellation_path: Path,
    source_manifest_path: Path,
    candidate_stage_path: Path,
    importer_api_path: Path,
    importer_manifest_path: Path,
) -> dict[str, Any]:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    tessellation = json.loads(tessellation_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    candidate_hash = _sha256(candidate_stage_path)
    input_gates = {
        "cad_installation_mapping_pass": (
            mapping["orientation_mapping_status"] == "PASS"
        ),
        "screenshot_visual_gate_pass": (
            mapping["visual_evidence"]["status"] == "PASS"
        ),
        "two_run_linear_tessellation_deterministic": (
            tessellation["determinism_gate"] == "PASS"
        ),
        "source_inventory_complete": (
            source_manifest["inventory_status"] == "PASS"
        ),
        "historical_candidate_hash_unchanged": (
            candidate_hash == EXPECTED_CANDIDATE_SHA256
        ),
    }
    return {
        "schema_version": 1,
        "status": "PARTIAL",
        "scope": (
            "Static authorization and source gate only; no USD Stage was "
            "opened, switched, authored, or simulated"
        ),
        "official_isaac_gateway_evidence": {
            "status": "PASS",
            "gateway": "mcpjungle_lab",
            "provider": "NVIDIA official Isaac documentation MCP",
            "queried_instruction_sets": [
                "physics",
                "omniverse_and_usd",
                "importers_and_exporters",
                "asset_structure",
                "api_documentation",
            ],
            "applicable_guidance": [
                (
                    "USD Physics and PhysxSchema APIs must be applied/read "
                    "through schema objects rather than guessed attributes"
                ),
                (
                    "source assets remain unchanged and physics/configuration "
                    "features belong in separate layers"
                ),
                (
                    "visual geometry and collision geometry are independent"
                ),
            ],
            "version_boundary": (
                "Gateway documentation is discovery evidence only; exact API "
                "names and defaults are pinned to local Isaac Sim 5.1 source "
                "and runtime probes"
            ),
        },
        "local_isaac_5_1_evidence": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
            "urdf_importer": "2.4.30",
            "importer_api_path": str(importer_api_path.resolve()),
            "importer_api_sha256": _sha256(importer_api_path),
            "importer_manifest_path": str(importer_manifest_path.resolve()),
            "importer_manifest_sha256": _sha256(importer_manifest_path),
            "confirmed_import_config_member": "convex_decomp",
            "confirmed_command": "URDFCreateImportConfig",
            "runtime_probe_status": "NOT_RUN_STAGE_NOT_AUTHORIZED",
        },
        "input_evidence": {
            "mapping": {
                "path": str(mapping_path.resolve()),
                "sha256": _sha256(mapping_path),
                "status": mapping["status"],
            },
            "tessellation": {
                "path": str(tessellation_path.resolve()),
                "sha256": _sha256(tessellation_path),
                "status": tessellation["status"],
                "production_tessellation_gate": tessellation[
                    "production_tessellation_gate"
                ],
            },
            "source_manifest": {
                "path": str(source_manifest_path.resolve()),
                "sha256": _sha256(source_manifest_path),
                "status": source_manifest["status"],
                "license_status": source_manifest["license"]["status"],
            },
        },
        "input_gates": input_gates,
        "stage_selection": {
            "status": "HARD_BLOCKER",
            "reason": (
                "No absolute Stage path has been user-approved for the "
                "post-2026-07-29 supplier-CAD finger diagnostic. The known "
                "file is historical evidence, not authorization to load or "
                "mutate it for this task."
            ),
            "historical_candidate": {
                "classification": (
                    "HISTORICAL_CANDIDATE_NOT_AUTHORIZED_CURRENT_TASK"
                ),
                "absolute_path": str(candidate_stage_path.resolve()),
                "sha256": candidate_hash,
                "expected_frozen_sha256": EXPECTED_CANDIDATE_SHA256,
                "root_prim": "UNVERIFIED_NOT_LOADED",
                "sublayers": "UNVERIFIED_NOT_LOADED",
                "required_key_prims": "UNVERIFIED_NOT_LOADED",
            },
            "required_to_clear": [
                "user-approved absolute Stage path",
                "frozen SHA-256",
                "expected root prim",
                "expected sublayers",
                "required follower/gripper prim paths",
            ],
        },
        "diagnostic_asset_plan": {
            "status": "STATIC_PREPARED_NOT_AUTHORED",
            "target_directory": str(
                mapping_path.resolve().parents[2]
                / "assets/Trossen/ALOHA1/1.0/diagnostics/"
                "cad_finger_installation"
            ),
            "source_fingers": {
                "left_finger": (
                    "Simple Viper embedded Part__Feature007 / CAD +X"
                ),
                "right_finger": (
                    "Simple Viper embedded Part__Feature008 / CAD -X"
                ),
            },
            "visual_mesh_policy": (
                "diagnostic only; never auto-promote to collider"
            ),
            "collision_mesh_policy": (
                "NOT_SELECTED; final/default collider remains unchanged"
            ),
            "composition_policy": (
                "separate diagnostic geometry/configuration layer; do not "
                "overwrite source USD or final/default configuration"
            ),
        },
        "execution_status": {
            "isolated_diagnostic_usd": "NOT_RUN",
            "isaac_open_closed_screenshots": "NOT_RUN",
            "task_5_correct_cad_finger": "NOT_RUN",
            "task_7": "NOT_RUN",
            "task_8": "NOT_RUN",
        },
        "hard_blockers": [
            {
                "id": "ISAAC_REVIEW_STAGE_NOT_USER_APPROVED",
                "scope_blocked": (
                    "Stage load/switch, USD authoring against a selected "
                    "follower, Isaac screenshots, Task 5, and Task 7"
                ),
            },
            {
                "id": "CAD_LICENSE_UNKNOWN",
                "scope_blocked": (
                    "committing or redistributing original STEP/PDF files"
                ),
            },
            {
                "id": "ANGULAR_TESSELLATION_CONTROL_UNAVAILABLE",
                "scope_blocked": (
                    "promotion of the linear-only diagnostic mesh as the "
                    "fully parameter-controlled production visual mesh"
                ),
            },
        ],
    }


def write_gate_report(
    report: Mapping[str, Any],
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    stage = report["stage_selection"]
    candidate = stage["historical_candidate"]
    lines = [
        "# ALOHA Supplier-CAD Finger Isaac Stage Gate",
        "",
        f"- Overall status: `{report['status']}`",
        f"- Stage selection: `{stage['status']}`",
        "- Stage opened/switched/authored: `false`",
        "- Task 5 / Task 7 / Task 8: `NOT_RUN / NOT_RUN / NOT_RUN`",
        "",
        stage["reason"],
        "",
        "## Historical candidate (not authorized)",
        "",
        f"- Path: `{candidate['absolute_path']}`",
        f"- Frozen SHA-256: `{candidate['sha256']}`",
        f"- Root prim: `{candidate['root_prim']}`",
        f"- Sublayers: `{candidate['sublayers']}`",
        f"- Required key prims: `{candidate['required_key_prims']}`",
        "",
        "## Independent work completed",
        "",
        "- Supplier CAD installation/orientation mapping: `PASS`",
        "- Raw/annotated CAD screenshot visual gate: `PASS`",
        "- Two-run linear-only tessellation determinism: `PASS`",
        "- Production angular-controlled tessellation: `HARD_BLOCKER`",
        "- Source license/redistribution: `UNKNOWN_HARD_BLOCKER`",
    ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
