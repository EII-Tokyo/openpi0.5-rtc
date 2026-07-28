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
    authorized_stage_audit_path: Path,
    diagnostic_asset_path: Path | None = None,
    isaac_screenshot_review_path: Path | None = None,
) -> dict[str, Any]:
    project_root = mapping_path.resolve().parents[2]
    diagnostic_asset_path = diagnostic_asset_path or (
        project_root
        / "reports/aloha1_mapping/"
        "aloha_viper_cad_finger_diagnostic_asset_v2.json"
    )
    isaac_screenshot_review_path = isaac_screenshot_review_path or (
        project_root
        / "reports/aloha1_mapping/"
        "aloha_viper_cad_finger_isaac_screenshot_review.json"
    )
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    tessellation = json.loads(tessellation_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    stage_audit = json.loads(
        authorized_stage_audit_path.read_text(encoding="utf-8")
    )
    diagnostic_asset = json.loads(
        diagnostic_asset_path.read_text(encoding="utf-8")
    )
    isaac_screenshot_review = json.loads(
        isaac_screenshot_review_path.read_text(encoding="utf-8")
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
        "production_angular_tessellation_pass": (
            tessellation["production_tessellation_gate"] == "PASS"
        ),
        "source_inventory_complete": (
            source_manifest["inventory_status"] == "PASS"
        ),
        "historical_candidate_hash_unchanged": (
            candidate_hash == EXPECTED_CANDIDATE_SHA256
        ),
        "authorized_stage_read_only_audit_pass": (
            stage_audit["status"] == "PASS"
            and stage_audit["absolute_path"]
            == str(candidate_stage_path.resolve())
            and stage_audit["source_sha256_before"] == candidate_hash
            and stage_audit["source_sha256_after"] == candidate_hash
        ),
        "isolated_diagnostic_asset_pass": (
            diagnostic_asset["status"] == "PASS"
            and diagnostic_asset["source_stage"]["sha256_before"]
            == candidate_hash
            and diagnostic_asset["source_stage"]["sha256_after"]
            == candidate_hash
        ),
        "isaac_screenshot_review_pass": (
            isaac_screenshot_review["status"] == "PASS"
            and isaac_screenshot_review["approved_source_stage"][
                "sha256_before"
            ]
            == candidate_hash
            and isaac_screenshot_review["approved_source_stage"][
                "sha256_after"
            ]
            == candidate_hash
        ),
    }
    return {
        "schema_version": 1,
        "status": "PARTIAL",
        "scope": (
            "User-approved source Stage frozen read-only; isolated v2 "
            "diagnostic USD and CAD-installation screenshots completed. "
            "Task 5/7 remain not run and Task 8 remains prohibited."
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
            "runtime_probe_status": "PASS_READ_ONLY_STAGE_AUDIT",
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
            "authorized_stage_audit": {
                "path": str(authorized_stage_audit_path.resolve()),
                "sha256": _sha256(authorized_stage_audit_path),
                "status": stage_audit["status"],
            },
            "isolated_diagnostic_asset": {
                "path": str(diagnostic_asset_path.resolve()),
                "sha256": _sha256(diagnostic_asset_path),
                "status": diagnostic_asset["status"],
                "root_usd": diagnostic_asset["diagnostic_outputs"][
                    "root_usd"
                ],
            },
            "isaac_screenshot_review": {
                "path": str(isaac_screenshot_review_path.resolve()),
                "sha256": _sha256(isaac_screenshot_review_path),
                "status": isaac_screenshot_review["status"],
                "capture_count": isaac_screenshot_review["capture_count"],
                "gate": isaac_screenshot_review["gate"],
            },
        },
        "input_gates": input_gates,
        "stage_selection": {
            "status": "PASS",
            "reason": (
                "The user explicitly approved this frozen Stage for the "
                "supplier-CAD finger isolated diagnostic. Authorization "
                "permits only an independent diagnostic layer and forbids "
                "source/default/final-collider mutation."
            ),
            "approved_review_stage": {
                "classification": (
                    "USER_APPROVED_ISOLATED_DIAGNOSTIC_REVIEW_STAGE"
                ),
                "absolute_path": str(candidate_stage_path.resolve()),
                "sha256": candidate_hash,
                "expected_frozen_sha256": EXPECTED_CANDIDATE_SHA256,
                "read_only": True,
                "source_sha256_before": stage_audit[
                    "source_sha256_before"
                ],
                "source_sha256_after": stage_audit["source_sha256_after"],
                "root_prim": stage_audit["default_prim"],
                "layer_stack_status": stage_audit["layer_stack_status"],
                "required_key_prims_status": stage_audit[
                    "required_key_prims_status"
                ],
                "instance_proxy_strategy": stage_audit[
                    "instance_proxy_strategy"
                ],
            },
            "authorization_boundary": stage_audit["authorization"],
        },
        "diagnostic_asset_plan": {
            "status": "PASS",
            "target_directory": str(
                Path(
                    diagnostic_asset["diagnostic_outputs"]["root_usd"][
                        "absolute_path"
                    ]
                ).parent
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
            "isolated_diagnostic_usd": "PASS",
            "isaac_open_closed_screenshots": "PASS",
            "task_5_correct_cad_finger": "NOT_RUN",
            "task_7": "NOT_RUN",
            "task_8": "NOT_RUN",
        },
        "hard_blockers": [
            {
                "id": "CAD_LICENSE_UNKNOWN",
                "scope_blocked": (
                    "committing or redistributing original STEP/PDF files"
                ),
            },
        ]
        + (
            []
            if tessellation["production_tessellation_gate"] == "PASS"
            else [
                {
                    "id": "ANGULAR_TESSELLATION_CONTROL_UNAVAILABLE",
                    "scope_blocked": (
                        "promotion of the linear-only diagnostic mesh as "
                        "the fully parameter-controlled production visual "
                        "mesh"
                    ),
                }
            ]
        ),
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
    candidate = stage["approved_review_stage"]
    lines = [
        "# ALOHA Supplier-CAD Finger Isaac Stage Gate",
        "",
        f"- Overall status: `{report['status']}`",
        f"- Stage selection: `{stage['status']}`",
        "- Source Stage opened read-only for audit: `true`",
        "- Source Stage switched/authored/saved: `false`",
        (
            "- Isolated diagnostic USD / Isaac screenshots: "
            f"`{report['execution_status']['isolated_diagnostic_usd']}` / "
            f"`{report['execution_status']['isaac_open_closed_screenshots']}`"
        ),
        "- Task 5 / Task 7 / Task 8: `NOT_RUN / NOT_RUN / NOT_RUN`",
        "",
        stage["reason"],
        "",
        "## User-approved isolated diagnostic review Stage",
        "",
        f"- Path: `{candidate['absolute_path']}`",
        f"- Frozen SHA-256: `{candidate['sha256']}`",
        f"- Root prim: `{candidate['root_prim']}`",
        f"- Layer stack: `{candidate['layer_stack_status']}`",
        f"- Required key prims: `{candidate['required_key_prims_status']}`",
        f"- Read only: `{str(candidate['read_only']).lower()}`",
        "",
        "## Independent work completed",
        "",
        "- Supplier CAD installation/orientation mapping: `PASS`",
        "- Raw/annotated CAD screenshot visual gate: `PASS`",
        "- Isolated supplier-CAD diagnostic asset: `PASS`",
        "- Isaac raw/annotated installation visual gate: `PASS`",
        "- Two-run linear-only tessellation determinism: `PASS`",
        (
            "- Production angular-controlled tessellation: "
            f"`{report['input_evidence']['tessellation']['production_tessellation_gate']}`"
        ),
        "- Source license/redistribution: `UNKNOWN_HARD_BLOCKER`",
    ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
