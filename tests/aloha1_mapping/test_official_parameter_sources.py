from __future__ import annotations

import json
from pathlib import Path

from tools.aloha1_mapping.official_parameter_sources import REQUIRED_SOURCE_IDS
from tools.aloha1_mapping.official_parameter_sources import build_source_audit
from tools.aloha1_mapping.official_parameter_sources import load_source_manifest
from tools.aloha1_mapping.official_parameter_sources import validate_source_manifest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs/aloha1_official_parameter_sources.yaml"
REPORT = ROOT / "reports/aloha1_mapping/aloha1_official_parameter_source_audit.json"


def _valid_source(source_id: str) -> dict[str, object]:
    return {
        "id": source_id,
        "authority": "Trossen Robotics",
        "evidence_class": "OFFICIAL_DIRECT",
        "url": "https://docs.trossenrobotics.com/exact",
        "retrieved_at_utc": "2026-08-02T00:00:00Z",
        "local_path": "/tmp/exact-source.html",
        "sha256": "a" * 64,
        "license": {
            "status": "LINK_AND_FACT_CITATION_ONLY",
            "identifier": "UNKNOWN",
            "evidence": "https://docs.trossenrobotics.com/",
            "redistribution": "NOT_REDISTRIBUTED",
        },
        "exact_model_scope": ["aloha_vx300s"],
    }


def _valid_manifest() -> dict[str, object]:
    sources = [_valid_source(source_id) for source_id in sorted(REQUIRED_SOURCE_IDS)]
    for source in sources:
        if source["id"] in {
            "interbotix_manipulators_humble",
            "interbotix_core_humble",
        }:
            source.update(
                {
                    "evidence_class": "OFFICIAL_PINNED_SOURCE",
                    "repository": "https://github.com/Interbotix/example.git",
                    "branch": "humble",
                    "commit": "b" * 40,
                    "license": {
                        "status": "CONFIRMED",
                        "identifier": "BSD-3-Clause",
                        "evidence": "LICENSE",
                        "redistribution": "ALLOWED_WITH_LICENSE",
                    },
                }
            )
    return {
        "schema_version": 1,
        "product": {
            "project_model": "aloha_vx300s",
            "manufacturer": "Trossen Robotics",
            "product": "Interbotix ViperX-300 6DOF",
            "follower_instances": ["follower_left", "follower_right"],
            "robot_local_geometry_relation": "IDENTICAL_NOT_MIRRORED",
        },
        "sources": sources,
        "source_conflicts": [
            {
                "id": "trossen_vx300s_servo_id_6_7_joint_name",
                "status": "RESOLVED_WITH_CONFLICT_RETAINED",
                "conflicting_claims": [
                    {"source_id": "trossen_vx300s_spec", "id6": "wrist_angle", "id7": "forearm_roll"},
                    {"source_id": "interbotix_vx300s_motor_config", "id6": "forearm_roll", "id7": "wrist_angle"},
                ],
                "resolution": {
                    "id6": "forearm_roll",
                    "id7": "wrist_angle",
                    "basis_source_ids": [
                        "interbotix_vx300s_motor_config",
                        "interbotix_aloha_vx300s_motor_config",
                        "interbotix_vx300s_xacro",
                    ],
                    "does_not_erase_conflict": True,
                },
            }
        ],
    }


def test_required_source_set_covers_exact_product_components_and_local_isaac() -> None:
    assert {
        "trossen_vx300s_spec",
        "robotis_xm540_w270_manual",
        "robotis_xm430_w350_manual",
        "interbotix_manipulators_humble",
        "interbotix_core_humble",
        "interbotix_vx300s_motor_config",
        "interbotix_aloha_vx300s_motor_config",
        "interbotix_vx300s_xacro",
        "interbotix_aloha_vx300s_xacro",
        "interbotix_xs_driver",
        "supplier_simple_aloha_viper_step",
        "isaacsim_urdf_importer_2_4_30",
        "physx_schema_107_3",
    } == REQUIRED_SOURCE_IDS


def test_manifest_rejects_related_model_substitution_and_mutable_repo() -> None:
    manifest = _valid_manifest()
    manifest["product"]["project_model"] = "wx250s"
    repo_source = next(source for source in manifest["sources"] if source["id"] == "interbotix_manipulators_humble")
    repo_source["commit"] = "main"

    findings = validate_source_manifest(manifest, verify_files=False)

    codes = {finding["code"] for finding in findings}
    assert "EXACT_PRODUCT_MISMATCH" in codes
    assert "INVALID_PINNED_COMMIT" in codes


def test_manifest_rejects_missing_provenance_and_unknown_source_id() -> None:
    manifest = _valid_manifest()
    manifest["sources"][0].pop("sha256")
    manifest["sources"].append(_valid_source("generic_viperx_blog"))

    findings = validate_source_manifest(manifest, verify_files=False)

    codes = {finding["code"] for finding in findings}
    assert "MISSING_REQUIRED_FIELD" in codes
    assert "UNAPPROVED_SOURCE_ID" in codes


def test_id67_resolution_requires_multiple_pinned_official_basis_sources() -> None:
    manifest = _valid_manifest()
    manifest["source_conflicts"][0]["resolution"]["basis_source_ids"] = ["interbotix_vx300s_motor_config"]

    findings = validate_source_manifest(manifest, verify_files=False)

    assert any(finding["code"] == "ID67_RESOLUTION_BASIS_INSUFFICIENT" for finding in findings)


def test_repository_manifest_and_generated_audit_are_machine_valid() -> None:
    manifest = load_source_manifest(MANIFEST)
    findings = validate_source_manifest(manifest, repository_root=ROOT, verify_files=True)
    audit = build_source_audit(manifest, findings)

    assert findings == []
    assert audit["status"] == "PASS"
    assert audit["source_chain_completeness"] == "PASS"
    assert audit["formal_parameter_candidate_gate"] == "PASS"
    assert audit["product"]["project_model"] == "aloha_vx300s"
    assert audit["source_count"] == len(REQUIRED_SOURCE_IDS)

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["deterministic_signature"] == audit["deterministic_signature"]
    assert report["status"] == "PASS"


def test_file_hash_verification_detects_content_change(tmp_path: Path) -> None:
    source_file = tmp_path / "source.txt"
    source_file.write_text("official", encoding="utf-8")
    manifest = _valid_manifest()
    source = manifest["sources"][0]
    source["local_path"] = str(source_file)
    source["sha256"] = "0" * 64

    findings = validate_source_manifest(manifest, verify_files=True)

    assert any(finding["code"] == "LOCAL_SHA256_MISMATCH" for finding in findings)
