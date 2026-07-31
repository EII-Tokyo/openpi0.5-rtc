#!/usr/bin/env python3
"""Build the evidence-linked post-grasp ALOHA1 Task 7 report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tools.aloha1_mapping.task7_post_grasp_acceptance import classify_post_grasp_task7

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/aloha1_mapping"

SOURCE_STAGE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/1.0"
    / "aloha1_signal_correspondence_workcell.usda"
)
SOURCE_STAGE_SHA256 = (
    "d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf"
)
ALIGNED_STAGE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/table_support_alignment/1.0"
    / "aloha1_table_support_aligned_workcell.usda"
)
ALIGNED_STAGE_SHA256 = (
    "2b3f76365ed67532f478d995ae859a88b5639975ac07cb7ac8a53ac679e8205c"
)

INPUTS = {
    "task7_runtime": REPORT_ROOT / "aloha1_task7_runtime_acceptance.json",
    "ik_correspondence": (
        REPORT_ROOT / "aloha1_ik_correspondence_v3.json"
    ),
    "table_alignment": (
        REPORT_ROOT / "aloha1_table_support_alignment_validation.json"
    ),
    "static_hold": (
        REPORT_ROOT / "aloha1_task7b_bottle_geometry_ab.json"
    ),
    "dynamic_grasp": (
        REPORT_ROOT
        / "aloha1_grasp_20cm_five_pose_downward_acceptance_v6.json"
    ),
    "asset_promotion": (
        REPORT_ROOT / "aloha1_task7_asset_promotion_readiness.json"
    ),
    "official_applicability": (
        REPORT_ROOT / "aloha1_task7_official_rule_applicability.json"
    ),
}

OUTPUT_JSON = REPORT_ROOT / "aloha1_task7_post_grasp_acceptance.json"
OUTPUT_MD = REPORT_ROOT / "aloha1_task7_post_grasp_acceptance.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _assert_stage_identity(
    task7_runtime: dict[str, Any],
    alignment: dict[str, Any],
    dynamic_grasp: dict[str, Any],
) -> dict[str, Any]:
    source_hash = _sha256(SOURCE_STAGE.resolve(strict=True))
    aligned_hash = _sha256(ALIGNED_STAGE.resolve(strict=True))
    if source_hash != SOURCE_STAGE_SHA256:
        raise ValueError("frozen signal-correspondence Stage hash changed")
    if aligned_hash != ALIGNED_STAGE_SHA256:
        raise ValueError("frozen table-aligned Stage hash changed")

    diagnostic = alignment["diagnostic_stage"]
    source = alignment["source_stage"]
    expected_source_sublayer = (
        "../../signal_correspondence/1.0/"
        "aloha1_signal_correspondence_workcell.usda"
    )
    checks = {
        "task7_runtime_uses_source_stage": (
            task7_runtime["stage"]["sha256"] == SOURCE_STAGE_SHA256
        ),
        "alignment_source_before_unchanged": (
            source["sha256_before"] == SOURCE_STAGE_SHA256
        ),
        "alignment_source_after_unchanged": (
            source["sha256_after"] == SOURCE_STAGE_SHA256
        ),
        "alignment_output_matches_frozen_stage": (
            diagnostic["sha256"] == ALIGNED_STAGE_SHA256
        ),
        "alignment_sublayers_source_stage": (
            expected_source_sublayer in diagnostic["sublayers"]
        ),
        "dynamic_grasp_uses_aligned_stage": (
            dynamic_grasp["stage"]["sha256"] == ALIGNED_STAGE_SHA256
        ),
        "table_translation_only": (
            alignment["boundaries"]["table_translation_only"] is True
        ),
        "source_stage_not_modified": (
            alignment["boundaries"]["source_stage_modified"] is False
        ),
        "robot_geometry_not_modified": (
            alignment["boundaries"]["robot_geometry_modified"] is False
        ),
        "collider_or_physics_not_modified": (
            alignment["boundaries"][
                "collider_or_physics_parameters_modified"
            ]
            is False
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Stage-composition checks failed: {failed}")
    return {
        "source": {
            "absolute_path": str(SOURCE_STAGE.resolve()),
            "sha256": source_hash,
        },
        "aligned": {
            "absolute_path": str(ALIGNED_STAGE.resolve()),
            "sha256": aligned_hash,
            "root_prim": diagnostic["default_prim"],
            "sublayers": diagnostic["sublayers"],
        },
        "composition_checks": checks,
        "composition_verified": True,
    }


def _assert_dynamic_grasp(dynamic: dict[str, Any]) -> None:
    checks = {
        "status": dynamic.get("status") == "PASS",
        "machine_status": dynamic.get("machine_status") == "PASS",
        "machine_pass_count": dynamic.get("machine_pass_count") == 5,
        "evidence_pass_count": dynamic.get("evidence_pass_count") == 5,
        "visual_model_review": dynamic.get("visual_model_review") == "PASS",
        "user_confirmation": dynamic.get("user_confirmation") == "PASS",
        "sample_count": len(dynamic.get("samples", [])) == 5,
        "task8": dynamic.get("boundaries", {}).get("task8") == "NOT_RUN",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"dynamic-grasp evidence checks failed: {failed}")


def _assert_ik_correspondence(ik_report: dict[str, Any]) -> None:
    checks = {
        "status": ik_report.get("status") == "PASS",
        "aloha_6dof_correspondence": (
            ik_report.get("aloha_6dof_correspondence") == "PASS"
        ),
        "ik": ik_report.get("ik") == "PASS",
        "task8": ik_report.get("task8") == "NOT_RUN",
        "current_horizontal_config": (
            ik_report["bindings"]["horizontal_grasp_config"]["sha256"]
            == _sha256(
                (
                    ROOT / "configs/aloha1_task7b2_horizontal_grasp.yaml"
                ).resolve(strict=True)
            )
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"ALOHA IK correspondence checks failed: {failed}")


def _markdown(report: dict[str, Any]) -> str:
    gates = report["runtime_grasp_gates"]
    lines = [
        "# ALOHA1 Task 7 post-grasp acceptance",
        "",
        f"- Runtime/grasp acceptance: `{report['runtime_grasp_acceptance']}`",
        f"- Asset-promotion readiness: `{report['asset_promotion_readiness']}`",
        (
            "- Literal NVIDIA official-rule status: "
            f"`{report['official_rules_literal_status']}`"
        ),
        f"- Task 7 aggregate: `{report['task7_aggregate']}`",
        f"- Task 8: `{report['task8']}`",
        "",
        "## Runtime and grasp gates",
        "",
    ]
    lines.extend(f"- `{name}`: `{status}`" for name, status in gates.items())
    lines.extend(
        [
            "",
            "The five-pose grasp is machine, visual-model and user `PASS`. "
            "This does not make the robot package SimReady. The Task 7 "
            "aggregate remains `PARTIAL` because literal NVIDIA rule "
            "findings keep asset-promotion readiness `PARTIAL`.",
            "",
            "The table-aligned Stage composes the frozen signal Stage and "
            "changes only tabletop translation. Robot geometry, colliders, "
            "drives and physics parameters are unchanged.",
            "",
            "No real robot or 192.168.1.103 access occurred. Task 8 remains "
            "`NOT_RUN`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    before_hashes = {
        name: _sha256(path.resolve(strict=True))
        for name, path in INPUTS.items()
    }
    reports = {name: _load(path) for name, path in INPUTS.items()}

    task7_runtime = reports["task7_runtime"]
    alignment = reports["table_alignment"]
    static_hold = reports["static_hold"]
    dynamic_grasp = reports["dynamic_grasp"]
    ik_correspondence = reports["ik_correspondence"]
    promotion = reports["asset_promotion"]
    official = reports["official_applicability"]

    _assert_dynamic_grasp(dynamic_grasp)
    _assert_ik_correspondence(ik_correspondence)
    stage = _assert_stage_identity(
        task7_runtime,
        alignment,
        dynamic_grasp,
    )
    if official.get("official_status_suppressed") is not False:
        raise ValueError("literal NVIDIA result was suppressed")
    if official.get("unclassified_issue_count") != 0:
        raise ValueError("official-rule applicability has unclassified issues")

    classified = classify_post_grasp_task7(
        {
            "runtime_control": task7_runtime["runtime_control"]["status"],
            "workcell_physics": task7_runtime["workcell_physics"]["status"],
            "aloha_6dof_ik_correspondence": ik_correspondence[
                "aloha_6dof_correspondence"
            ],
            "table_support_alignment": alignment["status"],
            "static_bottle_hold": static_hold["status"],
            "dynamic_five_pose_grasp": dynamic_grasp["status"],
            "visual_model_review": dynamic_grasp["visual_model_review"],
            "user_confirmation": dynamic_grasp["user_confirmation"],
            "asset_promotion_readiness": promotion["status"],
            "official_rules_literal_status": official["official_status"],
            "task8": "NOT_RUN",
        }
    )
    after_hashes = {
        name: _sha256(path.resolve(strict=True))
        for name, path in INPUTS.items()
    }
    if before_hashes != after_hashes:
        raise RuntimeError("an input report changed during aggregation")

    report = {
        "schema_version": 1,
        **classified,
        "official_rules_suppressed": False,
        "official_rule_issue_count": official["issue_count"],
        "stage": stage,
        "evidence": {
            name: {
                "absolute_path": str(INPUTS[name].resolve()),
                "sha256": before_hashes[name],
                "status": reports[name].get("status"),
            }
            for name in INPUTS
        },
        "dynamic_grasp": {
            "machine_pass_count": dynamic_grasp["machine_pass_count"],
            "evidence_pass_count": dynamic_grasp["evidence_pass_count"],
            "visual_model_review": dynamic_grasp["visual_model_review"],
            "user_confirmation": dynamic_grasp["user_confirmation"],
            "confirmed_video_sha256": dynamic_grasp[
                "user_confirmation_evidence"
            ]["confirmed_annotated_video_sha256"],
        },
        "input_immutability": {
            "before_sha256": before_hashes,
            "after_sha256": after_hashes,
            "all_hashes_unchanged": True,
        },
        "boundaries": {
            "report_only": True,
            "source_stage_modified": False,
            "aligned_stage_modified": False,
            "final_asset_promoted": False,
            "real_robot": False,
            "remote_103": False,
            "task8": "NOT_RUN",
        },
    }
    _write_json(OUTPUT_JSON, report)
    OUTPUT_MD.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["task7_aggregate"],
                "runtime_grasp_acceptance": report[
                    "runtime_grasp_acceptance"
                ],
                "asset_promotion_readiness": report[
                    "asset_promotion_readiness"
                ],
                "output": str(OUTPUT_JSON.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
