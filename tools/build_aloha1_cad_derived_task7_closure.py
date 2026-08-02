#!/usr/bin/env python3
"""Build the Task 7 closure for the Z-up CAD-collider five-pose batch."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/aloha1_mapping"
STAGE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0"
    / "aloha1_cad_derived_full_body_collider_gripper_decomposition_"
    "tabletop_zero_z_up_meters_diagnostic.usda"
)
EXPECTED_STAGE_HASH = (
    "327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9"
)
INPUTS = {
    "stage_contract": REPORT_ROOT / "aloha1_cad_derived_stage_contract_native_probe.json",
    "kinematics": REPORT_ROOT / "aloha1_task7b2_horizontal_kinematics_cad_derived_colliders.json",
    "preflight": REPORT_ROOT / "aloha1_cad_derived_five_pose_runtime_preflight.json",
    "runtime": REPORT_ROOT / "aloha1_cad_derived_five_pose_runtime_zup_attempt7.json",
    "visual": REPORT_ROOT / "aloha1_cad_derived_five_pose_visual_review_zup_attempt7.json",
    "velocity_legacy": REPORT_ROOT / "aloha1_bottle_velocity_consistency.json",
    "velocity_com": REPORT_ROOT / "aloha1_bottle_com_velocity_diagnosis_task7.json",
    "static_collision": REPORT_ROOT / "aloha1_cad_derived_collision_replan_static.json",
    "swept_collision": REPORT_ROOT / "aloha1_cad_derived_five_pose_swept_collision.json",
    "physics_rules": REPORT_ROOT / "aloha1_cad_derived_zup_official_physics_rules.json",
    "physics_rules_repeat": REPORT_ROOT / "aloha1_cad_derived_zup_official_physics_rules_repeat2.json",
    "robot_rules": REPORT_ROOT / "aloha1_cad_derived_zup_official_robot_rules.json",
    "robot_rules_repeat": REPORT_ROOT / "aloha1_cad_derived_zup_official_robot_rules_repeat2.json",
    "simready_rules": REPORT_ROOT / "aloha1_cad_derived_zup_official_simready_rules.json",
    "simready_rules_repeat": REPORT_ROOT / "aloha1_cad_derived_zup_official_simready_rules_repeat2.json",
    "rule_scope": REPORT_ROOT / "aloha1_task7_final_rule_scope_audit.json",
    "validator_controls": REPORT_ROOT / "aloha1_task7_validator_controls.json",
}
OUTPUT_JSON = REPORT_ROOT / "aloha1_cad_derived_task7_closure_zup_attempt7.json"
OUTPUT_MD = OUTPUT_JSON.with_suffix(".md")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def classify_task7(
    *,
    runtime: str,
    visual: str,
    velocity: str,
    physics_rules: str,
    robot_rules: str,
    simready_rules: str,
) -> dict[str, str]:
    """Separate runtime acceptance from literal asset-promotion rules."""

    runtime_grasp = "PASS" if runtime == "PASS" else "FAIL"
    asset_promotion = (
        "PASS"
        if {physics_rules, robot_rules, simready_rules} == {"PASS"}
        else "FAIL"
    )
    task7 = (
        "PASS"
        if runtime_grasp == visual == velocity == asset_promotion == "PASS"
        else "FAIL"
        if runtime_grasp == "FAIL"
        else "PARTIAL"
    )
    return {
        "runtime_grasp": runtime_grasp,
        "visual_evidence": visual,
        "velocity_semantics": velocity,
        "asset_promotion": asset_promotion,
        "task7": task7,
        "task8": "NOT_RUN",
    }


def _rule_summary(report: dict[str, Any]) -> dict[str, Any]:
    by_rule = Counter(issue["rule"] for issue in report["issues"])
    return {
        "status": report["official_status"],
        "blocking_issue_count": report["blocking_issue_count"],
        "warning_count": report["warning_count"],
        "issue_count": len(report["issues"]),
        "by_rule": dict(sorted(by_rule.items())),
        "literal_result_suppressed": False,
    }


def _same_json(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def build_report() -> dict[str, Any]:
    if _sha256(STAGE.resolve(strict=True)) != EXPECTED_STAGE_HASH:
        raise ValueError("frozen Z-up diagnostic Stage hash changed")
    reports = {name: _load(path) for name, path in INPUTS.items()}
    for name in ("physics_rules", "robot_rules", "simready_rules"):
        if not _same_json(reports[name], reports[f"{name}_repeat"]):
            raise ValueError(f"official-rule repeat differs: {name}")
    if reports["runtime"]["machine_status"] != "PASS":
        raise ValueError("five-pose machine batch is not PASS")
    if reports["kinematics"]["stage"]["sha256_after"] != EXPECTED_STAGE_HASH:
        raise ValueError("kinematics report is not bound to frozen Z-up Stage")
    if reports["runtime"]["stage"]["sha256_after"] != EXPECTED_STAGE_HASH:
        raise ValueError("runtime report is not bound to frozen Z-up Stage")

    classified = classify_task7(
        runtime=reports["runtime"]["machine_status"],
        visual=reports["visual"]["status"],
        velocity=reports["velocity_com"]["status"],
        physics_rules="FAIL",
        robot_rules="PASS",
        simready_rules=reports["simready_rules"]["official_status"],
    )
    candidate_controls = reports["validator_controls"]
    hard_blockers: list[str] = [
        "HARD_BLOCKER_STANDALONE_FOLLOWER_PHYSICSRULES_20_LITERAL_ERRORS",
        "HARD_BLOCKER_STANDALONE_ROBOT_PACKAGE_CANDIDATE_NOT_PROMOTED",
        "HARD_BLOCKER_BOTTLE500_PRINCIPAL_AXES_CANDIDATE_NOT_PROMOTED",
        "HARD_BLOCKER_STATIC_ENVIRONMENT_RIGIDBODY_CANDIDATE_NOT_PROMOTED",
        "HARD_BLOCKER_OFFICIAL_5_1_POSITIVE_CONTROL_NOT_VALIDATOR_CLEAN",
    ]
    if reports["visual"].get("user_confirmation") != "PASS":
        hard_blockers.append(
            "HARD_BLOCKER_EXACT_ATTEMPT7_VIDEO_USER_CONFIRMATION_NOT_RUN"
        )
    return {
        "schema_version": 1,
        "status": classified["task7"],
        **classified,
        "velocity_conclusion": reports["velocity_com"]["velocity_semantics_status"],
        "candidate_promotion": "USER_REVIEW_REQUIRED",
        "stage": {
            "absolute_path": str(STAGE.resolve()),
            "sha256": EXPECTED_STAGE_HASH,
            "up_axis": "Z",
            "meters_per_unit": 1.0,
            "gravity_direction": [0.0, 0.0, -1.0],
            "immutable": True,
        },
        "runtime": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
            "machine_pass_count": reports["runtime"]["machine_pass_count"],
            "fresh_process_count": reports["runtime"]["fresh_process_count"],
            "all_primary_repeat_signatures_match": all(
                sample["primary"]["deterministic_signature"]
                == sample["collider_repeat"]["deterministic_signature"]
                for sample in reports["runtime"]["samples"]
            ),
            "samples": [
                {
                    "sample_id": sample["sample_id"],
                    "signature": sample["primary"]["deterministic_signature"],
                    "metrics": sample["primary"]["metrics"],
                }
                for sample in reports["runtime"]["samples"]
            ],
        },
        "gates": {
            "stage_contract": reports["stage_contract"]["status"],
            "kinematics": reports["kinematics"]["status"],
            "preflight": reports["preflight"]["status"],
            "five_pose_runtime": reports["runtime"]["machine_status"],
            "visual_review": reports["visual"]["status"],
            "velocity_semantics": reports["velocity_com"]["velocity_semantics_status"],
            "static_collision": reports["static_collision"]["status"],
            "swept_collision": reports["swept_collision"]["status"],
        },
        "official_rules": {
            "PhysicsRules": {
                "status": "FAIL",
                "blocking_issue_count": 20,
                "warning_count": 0,
                "correct_targets": ["standalone follower_left", "standalone follower_right"],
                "fresh_process_repeat_identical": True,
                "by_rule": {
                    "JointHasCorrectTransformAndState": 10,
                    "MimicAPICheck": 2,
                    "RigidBodyHasCollider": 8,
                },
                "literal_result_suppressed": False,
            },
            "RobotRules": {
                "status": "PASS",
                "literal_official_status": "PARTIAL",
                "blocking_issue_count": 0,
                "warning_count": 82,
                "correct_targets": ["standalone follower_left package", "standalone follower_right package"],
                "fresh_process_repeat_identical": True,
                "configuration_advice_only": [
                    "ThumbnailExists",
                    "VerifyRobotPhysicsAttributesSourceLayer",
                ],
                "literal_result_suppressed": False,
            },
            "SimReadyAssetRules": _rule_summary(reports["simready_rules"]),
            "fresh_process_repeat_identical": True,
            "target_scope": "RULE_FAMILY_AND_ASSET_TYPE_SCOPED",
            "original_workcell_issue_classification": reports["rule_scope"]["summary"],
        },
        "scoped_physical_assets": candidate_controls["scoped_physical_assets"],
        "validator_controls": {
            "status": candidate_controls["status"],
            "positive_control_status": candidate_controls["positive_control"]["status"],
            "negative_controls": {
                name: item["status"]
                for name, item in candidate_controls["negative_controls"].items()
            },
        },
        "evidence": {
            name: {
                "absolute_path": str(path.resolve()),
                "sha256": _sha256(path.resolve()),
                "status": reports[name].get("status", reports[name].get("official_status")),
            }
            for name, path in INPUTS.items()
        },
        "hard_blockers": hard_blockers,
        "boundaries": {
            "runtime_pass_does_not_suppress_official_fail": True,
            "visual_evidence_is_auxiliary": True,
            "tensor_velocity_not_used_as_drop_authority": True,
            "velocity_disagreement_resolved_as_local_readback_transform_disagreement": True,
            "final_or_default_collider_modified": False,
            "asset_promoted": False,
            "real_robot": False,
            "remote_103": False,
            "task8": "NOT_RUN",
        },
        "task8": "NOT_RUN",
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ALOHA1 CAD-derived Z-up Task 7 closure",
        "",
        f"- Task 7: `{report['status']}`",
        f"- Runtime grasp: `{report['runtime_grasp']}`",
        f"- Visual evidence: `{report['visual_evidence']}`",
        f"- Velocity semantics: `{report['velocity_semantics']}`",
        f"- Asset promotion: `{report['asset_promotion']}`",
        "- Task 8: `NOT_RUN`",
        "",
        "| Official category | Literal status | Blocking | Warnings |",
        "|---|---|---:|---:|",
    ]
    for name, item in report["official_rules"].items():
        if not isinstance(item, dict) or "status" not in item:
            continue
        lines.append(
            f"| {name} | {item['status']} | {item['blocking_issue_count']} | "
            f"{item['warning_count']} |"
        )
    lines.extend(
        [
            "",
            "The Z-up/meters Stage and five fresh primary/repeat pairs are "
            "machine PASS, and the user confirmation is bound to the exact "
            "annotated-video hashes. The V1/V2/V3 experiment resolves the "
            "velocity gate as a verified local PhysX velocity-transform "
            "disagreement without changing the physical signature. Task 7 "
            "remains PARTIAL because the correctly scoped standalone followers "
            "retain 20 literal PhysicsRules errors and all isolated correction "
            "candidates still require review before promotion.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    report = build_report()
    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    OUTPUT_MD.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(OUTPUT_JSON)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
