#!/usr/bin/env python3
"""Build the final, non-promotional Task 7 PhysicsRules root-cause closure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/aloha1_mapping"
MATRIX = REPORT_ROOT / "aloha1_task7_physicsrules_root_cause_matrix.json"
MASS_AUDIT = REPORT_ROOT / "aloha1_task7_virtual_helper_mass_audit.json"
OUTPUT = REPORT_ROOT / "aloha1_task7_physicsrules_root_cause_closure.json"
OUTPUT_MD = OUTPUT.with_suffix(".md")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict[str, Any]:
    matrix = _load(MATRIX)
    mass = _load(MASS_AUDIT)
    captures: list[dict[str, Any]] = []
    review_inputs = []
    for side in ("left", "right"):
        path = REPORT_ROOT / f"aloha1_task7_virtual_helper_failure_screenshot_review_{side}.json"
        review = _load(path)
        review_inputs.append(
            {
                "absolute_path": str(path.resolve()),
                "sha256": _sha256(path),
                "status": review["status"],
            }
        )
        captures.extend(
            {
                    "follower": capture["follower"],
                    "view": capture["view"],
                    "raw_absolute_path": capture["raw_absolute_path"],
                    "raw_sha256": capture["raw_sha256"],
                    "annotated_absolute_path": capture["annotated_absolute_path"],
                    "annotated_sha256": capture["annotated_sha256"],
                    "visual_model_review": capture["visual_model_review"],
                    "failure_shown": (
                        "UNCOMPENSATED_HELPER_BODY_REMOVAL_CREATES_"
                        "NONADJACENT_COLLIDER_CLASH_FINDINGS"
                    ),
            }
            for capture in review["captures"]
        )
    combined = matrix["profiles"]["combined_topology_joint_state"]
    report = {
        "schema_version": 1,
        "status": "PARTIAL",
        "task7": "PARTIAL",
        "task8": "NOT_RUN",
        "scope": "TASK7_PHYSICSRULES_ROOT_CAUSE_CLOSURE_DIAGNOSTIC_ONLY",
        "runtime": matrix["runtime"],
        "frozen_stage_sha256": matrix["frozen_stage_sha256"],
        "original_physicsrules_finding_count": 20,
        "original_rule_counts": {
            "JointHasCorrectTransformAndState": 10,
            "MimicAPICheck": 2,
            "RigidBodyHasCollider": 8,
        },
        "validator_fresh_process_count": matrix["validator_fresh_process_count"],
        "runtime_fresh_process_count": matrix["runtime_fresh_process_count"],
        "combined_candidate_literal_blocking_count": sum(
            item["repeats"][0]["blocking_issue_count"]
            for item in combined["followers"].values()
        ),
        "combined_candidate_literal_rule_counts": combined["blocking_rule_counts"],
        "root_cause_dispositions": {
            "joint_state": (
                "VALIDATOR_EXPECTS_GEOMETRY_ZERO_STATE_WHILE_PACKAGE_AUTHORS_HOME_STATE; "
                "ZERO_STATE_OVERRIDE_REMOVES_10_FINDINGS_AND_IS_RUNTIME_EQUIVALENT"
            ),
            "mimic": (
                "ISAAC_SIM_ASSET_VALIDATION_1_1_0_FORMULA_CONFLICT; "
                "PHYSX_107_3_OPPOSED_AXIS_AUTHORING_REMAINS_UNCHANGED"
            ),
            "gripper_bar": (
                "COLLIDER_EXISTS_IN_SUPPLIER_CAD_FIXED_GROUP; SPLIT_CANDIDATE "
                "REMOVES_2 FINDINGS BUT CHANGES_COLLIDER_PATHS"
            ),
            "empty_helpers": (
                "SIMPLE_RIGID_BODY_REMOVAL_REJECTED; FRAME_PRESERVING_TOPOLOGY "
                "COLLAPSE_REMOVES_6_FINDINGS_BUT_DOES_NOT_PRESERVE_MASS_SEMANTICS"
            ),
        },
        "helper_mass_semantics": {
            "removed_mass_per_follower_kg": matrix["helper_mass_semantics"][
                "removed_mass_per_follower_kg"
            ],
            "physical_calibration_status": mass["physical_calibration_status"],
            "uncompensated_collapse_allowed": False,
            "audit_absolute_path": str(MASS_AUDIT.resolve()),
            "audit_sha256": _sha256(MASS_AUDIT),
        },
        "failure_evidence": {
            "status": (
                "PASS"
                if len(captures) == 4
                and all(item["visual_model_review"] == "PASS" for item in captures)
                else "FAIL"
            ),
            "trigger": "TWO_IDENTICAL_FRESH_PROCESS_FAILURES_PER_FOLLOWER",
            "review_reports": review_inputs,
            "captures": captures,
        },
        "remaining_real_blockers": [
            "HELPER_MASS_COM_INERTIA_SEMANTICS_NOT_PRESERVED_IN_TOPOLOGY_CANDIDATE",
            "COLLIDER_SPLIT_AND_TOPOLOGY_CANDIDATE_NOT_PROMOTED_OR_GRASP_REGRESSED",
        ],
        "literal_validator_boundary": (
            "The combined isolated candidate still reports one MimicAPICheck per "
            "follower. The finding is retained unsuppressed even though local PhysX "
            "107.3 equation evidence supports the current opposed-axis mapping."
        ),
        "final_or_default_asset_modified": False,
        "real_robot_connected": False,
        "remote_192_168_1_103_accessed": False,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# ALOHA1 Task 7 PhysicsRules root-cause closure",
        "",
        "- Status: `PARTIAL`",
        "- Task 7: `PARTIAL`",
        "- Task 8: `NOT_RUN`",
        f"- Frozen Stage SHA-256: `{report['frozen_stage_sha256']}`",
        f"- Validator fresh processes: `{report['validator_fresh_process_count']}`",
        f"- Runtime fresh processes: `{report['runtime_fresh_process_count']}`",
        "- Final/default asset modified: `false`",
        "",
        "## Measured outcome",
        "",
        "The isolated combined candidate reduces the original 20 standalone-follower "
        "PhysicsRules blockers to one unsuppressed `MimicAPICheck` per follower. Both "
        "followers pass two fresh 120-frame runtime probes with identical per-side "
        "signatures.",
        "",
        "The straightforward helper-body removal is rejected. It reproducibly creates "
        "57 non-adjacent collider-clash findings per follower. Four vision-reviewed raw/"
        "annotated images identify the affected helper chain and collision region; their "
        "absolute paths and hashes are in the JSON report.",
        "",
        "The frame-preserving topology candidate avoids that clash regression, but removes "
        f"`{report['helper_mass_semantics']['removed_mass_per_follower_kg']:.9g} kg` of "
        "source-authored helper mass per follower. Those values are source placeholders, "
        "not physically calibrated measurements. The candidate therefore cannot be "
        "promoted until its mass/COM/inertia semantics are preserved and the changed "
        "collider composition passes the accepted grasp regression.",
        "",
        "## Remaining real blockers",
        "",
        *[f"- `{item}`" for item in report["remaining_real_blockers"]],
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    report = build()
    print(json.dumps({"status": report["status"], "task7": report["task7"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
