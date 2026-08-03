#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tools.aloha1_mapping.task8_optimization import build_task8_progression_gate

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/aloha1_mapping/aloha1_task8_progression_authorization.json"
MARKDOWN = ROOT / "reports/aloha1_mapping/aloha1_task8_progression_authorization.md"


KNOWN_ISSUES: list[dict[str, str]] = [
    {
        "id": "continuous_actuator_envelope",
        "status": "APPROXIMATION_ALLOWED_NOT_CALIBRATED",
        "summary": "The measured continuous torque-speed-current thermal envelope is unavailable.",
    },
    {
        "id": "physx_joint_drive_mapping",
        "status": "APPROXIMATION_ALLOWED_NOT_CALIBRATED",
        "summary": "The exact controller/transmission to PhysX drive mapping is incomplete.",
    },
    {
        "id": "finite_contact_patch_miss",
        "status": "REJECTED_DIAGNOSTIC_CANDIDATE",
        "summary": "The rejected compound patch misses the Bottle500 central tangent by about 1.614 mm; the final/default collider was not changed.",
    },
    {
        "id": "finger_bottle_table_contact_materials",
        "status": "TEMPORARY_UNCALIBRATED",
        "summary": "Exact finger/bottle/table material-pair coefficients are not measured.",
    },
    {
        "id": "physics_timestep_solver_selection",
        "status": "FUNCTIONALLY_PASSING_NOT_NUMERICALLY_CONVERGED",
        "summary": "All tested rates held the bottle, but the strict cross-rate trajectory convergence bounds did not pass.",
    },
]


def _signature(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_report() -> dict[str, Any]:
    gate = build_task8_progression_gate(
        runtime_grasp_status="PASS",
        finger_safety_status="PASS",
        model_first_status="PARTIAL_MODEL_PROOF",
        known_issues=KNOWN_ISSUES,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": gate["status"],
        "authorization_basis": "USER_EXPLICITLY_RELAXED_MODEL_PROOF_AS_A_TASK8_ENTRY_GATE_2026_08_03",
        "task7": "PARTIAL_ACCEPTED_FOR_TASK8",
        "task8": "AUTHORIZED_IN_PROGRESS",
        "progression_gate": gate,
        "policy": {
            "approximate_simulation": "ALLOWED_WITH_EXPLICIT_PROVENANCE_AND_LIMITATIONS",
            "parameter_sensitivity": "ONE_VARIABLE_AT_A_TIME_WHEN_NEEDED",
            "known_issues": "RECALL_ON_MATCHING_FAILURE_OR_PROMOTION_REVIEW",
            "additional_contact_patch_screenshots": "NOT_REQUESTED",
            "repeat_five_grasp_videos": "NOT_REQUIRED_BY_DEFAULT",
            "final_default_asset_promotion": "REQUIRES_SEPARATE_REVIEW",
        },
        "history_preserved": {
            "strict_model_first_report_rewritten": False,
            "rejected_compound_candidate_promoted": False,
            "final_or_default_asset_modified": False,
        },
    }
    report["deterministic_signature"] = _signature(report)
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ALOHA1 Task 8 progression authorization",
        "",
        f"Status: `{report['status']}`",
        "",
        "The user explicitly removed the strict model-proof findings as a Task 8 entry gate. "
        "Approximate digital simulation is allowed when provenance and limitations remain explicit. "
        "This authorization does not rewrite historical reports or promote a final/default asset.",
        "",
        "## Non-blocking reminders",
        "",
    ]
    lines.extend(
        f"- `{item['id']}` — `{item['status']}`: {item['summary']}"
        for item in report["progression_gate"]["known_issue_reminders"]
    )
    lines.extend(
        [
            "",
            "These items are recalled only when a matching Task 8 failure appears or during a later "
            "final/default promotion review. No additional contact-patch screenshots or repeated five-grasp "
            "videos are required now.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    report = build_report()
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    MARKDOWN.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(OUTPUT)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
