"""Pure status contracts for post-grasp ALOHA1 Task 7 closure."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

RUNTIME_GRASP_GATES = (
    "runtime_control",
    "workcell_physics",
    "aloha_6dof_ik_correspondence",
    "table_support_alignment",
    "static_bottle_hold",
    "dynamic_five_pose_grasp",
    "visual_model_review",
    "user_confirmation",
)

VALID_STATUSES = {"PASS", "FAIL", "PARTIAL", "NOT_RUN"}


def _combined_status(statuses: list[str]) -> str:
    if not statuses or any(status == "FAIL" for status in statuses):
        return "FAIL"
    if any(status in {"PARTIAL", "NOT_RUN"} for status in statuses):
        return "PARTIAL"
    if all(status == "PASS" for status in statuses):
        return "PASS"
    return "FAIL"


def classify_post_grasp_task7(
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Separate measured grasp acceptance from asset-promotion readiness."""

    gates = {
        name: str(inputs.get(name, "NOT_RUN"))
        for name in RUNTIME_GRASP_GATES
    }
    invalid = {
        name: status
        for name, status in gates.items()
        if status not in VALID_STATUSES
    }
    if invalid:
        raise ValueError(f"invalid Task 7 status values: {invalid}")

    runtime_grasp = _combined_status(list(gates.values()))
    promotion = str(
        inputs.get("asset_promotion_readiness", "NOT_RUN")
    )
    if promotion not in VALID_STATUSES:
        raise ValueError(
            f"invalid asset-promotion status: {promotion}"
        )

    aggregate = (
        "FAIL"
        if runtime_grasp == "FAIL"
        else _combined_status([runtime_grasp, promotion])
    )

    requested_task8 = str(inputs.get("task8", "NOT_RUN"))
    boundaries: list[str] = [
        "runtime_grasp_acceptance_is_not_asset_promotion",
        "literal_official_rule_status_is_not_suppressed",
        "no_stage_or_physics_mutation",
    ]
    if requested_task8 != "NOT_RUN":
        boundaries.append("task8_input_was_ignored")

    return {
        "runtime_grasp_acceptance": runtime_grasp,
        "runtime_grasp_gates": gates,
        "failed_runtime_grasp_gates": [
            name for name, status in gates.items() if status == "FAIL"
        ],
        "partial_or_not_run_runtime_grasp_gates": [
            name
            for name, status in gates.items()
            if status in {"PARTIAL", "NOT_RUN"}
        ],
        "asset_promotion_readiness": promotion,
        "official_rules_literal_status": str(
            inputs.get("official_rules_literal_status", "NOT_RUN")
        ),
        "task7_aggregate": aggregate,
        "task8": "NOT_RUN",
        "boundaries": boundaries,
    }
