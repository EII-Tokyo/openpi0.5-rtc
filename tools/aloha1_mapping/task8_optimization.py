from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any

PROTECTED_PHYSICS_KEYS = ("articulations", "joints", "rigid_bodies", "colliders")
MODEL_FIRST_REQUIRED_GATES = (
    "source_audit",
    "parameter_matrix",
    "kinematic_contract",
    "dynamics_contract",
    "gripper_geometry_contract",
    "collider_geometry_contract",
    "runtime_contract",
)


def _finite_json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "+Infinity" if value > 0 else "-Infinity"
    if isinstance(value, Mapping):
        return {str(key): _finite_json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_finite_json_value(item) for item in value]
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        _finite_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _duplicate_groups(
    mesh_records: Sequence[Mapping[str, Any]], *, collision: bool
) -> list[dict[str, object]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in mesh_records:
        if bool(record["is_collision"]) != collision:
            continue
        grouped[str(record["geometry_signature"])].append(str(record["path"]))
    return [
        {
            "geometry_signature": signature,
            "count": len(paths),
            "paths": sorted(paths),
        }
        for signature, paths in sorted(grouped.items())
        if len(paths) > 1
    ]


def _duplicate_material_groups(
    material_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in material_records:
        grouped[str(record["material_signature"])].append(str(record["path"]))
    return [
        {
            "material_signature": signature,
            "count": len(paths),
            "paths": sorted(paths),
        }
        for signature, paths in sorted(grouped.items())
        if len(paths) > 1
    ]


def build_inventory_summary(
    *,
    mesh_records: Sequence[Mapping[str, Any]],
    material_records: Sequence[Mapping[str, Any]],
    prim_type_counts: Mapping[str, int],
    composition_records: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    visual = [record for record in mesh_records if not bool(record["is_collision"])]
    collision = [record for record in mesh_records if bool(record["is_collision"])]
    repeated_visual = _duplicate_groups(mesh_records, collision=False)
    repeated_collision = _duplicate_groups(mesh_records, collision=True)
    duplicate_materials = _duplicate_material_groups(material_records)
    return {
        "prim_count": sum(int(value) for value in prim_type_counts.values()),
        "prim_type_counts": dict(sorted(prim_type_counts.items())),
        "mesh_count": len(mesh_records),
        "visual_mesh_count": len(visual),
        "collision_mesh_count": len(collision),
        "point_count": sum(int(record["point_count"]) for record in mesh_records),
        "face_count": sum(int(record["face_count"]) for record in mesh_records),
        "material_count": len(material_records),
        "instance_proxy_mesh_count": sum(
            bool(record.get("is_instance_proxy")) for record in mesh_records
        ),
        "instanceable_mesh_count": sum(
            bool(record.get("is_instanceable")) for record in mesh_records
        ),
        "instanceable_prim_count": sum(
            bool(record.get("is_instanceable")) for record in composition_records
        ),
        "reference_prim_count": sum(
            bool(record.get("has_authored_references"))
            for record in composition_records
        ),
        "payload_prim_count": sum(
            bool(record.get("has_authored_payloads")) for record in composition_records
        ),
        "repeated_visual_geometry_groups": len(repeated_visual),
        "repeated_visual_mesh_instances": sum(
            int(group["count"]) for group in repeated_visual
        ),
        "repeated_collision_geometry_groups": len(repeated_collision),
        "repeated_collision_mesh_instances": sum(
            int(group["count"]) for group in repeated_collision
        ),
        "duplicate_material_groups": len(duplicate_materials),
        "repeated_visual_geometry": repeated_visual,
        "repeated_collision_geometry": repeated_collision,
        "duplicate_materials": duplicate_materials,
    }


def rank_optimization_opportunities(
    summary: Mapping[str, Any], *, known_hydra_instance_regression: bool
) -> list[dict[str, object]]:
    opportunities: list[dict[str, object]] = []
    visual_groups = int(summary.get("repeated_visual_geometry_groups", 0))
    if visual_groups:
        opportunities.append(
            {
                "id": "deduplicate_repeated_visual_geometry",
                "priority": 1,
                "decision": "ISOLATED_CANDIDATE",
                "risk": (
                    "MEDIUM_HYDRA_REGRESSION_KNOWN"
                    if known_hydra_instance_regression
                    else "LOW_PHYSICS_MEDIUM_RENDER"
                ),
                "changes_physics_composition": False,
                "group_count": visual_groups,
                "mesh_instance_count": int(
                    summary.get("repeated_visual_mesh_instances", 0)
                ),
            }
        )
    material_groups = int(summary.get("duplicate_material_groups", 0))
    if material_groups:
        opportunities.append(
            {
                "id": "deduplicate_materials",
                "priority": 2,
                "decision": "ISOLATED_CANDIDATE_AFTER_BINDING_AUDIT",
                "risk": "LOW_PHYSICS_MEDIUM_VISUAL",
                "changes_physics_composition": False,
                "group_count": material_groups,
            }
        )
    collision_groups = int(summary.get("repeated_collision_geometry_groups", 0))
    if collision_groups:
        opportunities.append(
            {
                "id": "deduplicate_collision_geometry",
                "priority": 90,
                "decision": "DEFER_UNTIL_VISUAL_CANDIDATE_EVALUATED",
                "risk": "HIGH_PHYSICS_REGRESSION",
                "changes_physics_composition": True,
                "group_count": collision_groups,
                "mesh_instance_count": int(
                    summary.get("repeated_collision_mesh_instances", 0)
                ),
            }
        )
    opportunities.append(
        {
            "id": "add_payloads",
            "priority": 99,
            "decision": (
                "NO_ACTION_ALREADY_PRESENT"
                if int(summary.get("payload_prim_count", 0)) > 0
                else "REVIEW_FOR_DEFERRED_LOADING"
            ),
            "risk": "LOW_PHYSICS_COMPOSITION_ONLY",
            "changes_physics_composition": False,
        }
    )
    return sorted(opportunities, key=lambda item: (int(item["priority"]), str(item["id"])))


def build_protected_signature(inventory: Mapping[str, Any]) -> str:
    protected = {
        key: inventory.get(key, [])
        for key in PROTECTED_PHYSICS_KEYS
    }
    return hashlib.sha256(_canonical_json(protected).encode("utf-8")).hexdigest()


def failure_evidence_contract(*, reproducible: bool) -> dict[str, object]:
    phases = ["before_anomaly", "first_anomalous_frame", "final_failure"]
    return {
        "reproducible": reproducible,
        "raw_screenshots": phases if reproducible else [],
        "annotated_screenshots": phases if reproducible else [],
        "full_arm_collision_enabled_video_required": reproducible,
        "visual_review_required": reproducible,
        "machine_telemetry_required": reproducible,
    }


def build_model_first_gate(gates: Mapping[str, Mapping[str, Any]]) -> dict[str, object]:
    records = []
    for gate_id in MODEL_FIRST_REQUIRED_GATES:
        gate = gates.get(gate_id)
        if gate is None:
            records.append(
                {"id": gate_id, "status": "NOT_RUN", "blocking": True, "reason": "MISSING_GATE"}
            )
            continue
        status = str(gate.get("status", "UNKNOWN"))
        nested_candidate_status = gate.get("candidate_gate")
        blocking = status != "PASS" or nested_candidate_status not in (None, "PASS")
        records.append(
            {
                "id": gate_id,
                "status": status,
                "candidate_gate": nested_candidate_status,
                "blocking": blocking,
                "reason": "PASS" if not blocking else "MODEL_PROOF_INCOMPLETE",
            }
        )
    blockers = [record for record in records if record["blocking"]]
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "candidate_authoring_allowed": not blockers,
        "required_gates": records,
        "blocking_gates": blockers,
    }


def build_task8_progression_gate(
    *,
    runtime_grasp_status: str,
    finger_safety_status: str,
    model_first_status: str,
    known_issues: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Separate functional Task 8 progression from calibrated asset promotion.

    Model-first findings remain visible and continue to block claims of a
    calibrated sim-to-real model or promotion into final/default assets.  They
    do not block isolated Task 8 candidates once the accepted runtime grasp and
    finger-safety baselines pass.
    """

    functional_gates = {
        "runtime_grasp": str(runtime_grasp_status),
        "finger_safety": str(finger_safety_status),
    }
    functional_pass = all(status == "PASS" for status in functional_gates.values())
    reminders = [
        {
            "id": str(issue["id"]),
            "status": str(issue["status"]),
            "summary": str(issue["summary"]),
            "blocking_task8": False,
            "recall_when": "matching Task 8 failure or final/default promotion review",
        }
        for issue in known_issues
    ]
    return {
        "status": (
            "AUTHORIZED_IN_PROGRESS"
            if functional_pass
            else "BLOCKED_BY_FUNCTIONAL_BASELINE"
        ),
        "functional_gates": functional_gates,
        "model_first_status": str(model_first_status),
        "isolated_candidate_authoring_allowed": functional_pass,
        "approximate_simulation_allowed": functional_pass,
        "final_default_promotion_allowed": False,
        "sim_to_real_calibrated_claim_allowed": False,
        "known_issue_reminders": reminders,
    }
