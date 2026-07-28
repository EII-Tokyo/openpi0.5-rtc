"""Evidence-backed classification of Isaac Sim 5.1 PhysicsRules findings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_MASS_ONLY_LINK_SUFFIXES = (
    "_ee_arm_link>",
    "_ee_gripper_link>",
    "_fingers_link>",
)


def _classify_issue(issue: Mapping[str, Any]) -> dict[str, Any]:
    rule = str(issue.get("rule", ""))
    location = str(issue.get("at", ""))
    result = dict(issue)

    if rule == "JointHasJointStateAPI" and location.endswith(
        "/joints/gripper>"
    ):
        result.update(
            {
                "classification": "CONFIGURATION_LAYER_JOINT_STATE_OMISSION",
                "resolution": "FIXED_IN_CONFIGURATION_LAYER",
                "evidence": {
                    "isaac_sim_version": "5.1.0.0",
                    "validator_action": (
                        "PhysxSchema.JointStateAPI.Apply(prim, 'angular')"
                    ),
                    "authoring_layer": (
                        "debug/sim2real configuration layer; raw importer "
                        "source remains immutable"
                    ),
                    "official_source_result_suppressed": False,
                },
            }
        )
    elif rule == "MimicAPICheck" and location.endswith(
        "/joints/right_finger>"
    ):
        result.update(
            {
                "classification": (
                    "ISAAC_SIM_5_1_MIMIC_VALIDATOR_SCHEMA_CONFLICT"
                ),
                "resolution": "FORMALLY_RECORDED",
                "evidence": {
                    "isaac_sim_version": "5.1.0.0",
                    "kit_version": "107.3.3",
                    "schema_equation": (
                        "jointPosition + gearing * "
                        "referenceJointPosition + offset = 0"
                    ),
                    "urdf_relation": "right_finger = -1 * left_finger + 0",
                    "imported_gearing": 1.0,
                    "imported_offset": 0.0,
                    "runtime_relation": "right_finger ~= -left_finger",
                    "asset_mutation_allowed": False,
                    "reason": (
                        "Changing gearing to satisfy the validator would "
                        "invert the verified runtime relation."
                    ),
                },
            }
        )
    elif rule == "RigidBodyHasCollider" and location.endswith(
        _MASS_ONLY_LINK_SUFFIXES
    ):
        result.update(
            {
                "classification": "SOURCE_MASS_ONLY_LINK_WITHOUT_GEOMETRY",
                "resolution": "HARD_BLOCKER_NO_GEOMETRY_EVIDENCE",
                "evidence": {
                    "source_urdf": (
                        "link has inertial data but no visual or collision"
                    ),
                    "import_policy": "merge_fixed_joints=false",
                    "runtime_articulation_created": True,
                    "asset_mutation_allowed": False,
                    "reason": (
                        "No STL, CAD, or measured geometry exists for a "
                        "non-guessed collider."
                    ),
                },
            }
        )
    else:
        result.update(
            {
                "classification": "UNCLASSIFIED",
                "resolution": "FIX_REQUIRED",
                "evidence": {
                    "asset_mutation_allowed": False,
                    "reason": "No evidence-backed exception is defined.",
                },
            }
        )
    return result


def classify_physics_rule_category(
    category: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify PhysicsRules issues without suppressing official outcomes."""

    classified = []
    for target in category.get("targets", []):
        for issue in target.get("issues", []):
            record = _classify_issue(issue)
            record["target"] = target.get("target")
            classified.append(record)

    unclassified = sum(
        item["classification"] == "UNCLASSIFIED" for item in classified
    )
    fix_required = sum(
        item["resolution"] == "FIX_REQUIRED" for item in classified
    )
    if fix_required or unclassified:
        status = "FAIL"
    elif classified:
        status = "PARTIAL"
    else:
        status = "PASS"
    return {
        "schema_version": 1,
        "status": status,
        "official_status": category.get("status"),
        "official_status_suppressed": False,
        "issues": classified,
        "issue_count": len(classified),
        "fix_required_count": fix_required,
        "unclassified_error_count": unclassified,
    }


def build_physicsrules_report(
    asset_validator_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Select and classify the exact PhysicsRules category."""

    matches = [
        category
        for category in asset_validator_report.get("categories", [])
        if category.get("category") == "IsaacSim.PhysicsRules"
    ]
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one IsaacSim.PhysicsRules category, "
            f"found {len(matches)}"
        )
    return classify_physics_rule_category(matches[0])
