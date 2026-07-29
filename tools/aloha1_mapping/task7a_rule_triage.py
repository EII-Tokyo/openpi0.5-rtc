"""Evidence-bound triage for ALOHA1 Task 7A official-rule findings."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

ALLOWED_CLASSIFICATIONS = {
    "ASSET_AUTHORING_DEFECT",
    "LAYER_PACKAGING_DEFECT",
    "ISAAC_5_1_VALIDATOR_SCHEMA_CONFLICT",
    "MISSING_SOURCE_EVIDENCE",
    "NON_APPLICABLE_FALSE_POSITIVE",
    "INCONCLUSIVE",
}

PACKAGING_RULES = {
    "NoOverrides",
    "RobotNaming",
    "RobotSchema",
    "ThumbnailExists",
    "VerifyRobotPhysicsAttributesSourceLayer",
    "VerifyRobotPhysicsSchemaSourceLayer",
}

SOURCE_BY_RULE = {
    "JointHasJointStateAPI": (
        "joint_rules.py",
        "JointHasJointStateAPI",
    ),
    "MimicAPICheck": ("joint_rules.py", "MimicAPICheck"),
    "RigidBodyHasCollider": ("physics_rules.py", "RigidBodyHasCollider"),
    "NoOverrides": ("robot_rules.py", "NoOverrides"),
    "RobotNaming": ("robot_rules.py", "RobotNaming"),
    "RobotSchema": ("robot_rules.py", "RobotSchema"),
    "ThumbnailExists": ("robot_rules.py", "ThumbnailExists"),
    "VerifyRobotPhysicsAttributesSourceLayer": (
        "robot_rules.py",
        "VerifyRobotPhysicsAttributesSourceLayer",
    ),
    "VerifyRobotPhysicsSchemaSourceLayer": (
        "robot_rules.py",
        "VerifyRobotPhysicsSchemaSourceLayer",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_gripper_joint_state(rule: str, location: str) -> bool:
    return rule == "JointHasJointStateAPI" and location.endswith(
        "/joints/gripper>"
    )


def _is_right_finger_mimic(rule: str, location: str) -> bool:
    return rule == "MimicAPICheck" and location.endswith(
        "/joints/right_finger>"
    )


def _is_mass_only_helper(rule: str, location: str) -> bool:
    return rule == "RigidBodyHasCollider" and any(
        location.endswith(suffix)
        for suffix in (
            "_ee_arm_link>",
            "_ee_gripper_link>",
            "_fingers_link>",
        )
    )


def classify_rule_issue(
    issue: Mapping[str, Any],
    *,
    category: str,
    target_name: str,
    proven_asset_authoring_defect: bool = False,
) -> dict[str, Any]:
    """Classify one literal official result without suppressing it."""
    rule = str(issue.get("rule"))
    location = str(issue.get("at"))
    result: dict[str, Any] = {
        **dict(issue),
        "category": category,
        "target_name": target_name,
        "official_severity": issue.get("severity"),
        "official_result_suppressed": False,
        "provenance_class": "LOCAL_ISAAC_5_1_RULE_PLUS_RUNTIME_EVIDENCE",
    }
    if proven_asset_authoring_defect:
        result.update(
            {
                "classification": "ASSET_AUTHORING_DEFECT",
                "closure": "REQUIRES_ISOLATED_ASSET_FIX_AND_REGRESSION",
                "task7a_applicability": "DIRECT",
            }
        )
    elif _is_gripper_joint_state(rule, location):
        result.update(
            {
                "classification": "LAYER_PACKAGING_DEFECT",
                "closure": "FORMALLY_RECORDED_CHILD_TARGET_BOUNDARY",
                "task7a_applicability": "DIRECT_LAYER_BOUNDARY",
                "workcell_home_layer_has_required_api": True,
                "runtime_evidence_required": (
                    "gripper target/readback one-joint validation"
                ),
            }
        )
    elif _is_right_finger_mimic(rule, location):
        result.update(
            {
                "classification": (
                    "ISAAC_5_1_VALIDATOR_SCHEMA_CONFLICT"
                ),
                "closure": "FORMALLY_RECORDED_VERSION_SPECIFIC_CONFLICT",
                "task7a_applicability": "DIRECT_RUNTIME_CROSS_CHECK",
                "runtime_evidence_required": (
                    "opposed local-axis limits, mimic target/readback, "
                    "and zero active drive"
                ),
            }
        )
    elif _is_mass_only_helper(rule, location):
        result.update(
            {
                "classification": "MISSING_SOURCE_EVIDENCE",
                "closure": "HARD_BLOCKER_NO_SOURCE_COLLIDER_GEOMETRY",
                "task7a_applicability": (
                    "OUT_OF_SCOPE_GEOMETRY_BOUNDARY"
                ),
                "invent_collider_allowed": False,
            }
        )
    elif rule in PACKAGING_RULES:
        result.update(
            {
                "classification": "LAYER_PACKAGING_DEFECT",
                "closure": "FORMALLY_RECORDED_PACKAGE_STRUCTURE_BOUNDARY",
                "task7a_applicability": "PACKAGE_NOT_CONTROL_SIGNAL",
            }
        )
    elif issue.get("severity") in {"INFO"}:
        result.update(
            {
                "classification": "NON_APPLICABLE_FALSE_POSITIVE",
                "closure": "OFFICIAL_NON_BLOCKING_INFORMATION_RECORDED",
                "task7a_applicability": "NON_BLOCKING",
            }
        )
    else:
        result.update(
            {
                "classification": "INCONCLUSIVE",
                "closure": "OPEN_REQUIRES_SOURCE_AND_RUNTIME_INVESTIGATION",
                "task7a_applicability": "DIRECT_UNRESOLVED",
            }
        )
    return result


def _extension_root(project_root: Path) -> Path:
    return (
        project_root
        / ".venv_issac/lib/python3.11/site-packages/isaacsim/exts/"
        "isaacsim.asset.validation"
    )


def _source_manifest(project_root: Path) -> dict[str, Any]:
    root = _extension_root(project_root)
    python_root = root / "isaacsim/asset/validation"
    manifest = root / "config/extension.toml"
    paths = [
        manifest,
        python_root / "joint_rules.py",
        python_root / "physics_rules.py",
        python_root / "robot_rules.py",
    ]
    records = []
    for path in paths:
        resolved = path.resolve(strict=True)
        records.append(
            {
                "absolute_path": str(resolved),
                "sha256": _sha256(resolved),
                "size_bytes": resolved.stat().st_size,
            }
        )
    return {
        "extension": "isaacsim.asset.validation",
        "version": "1.1.0",
        "files": records,
    }


def _home_layer_evidence(project_root: Path) -> dict[str, Any]:
    path = (
        project_root
        / "assets/Trossen/ALOHA1/1.0/diagnostics/"
        "signal_correspondence/1.0/configuration/"
        "aloha1_signal_home_targets.usda"
    ).resolve(strict=True)
    text = path.read_text(encoding="utf-8")
    return {
        "absolute_path": str(path),
        "sha256": _sha256(path),
        "gripper_joint_state_api_occurrences": text.count(
            'over "gripper" (\n'
            '                prepend apiSchemas = ["PhysicsJointStateAPI:angular"]'
        ),
        "expected_occurrences": 2,
        "status": (
            "PASS"
            if text.count(
                'over "gripper" (\n'
                '                prepend apiSchemas = ["PhysicsJointStateAPI:angular"]'
            )
            == 2
            else "FAIL"
        ),
    }


def build_rule_triage(
    project_root: Path,
    official_report_path: Path,
    *,
    mimic_probe_path: Path | None = None,
) -> dict[str, Any]:
    """Build complete deterministic triage for a combined official report."""
    root = project_root.resolve(strict=True)
    official_path = official_report_path.resolve(strict=True)
    official = json.loads(official_path.read_text(encoding="utf-8"))
    home_evidence = _home_layer_evidence(root)
    if home_evidence["status"] != "PASS":
        raise ValueError("frozen workcell home layer lacks two gripper APIs")

    triaged: list[dict[str, Any]] = []
    issue_keys: list[str] = []
    for target in official["targets"]:
        category = str(target["category"])
        target_name = str(target["target_name"])
        for issue in target.get("issues", []):
            classified = classify_rule_issue(
                issue,
                category=category,
                target_name=target_name,
            )
            source = SOURCE_BY_RULE.get(str(issue.get("rule")))
            if source is not None:
                classified["installed_rule_source"] = {
                    "file": source[0],
                    "class": source[1],
                }
            key_payload = {
                "category": category,
                "target_name": target_name,
                "severity": issue.get("severity"),
                "rule": issue.get("rule"),
                "at": issue.get("at"),
                "message": issue.get("message"),
            }
            key = hashlib.sha256(
                json.dumps(
                    key_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            classified["issue_key"] = key
            issue_keys.append(key)
            triaged.append(classified)

    counts = Counter(item["classification"] for item in triaged)
    source_count = sum(
        len(target.get("issues", [])) for target in official["targets"]
    )
    report = {
        "schema_version": 1,
        "status": (
            "PARTIAL"
            if official["official_status"] == "FAIL"
            else official["official_status"]
        ),
        "official_status": official["official_status"],
        "official_status_suppressed": False,
        "official_report": {
            "absolute_path": str(official_path),
            "sha256": _sha256(official_path),
        },
        "runtime": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
            "asset_validation_extension": "1.1.0",
        },
        "source_issue_count": source_count,
        "triaged_issue_count": len(triaged),
        "duplicate_issue_count": len(issue_keys) - len(set(issue_keys)),
        "unclassified_issue_count": counts.get("INCONCLUSIVE", 0),
        "classification_counts": dict(sorted(counts.items())),
        "allowed_classifications": sorted(ALLOWED_CLASSIFICATIONS),
        "home_layer_evidence": home_evidence,
        "installed_source_manifest": _source_manifest(root),
        "issues": triaged,
        "real_robot_connected": False,
        "remote_192_168_1_103_accessed": False,
        "task_8": "NOT_RUN",
    }
    if mimic_probe_path is not None:
        probe_path = mimic_probe_path.resolve(strict=True)
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        if probe.get("status") != "PASS":
            raise ValueError("mimic runtime probe did not pass")
        if probe.get("stage_modified") is not False:
            raise ValueError("mimic runtime probe modified its Stage")
        robots = probe.get("robots", {})
        if set(robots) != {"follower_left", "follower_right"}:
            raise ValueError("mimic runtime probe robot coverage mismatch")
        limits: dict[str, Any] = {}
        for robot, joints in robots.items():
            left = joints["left_finger"]
            right = joints["right_finger"]
            gearing = right["mimic_attributes"][
                "physxMimicJoint:rotY:gearing"
            ]
            if not (
                float(left["runtime_lower"]) > 0.0
                and float(left["runtime_upper"]) > 0.0
                and float(right["runtime_lower"]) < 0.0
                and float(right["runtime_upper"]) < 0.0
                and float(gearing) == 1.0
            ):
                raise ValueError(
                    f"{robot} mimic probe does not show opposed local axes"
                )
            limits[robot] = {
                "left_finger_runtime_limits": [
                    left["runtime_lower"],
                    left["runtime_upper"],
                ],
                "right_finger_runtime_limits": [
                    right["runtime_lower"],
                    right["runtime_upper"],
                ],
                "mimic_gearing": gearing,
            }
        report["mimic_runtime_probe"] = {
            "status": "PASS",
            "absolute_path": str(probe_path),
            "sha256": _sha256(probe_path),
            "stage_sha256": probe["stage"]["sha256"],
            "stage_modified": False,
            "loaded_rule_module": probe["loaded_rule_module"],
            "robot_limits": limits,
            "interpretation": (
                "OPPOSED_LOCAL_JOINT_AXES_NOT_MODELED_BY_NUMERIC_LIMIT_RULE"
            ),
            "official_result_suppressed": False,
        }
        for issue in report["issues"]:
            if issue["rule"] == "MimicAPICheck":
                issue["runtime_probe_status"] = "PASS"
                issue["runtime_probe_interpretation"] = report[
                    "mimic_runtime_probe"
                ]["interpretation"]
    return report
