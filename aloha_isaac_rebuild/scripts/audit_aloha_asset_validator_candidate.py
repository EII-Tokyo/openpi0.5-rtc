#!/usr/bin/env python3
"""Run a read-only A20 Asset Validator audit for the A19 candidate stage.

This script intentionally does not fix issues, save the stage, step physics,
initialize articulation control, or claim replay/RL readiness.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path("aloha_isaac_rebuild/configs/physical_reconstruction/stationary_style_rebuild.yaml")
PASS_STATUS = "PASS_A20_ASSET_VALIDATOR_READ_ONLY_NO_BLOCKING_ISSUES"
FAIL_STATUS = "FAIL_A20_ASSET_VALIDATOR_BLOCKING_ISSUES"


def _asset_validator_status(ok: bool) -> str:
    return PASS_STATUS if ok else FAIL_STATUS


def _start_isaac_headless():
    from isaacsim import SimulationApp

    return SimulationApp({"headless": True})


def _issue_to_dict(issue) -> dict:
    severity = getattr(issue, "severity", None)
    rule = getattr(issue, "rule", None)
    at = getattr(issue, "at", None)
    suggestion = getattr(issue, "suggestion", None)
    return {
        "severity": getattr(severity, "name", str(severity)),
        "rule": getattr(rule, "__name__", str(rule) if rule else None),
        "code": getattr(issue, "code", None),
        "message": getattr(issue, "message", None),
        "at": str(at) if at is not None else None,
        "suggestion": getattr(suggestion, "message", None) if suggestion is not None else None,
    }


def run_audit(config_path: Path) -> dict:
    import omni.kit.app
    from pxr import PhysxSchema
    from pxr import Usd
    from pxr import UsdPhysics

    joint_state_schema_registered = hasattr(PhysxSchema, "JointStateAPI")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    outputs = {key: Path(value) for key, value in config["outputs"].items()}
    stage_path = outputs["a19_clean_articulation_candidate"].resolve()
    output_json = outputs["a20_asset_validator_json"]
    output_md = outputs["a20_asset_validator_md"]

    manager = omni.kit.app.get_app().get_extension_manager()
    enabled_extensions = {}
    for ext in ("omni.asset_validator.core", "omni.physx.asset_validator"):
        manager.set_extension_enabled_immediate(ext, True)
        enabled_extensions[ext] = bool(manager.is_extension_enabled(ext))

    # Isaac Sim 5.1's headless Python can exit quietly if these modules are
    # imported before their extensions are enabled.
    import omni.asset_validator.core as av
    from omni.physxassetvalidator.scripts.apiConflictChecker import APIConflictChecker
    from omni.physxassetvalidator.scripts.articulationJointsChecker import ArticulationJointsChecker
    from omni.physxassetvalidator.scripts.backwardCompatibilityChecker import BackwardCompatibilityChecker
    from omni.physxassetvalidator.scripts.cookingApproximationChecker import CookingApproximationChecker
    from omni.physxassetvalidator.scripts.cookingApproximationChecker import CookingApproximationFallbackChecker
    from omni.physxassetvalidator.scripts.deformableSchemaChecker import DeformableSchemaChecker
    from omni.physxassetvalidator.scripts.jointPoseChecker import JointPoseChecker
    from omni.physxassetvalidator.scripts.jointStateChecker import JointStateChecker
    from omni.physxassetvalidator.scripts.massChecker import MassChecker
    from omni.physxassetvalidator.scripts.simulationOwnerChecker import SimulationOwnerChecker

    stage = Usd.Stage.Open(str(stage_path), load=Usd.Stage.LoadAll)
    opened = stage is not None
    if stage is None:
        raise RuntimeError(f"Could not open stage for validation: {stage_path}")

    usd_physics_rules = [
        av.ArticulationChecker,
        av.ColliderChecker,
        av.PhysicsJointChecker,
        av.RigidBodyChecker,
    ]
    physx_rules = [
        APIConflictChecker,
        ArticulationJointsChecker,
        BackwardCompatibilityChecker,
        CookingApproximationChecker,
        CookingApproximationFallbackChecker,
        DeformableSchemaChecker,
        JointPoseChecker,
        JointStateChecker,
        MassChecker,
        SimulationOwnerChecker,
    ]
    selected_rules = usd_physics_rules + physx_rules

    # Inventory the authored asset before adding validation-only session data.
    type_counts: dict[str, int] = {}
    api_counts: dict[str, int] = {}
    for prim in stage.Traverse():
        prim_type = prim.GetTypeName() or "Typeless"
        type_counts[prim_type] = type_counts.get(prim_type, 0) + 1
        for schema in prim.GetAppliedSchemas():
            api_counts[schema] = api_counts.get(schema, 0) + 1

    authored_physics_scene_paths = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.GetTypeName() == "PhysicsScene"
    ]
    temporary_physics_scene_path = "/aloha/__assetValidatorPhysicsScene"
    temporary_validation_physics_scene_authored = not authored_physics_scene_paths
    original_edit_target = stage.GetEditTarget()
    session_layer = stage.GetSessionLayer()
    stage.SetEditTarget(session_layer)
    try:
        if temporary_validation_physics_scene_authored:
            UsdPhysics.Scene.Define(stage, temporary_physics_scene_path)
        engine = av.ValidationEngine(init_rules=False, variants=False)
        for rule in selected_rules:
            engine.enable_rule(rule)
        results = engine.validate(stage)
        issues = [_issue_to_dict(issue) for issue in results.issues()]
    finally:
        session_layer.Clear()
        stage.SetEditTarget(original_edit_target)

    severity_counts = Counter(issue["severity"] for issue in issues)
    rule_counts = Counter(issue["rule"] for issue in issues)

    blocking_severities = {"ERROR", "FAILURE"}
    blocking_issue_count = sum(count for severity, count in severity_counts.items() if severity in blocking_severities)
    ok = (
        opened
        and all(enabled_extensions.values())
        and joint_state_schema_registered
        and str(stage.GetDefaultPrim().GetPath()) == "/aloha"
        and api_counts.get("PhysicsArticulationRootAPI", 0) == 1
        and type_counts.get("PhysicsFixedJoint", 0) == 5
        and type_counts.get("PhysicsRevoluteJoint", 0) == 12
        and type_counts.get("PhysicsPrismaticJoint", 0) == 4
        and api_counts.get("PhysicsCollisionAPI", 0) == 0
        and blocking_issue_count == 0
    )
    result = {
        "ok": ok,
        "status": _asset_validator_status(ok),
        "stage_path": str(stage_path),
        "opened_in_isaac_runtime": opened,
        "joint_state_schema_registered": joint_state_schema_registered,
        "default_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None,
        "enabled_extensions": enabled_extensions,
        "selected_rule_count": len(selected_rules),
        "selected_rules": [rule.__name__ for rule in selected_rules],
        "issue_count": len(issues),
        "blocking_issue_count": blocking_issue_count,
        "severity_counts": dict(sorted(severity_counts.items())),
        "rule_counts": dict(sorted(rule_counts.items())),
        "issues": issues,
        "type_counts": type_counts,
        "api_counts": api_counts,
        "authored_physics_scene_paths": authored_physics_scene_paths,
        "temporary_validation_physics_scene_authored": (
            temporary_validation_physics_scene_authored
        ),
        "temporary_validation_physics_scene_path": (
            temporary_physics_scene_path
            if temporary_validation_physics_scene_authored
            else None
        ),
        "physics_stepped": False,
        "auto_fix_applied": False,
        "stage_saved": False,
        "collision_ready": False,
        "control_ready": False,
        "replay_ready": False,
        "training_eligible": False,
        "next_required_gates": [
            "Resolve any Asset Validator warnings that affect articulation semantics",
            "Run Isaac articulation discovery and DOF readback without stepping contact",
            "Validate canonical 16-DOF order against A17 mapping",
            "Run small set-target/readback sign gate",
            "Run gravity-off and gravity-on hold gates before replay/contact/RL",
        ],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_format_report(result), encoding="utf-8")
    return result


def _format_report(result: dict) -> str:
    top_issues = result["issues"][:20]
    issue_lines = [
        f"- `{issue['severity']}` `{issue['rule']}` at `{issue['at']}`: {issue['message']}"
        for issue in top_issues
    ]
    if not issue_lines:
        issue_lines = ["- No validator issues reported by the selected rules."]
    return "\n".join(
        [
            "# A20 Asset Validator Read-Only Audit",
            "",
            "This audit runs selected Isaac Sim 5.1 Asset Validator rules against the A19 candidate stage.",
            "",
            "It does not apply fixes, save the stage, step physics, initialize control, replay HDF5, or claim RL readiness.",
            "",
            "```text",
            f"status = {result['status']}",
            f"opened_in_isaac_runtime = {str(result['opened_in_isaac_runtime']).lower()}",
            f"selected_rule_count = {result['selected_rule_count']}",
            f"issue_count = {result['issue_count']}",
            f"blocking_issue_count = {result['blocking_issue_count']}",
            f"default_prim = {result['default_prim']}",
            "physics_stepped = false",
            "control_ready = false",
            "replay_ready = false",
            "training_eligible = false",
            "```",
            "",
            "## Selected Rules",
            "",
            "\n".join(f"- `{rule}`" for rule in result["selected_rules"]),
            "",
            "## Severity Counts",
            "",
            "\n".join(f"- `{key}`: {value}" for key, value in result["severity_counts"].items()) or "- none",
            "",
            "## First Issues",
            "",
            "\n".join(issue_lines),
            "",
            "## Next Gates",
            "",
            "\n".join(f"- {gate}" for gate in result["next_required_gates"]),
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    print("A20_STATUS starting_simulation_app", flush=True)
    app = _start_isaac_headless()
    try:
        print("A20_STATUS running_asset_validator", flush=True)
        result = run_audit(args.config)
        print("A20_STATUS writing_summary", flush=True)
        print(
            json.dumps(
                {
                    key: value
                    for key, value in result.items()
                    if key not in {"issues", "type_counts", "api_counts", "selected_rules"}
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(0 if result["ok"] else 1)
    finally:
        print("A20_STATUS closing_simulation_app", flush=True)
        app.close()


if __name__ == "__main__":
    main()
