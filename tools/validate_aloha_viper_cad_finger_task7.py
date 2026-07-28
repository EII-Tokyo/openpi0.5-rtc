#!/usr/bin/env python3
"""Run Task 7 against the isolated supplier-CAD follower-left diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_STAGE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "cad_finger_task5_bottle/"
    "aloha_viperx_supplier_cad_bottle_task5.usda"
)
APPROVED_SOURCE = (
    ROOT / "local_eval_assets/aloha_isaac_assets/aloha_viperx.usd"
)
TASK5_STRUCTURE = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_dynamic_structure_diagnosis.json"
)
TASK5_GEOMETRY = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_geometry_audit.json"
)
TASK5_BOTTLE = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_bottle.json"
)
SCREENSHOT_REVIEW = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_bottle_screenshot_review.json"
)
OUTPUT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task7_validation.json"
)
OUTPUT_MD = OUTPUT.with_suffix(".md")

EXPECTED_SOURCE_HASH = (
    "b24afe3678155654892c69517fc58ecd970108d68cec56b02dc6fdcb8bf4493e"
)
EXPECTED_STAGE_HASH = (
    "62697e4b25a7ec82234cc9ebd79d4a6d530a6ead0165519cbd275c0fa3f32178"
)
ROBOT_ROOT = "/workcell/vx300s_left/vx300s_left"
EXPECTED_DOF_ORDER = [
    "vx300s_left_waist",
    "vx300s_left_shoulder",
    "vx300s_left_elbow",
    "vx300s_left_forearm_roll",
    "vx300s_left_wrist_angle",
    "vx300s_left_wrist_rotate",
    "vx300s_left_left_finger",
    "vx300s_left_right_finger",
]
OFFICIAL_RULE_CATEGORIES = (
    "IsaacSim.PhysicsRules",
    "IsaacSim.RobotRules",
    "IsaacSim.SimReadyAssetRules",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def _check(
    name: str,
    status: str,
    **evidence: Any,
) -> dict[str, Any]:
    if status not in {"PASS", "FAIL", "PARTIAL", "NOT_RUN"}:
        raise ValueError(f"invalid status for {name}: {status}")
    return {"name": name, "status": status, "evidence": evidence}


def _serialize_issue(issue: Any) -> dict[str, Any]:
    severity = getattr(issue.severity, "name", str(issue.severity))
    at = issue.at.as_str() if issue.at is not None else None
    return {
        "severity": severity,
        "rule": issue.rule.__name__ if issue.rule else None,
        "message": issue.message,
        "at": at,
    }


def _run_official_rules(stage: Any) -> dict[str, Any]:
    import isaacsim.asset.validation  # noqa: F401
    import omni.asset_validator.core as av_core

    categories = []
    for category in OFFICIAL_RULE_CATEGORIES:
        rules = list(
            av_core.ValidationRulesRegistry.rules(
                category,
                enabledOnly=False,
            )
        )
        engine = av_core.ValidationEngine(init_rules=False, variants=False)
        for rule in rules:
            engine.enable_rule(rule)
        issues = sorted(
            (_serialize_issue(issue) for issue in engine.validate(stage)),
            key=lambda item: (
                item["severity"],
                item["rule"] or "",
                item["at"] or "",
                item["message"] or "",
            ),
        )
        blocking = [
            issue
            for issue in issues
            if issue["severity"] in {"ERROR", "FAILURE"}
        ]
        warnings = [
            issue for issue in issues if issue["severity"] == "WARNING"
        ]
        status = "FAIL" if blocking else "PARTIAL" if warnings else "PASS"
        categories.append(
            {
                "category": category,
                "status": status,
                "rule_count": len(rules),
                "rules": sorted(rule.__name__ for rule in rules),
                "issues": issues,
                "blocking_issue_count": len(blocking),
                "warning_count": len(warnings),
            }
        )
    return {
        "status": (
            "FAIL"
            if any(item["status"] == "FAIL" for item in categories)
            else (
                "PARTIAL"
                if any(item["status"] == "PARTIAL" for item in categories)
                else "PASS"
            )
        ),
        "categories": categories,
    }


def _finite_positive(value: Any) -> bool:
    if value is None:
        return False
    return math.isfinite(float(value)) and float(value) > 0.0


def _build_once() -> dict[str, Any]:
    from pxr import PhysxSchema
    from pxr import Usd
    from pxr import UsdPhysics
    from pxr import UsdUtils

    from tools.aloha1_mapping.cad_finger_task7 import classify_task7

    stage_path = DIAGNOSTIC_STAGE.resolve(strict=True)
    source_path = APPROVED_SOURCE.resolve(strict=True)
    source_hash = _sha256(source_path)
    stage_hash = _sha256(stage_path)
    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"unable to open diagnostic Stage: {stage_path}")

    _layers, _assets, unresolved = UsdUtils.ComputeAllDependencies(
        stage.GetRootLayer().identifier
    )
    all_articulation_roots = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    robot_articulation_roots = [
        path
        for path in all_articulation_roots
        if path.startswith("/workcell/vx300s_left/")
    ]
    joints = [
        prim
        for prim in stage.Traverse()
        if (
            str(prim.GetPath()).startswith("/workcell/joints/")
            and (
                prim.IsA(UsdPhysics.RevoluteJoint)
                or prim.IsA(UsdPhysics.PrismaticJoint)
            )
        )
    ]
    dof_order = [prim.GetName() for prim in joints]
    dof_records = []
    for prim in joints:
        axis = (
            "angular"
            if prim.IsA(UsdPhysics.RevoluteJoint)
            else "linear"
        )
        drive = UsdPhysics.DriveAPI(prim, axis)
        has_drive = bool(drive) and bool(drive.GetTypeAttr())
        applied = list(prim.GetAppliedSchemas())
        mimic = any(
            item.startswith("PhysxMimicJointAPI") for item in applied
        )
        physx_joint = PhysxSchema.PhysxJointAPI(prim)
        max_velocity = physx_joint.GetMaxJointVelocityAttr().Get()
        max_force = (
            drive.GetMaxForceAttr().Get()
            if has_drive and drive.GetMaxForceAttr()
            else None
        )
        state = PhysxSchema.JointStateAPI(prim, axis)
        state_position = (
            state.GetPositionAttr().Get()
            if state and state.GetPositionAttr()
            else None
        )
        target_position = (
            drive.GetTargetPositionAttr().Get()
            if has_drive and drive.GetTargetPositionAttr()
            else None
        )
        dof_records.append(
            {
                "name": prim.GetName(),
                "path": str(prim.GetPath()),
                "has_drive": has_drive,
                "mimic": mimic,
                "drive_type": (
                    drive.GetTypeAttr().Get() if has_drive else None
                ),
                "max_velocity": max_velocity,
                "max_force": max_force,
                "state_position": state_position,
                "target_position": target_position,
                "initial_matches_target": (
                    state_position is not None
                    and target_position is not None
                    and abs(
                        float(state_position) - float(target_position)
                    )
                    < 1.0e-8
                ),
            }
        )

    mass_records = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        mass_api = UsdPhysics.MassAPI(prim)
        mass = mass_api.GetMassAttr().Get()
        inertia = mass_api.GetDiagonalInertiaAttr().Get()
        inertia_values = (
            [float(item) for item in inertia]
            if inertia is not None
            else []
        )
        mass_records.append(
            {
                "path": str(prim.GetPath()),
                "mass": float(mass) if mass is not None else None,
                "diagonal_inertia": inertia_values,
                "finite_positive": (
                    _finite_positive(mass)
                    and len(inertia_values) == 3
                    and all(_finite_positive(item) for item in inertia_values)
                ),
            }
        )

    structure = _load(TASK5_STRUCTURE)
    geometry = _load(TASK5_GEOMETRY)
    bottle = _load(TASK5_BOTTLE)
    screenshots = _load(SCREENSHOT_REVIEW)
    first_trial = bottle["first_trial"]
    official_rules = _run_official_rules(stage)

    checks = [
        _check(
            "approved_source_hash_immutable",
            "PASS" if source_hash == EXPECTED_SOURCE_HASH else "FAIL",
            absolute_path=str(source_path),
            expected_sha256=EXPECTED_SOURCE_HASH,
            actual_sha256=source_hash,
        ),
        _check(
            "diagnostic_stage_hash",
            "PASS" if stage_hash == EXPECTED_STAGE_HASH else "FAIL",
            absolute_path=str(stage_path),
            expected_sha256=EXPECTED_STAGE_HASH,
            actual_sha256=stage_hash,
        ),
        _check(
            "external_references_resolve",
            "PASS" if not unresolved else "FAIL",
            unresolved=[str(item.path) for item in unresolved],
        ),
        _check(
            "one_robot_articulation_root",
            (
                "PASS"
                if robot_articulation_roots == [ROBOT_ROOT]
                else "FAIL"
            ),
            expected=[ROBOT_ROOT],
            actual=robot_articulation_roots,
            all_stage_articulation_roots=all_articulation_roots,
            note=(
                "Fixed environment assemblies also carry ArticulationRootAPI; "
                "the robot-scoped count is the acceptance gate."
            ),
        ),
        _check(
            "dof_name_and_order",
            "PASS" if dof_order == EXPECTED_DOF_ORDER else "FAIL",
            expected=EXPECTED_DOF_ORDER,
            actual=dof_order,
        ),
        _check(
            "all_nonfixed_joints_have_drive_or_mimic",
            (
                "PASS"
                if all(
                    record["has_drive"] or record["mimic"]
                    for record in dof_records
                )
                else "FAIL"
            ),
            dofs=dof_records,
        ),
        _check(
            "finite_positive_max_velocity_and_force",
            (
                "PASS"
                if all(
                    _finite_positive(record["max_velocity"])
                    and (
                        record["mimic"]
                        or _finite_positive(record["max_force"])
                    )
                    for record in dof_records
                )
                else "FAIL"
            ),
            dofs=dof_records,
        ),
        _check(
            "initial_joint_state_matches_drive_target",
            (
                "PASS"
                if all(
                    record["mimic"]
                    or record["initial_matches_target"]
                    for record in dof_records
                )
                else "FAIL"
            ),
            dofs=dof_records,
        ),
        _check(
            "mass_and_inertia_finite_positive",
            (
                "PASS"
                if mass_records
                and all(
                    record["finite_positive"]
                    for record in mass_records
                )
                else "FAIL"
            ),
            rigid_bodies=mass_records,
        ),
        _check(
            "first_frame_jump_and_static_structure",
            (
                "PASS"
                if structure["numeric_structure_gate"] == "PASS"
                else "FAIL"
            ),
            report=str(TASK5_STRUCTURE.resolve()),
            classification=structure["classification"],
        ),
        _check(
            "one_joint_direction_and_range",
            "PARTIAL",
            finger_direction=structure["profiles"][
                "arm_max_force_over_combined"
            ]["all_intended_directions_correct"],
            six_arm_dofs=(
                "NOT_RERUN_IN_SUPPLIER_CAD_TASK5; ARM GEOMETRY WAS NOT "
                "THE CHANGED CAUSAL VARIABLE"
            ),
        ),
        _check(
            "mimic_or_symmetric_control_mapping",
            "PARTIAL",
            control=bottle["frozen"]["control"],
            symmetric_residual_m=first_trial["states"]["hold_end"][
                "symmetric_residual_m"
            ],
            boundary="DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING",
        ),
        _check(
            "initial_overlap",
            "PARTIAL",
            geometry_report_status=geometry["status"],
            no_finger_to_finger_overlap=geometry["gates"][
                "no_finger_to_finger_overlap"
            ],
            attachment_semantics=(
                "finger-to-bar common volume retained for assembly semantic "
                "review; not silently classified as an unexpected collision"
            ),
        ),
        _check(
            "bilateral_contact_and_static_hold",
            "PASS" if bottle["status"] == "PASS" else "FAIL",
            pass_count=bottle["summary"]["pass_count"],
            trial_count=bottle["summary"]["trial_count"],
            maximum_drop_m=bottle["summary"]["maximum_drop_m"],
            persistent_penetration=first_trial["contacts"]["all_summary"][
                "persistent_penetration"
            ],
            no_constraint=not first_trial["released_hold"][
                "constraint_found"
            ],
        ),
        _check(
            "screenshot_visual_review",
            "PASS" if screenshots["status"] == "PASS" else "FAIL",
            report=str(SCREENSHOT_REVIEW.resolve()),
            raw_count=screenshots["raw_capture_count"],
            annotated_count=screenshots["annotated_capture_count"],
            screenshot_is_auxiliary=True,
        ),
        _check(
            "task5_repeat_determinism",
            (
                "PASS"
                if bottle["summary"]["deterministic"]
                and bottle["summary"]["unique_signature_count"] == 1
                else "FAIL"
            ),
            summary=bottle["summary"],
        ),
    ]
    checks.extend(
        _check(
            category["category"],
            category["status"],
            blocking_issue_count=category["blocking_issue_count"],
            warning_count=category["warning_count"],
        )
        for category in official_rules["categories"]
    )
    hard_blockers = [
        "HARD_BLOCKER_APPROVED_STAGE_CONTAINS_FOLLOWER_LEFT_ONLY",
        "HARD_BLOCKER_NO_USER_APPROVED_SUPPLIER_STAGE_LIFT_TRAJECTORY",
        "HARD_BLOCKER_UNCALIBRATED_FINGER_BOTTLE_FRICTION",
        "HARD_BLOCKER_INCOMPLETE_BOTTLE_GEOMETRY_AND_INERTIA",
        "HARD_BLOCKER_PRODUCTION_ANGULAR_TESSELLATION",
    ]
    return {
        "schema_version": 1,
        "status": classify_task7(checks, hard_blockers),
        "scope": "SUPPLIER_CAD_FOLLOWER_LEFT_DIAGNOSTIC_ONLY",
        "stage": {
            "absolute_path": str(stage_path),
            "sha256": stage_hash,
            "default_prim": str(stage.GetDefaultPrim().GetPath()),
        },
        "approved_source": {
            "absolute_path": str(source_path),
            "sha256": source_hash,
        },
        "all_stage_articulation_roots": all_articulation_roots,
        "robot_articulation_roots": robot_articulation_roots,
        "runtime_dof_order": first_trial["states"]["dof_order"],
        "stage_dof_order": dof_order,
        "dof_records": dof_records,
        "checks": checks,
        "official_rules": official_rules,
        "task5_static_hold": {
            "status": bottle["status"],
            "pass_count": bottle["summary"]["pass_count"],
            "trial_count": bottle["summary"]["trial_count"],
            "maximum_drop_m": bottle["summary"]["maximum_drop_m"],
            "drop_gate_m": bottle["frozen"]["drop_gate_m"],
            "friction_status": bottle["frozen"]["friction_status"],
            "velocity_readback_status": first_trial["released_hold"][
                "api_velocity_vs_pose_difference"
            ]["status"],
        },
        "hard_blockers": hard_blockers,
        "task8": "NOT_RUN",
        "final_default_collider_modified": False,
        "source_stage_modified": False,
        "acceptance_boundary": (
            "This validates only the isolated supplier-CAD follower_left "
            "diagnostic. It does not promote the collider/configuration, "
            "claim calibrated dynamics, validate follower_right, or run a "
            "lift trajectory."
        ),
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    check_rows = [
        f"| {item['name']} | {item['status']} |"
        for item in report["checks"]
    ]
    category_rows = [
        (
            f"| {item['category']} | {item['status']} | "
            f"{item['blocking_issue_count']} | {item['warning_count']} |"
        )
        for item in report["official_rules"]["categories"]
    ]
    path.write_text(
        "\n".join(
            [
                "# Supplier-CAD follower_left Task 7 validation",
                "",
                f"- Status: `{report['status']}`",
                f"- Stage: `{report['stage']['absolute_path']}`",
                f"- Stage SHA-256: `{report['stage']['sha256']}`",
                (
                    "- Task 5 static hold: "
                    f"`{report['task5_static_hold']['pass_count']}/"
                    f"{report['task5_static_hold']['trial_count']} PASS`, "
                    "maximum drop "
                    f"`{report['task5_static_hold']['maximum_drop_m']:.12f} m`"
                ),
                (
                    "- Repeat validation signature: "
                    f"`{report['repeat_determinism']['signatures'][0]}`"
                ),
                f"- Task 8: `{report['task8']}`",
                "",
                "| Check | Status |",
                "|---|---|",
                *check_rows,
                "",
                "| Official category | Status | Blocking | Warnings |",
                "|---|---|---:|---:|",
                *category_rows,
                "",
                "## HARD_BLOCKER",
                "",
                *[f"- `{item}`" for item in report["hard_blockers"]],
                "",
                report["acceptance_boundary"],
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    from tools.aloha1_mapping.cad_finger_task7 import deterministic_signature

    first = _build_once()
    second = _build_once()
    first_signature = deterministic_signature(first)
    second_signature = deterministic_signature(second)
    second["repeat_determinism"] = {
        "pass": first_signature == second_signature,
        "run_count": 2,
        "signatures": [first_signature, second_signature],
        "fresh_stage_open_each_run": True,
        "physics_steps": 0,
    }
    if not second["repeat_determinism"]["pass"]:
        second["status"] = "FAIL"

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(second, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = output.with_suffix(".md")
    _write_markdown(second, markdown)
    print(f"status={second['status']}")
    print(f"repeat_pass={second['repeat_determinism']['pass']}")
    print(f"json={output}")
    print(f"markdown={markdown}")
    return 0


def run() -> int:
    """Launch Kit, preserve any Python failure, and then close the app."""

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    exit_code = 1
    try:
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        extension_id = "isaacsim.asset.validation"
        if not manager.is_extension_enabled(extension_id):
            manager.set_extension_enabled_immediate(
                extension_id,
                True,  # noqa: FBT003
            )
        if not manager.is_extension_enabled(extension_id):
            raise RuntimeError(
                f"required extension disabled: {extension_id}"
            )
        exit_code = main()
    except Exception:
        traceback.print_exc()
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
