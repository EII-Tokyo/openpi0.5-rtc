#!/usr/bin/env python3
"""Headless machine-readable validation for Stationary ALOHA 1 assets."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from tools.aloha1_mapping.gripper_validation import classify_repeat_determinism
from tools.aloha1_mapping.physicsrules import build_physicsrules_report
from tools.aloha1_mapping.validation import build_validation_plan
from tools.aloha1_mapping.validation import classify_validation
from tools.aloha1_mapping.validation import load_required_machine_report


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(
    name: str,
    passed: bool,  # noqa: FBT001
    **evidence: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "evidence": evidence,
    }


def _asset_path_text(value: Any) -> str:
    path = getattr(value, "path", None)
    return str(path) if path is not None else str(value)


def _serialize_issue(issue: Any) -> dict[str, Any]:
    severity = getattr(issue.severity, "name", str(issue.severity))
    at = issue.at.as_str() if issue.at is not None else None
    return {
        "severity": severity,
        "rule": issue.rule.__name__ if issue.rule else None,
        "message": issue.message,
        "at": at,
    }


def _run_official_rules(
    targets_by_category: Mapping[str, Sequence[tuple[str, Any]]],
) -> dict[str, Any]:
    import isaacsim.asset.validation  # noqa: F401 - registers Isaac rules
    import omni.asset_validator.core as av_core

    category_reports = []
    for category, targets in targets_by_category.items():
        rules = list(av_core.ValidationRulesRegistry.rules(category, enabledOnly=False))
        target_reports = []
        for target_name, target_stage in targets:
            engine = av_core.ValidationEngine(init_rules=False, variants=False)
            for rule in rules:
                engine.enable_rule(rule)
            result = engine.validate(target_stage)
            issues = sorted(
                (_serialize_issue(issue) for issue in result),
                key=lambda item: (
                    item["severity"],
                    item["rule"] or "",
                    item["at"] or "",
                    item["message"] or "",
                ),
            )
            blocking = [issue for issue in issues if issue["severity"] in {"ERROR", "FAILURE"}]
            warnings = [issue for issue in issues if issue["severity"] == "WARNING"]
            target_reports.append(
                {
                    "target": target_name,
                    "status": ("FAIL" if blocking else "PARTIAL" if warnings else "PASS"),
                    "issues": issues,
                }
            )
        category_reports.append(
            {
                "category": category,
                "status": classify_validation(target_reports, []),
                "rule_count": len(rules),
                "rules": [rule.__name__ for rule in rules],
                "targets": target_reports,
                "issues": [issue for target in target_reports for issue in target["issues"]],
            }
        )
    return {
        "schema_version": 1,
        "status": classify_validation(category_reports, []),
        "categories": category_reports,
    }


def _inspect_stage(plan: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    from pxr import PhysxSchema
    from pxr import Usd
    from pxr import UsdPhysics
    from pxr import UsdUtils

    diagnostic_delegate = UsdUtils.CoalescingDiagnosticDelegate()
    stage = Usd.Stage.Open(plan["workcell"])
    if stage is None:
        raise RuntimeError(f"unable to open workcell: {plan['workcell']}")
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(stage.GetRootLayer().identifier)
    diagnostics = [
        {
            "code": item.diagnosticCodeString,
            "commentary": item.commentary,
        }
        for item in diagnostic_delegate.TakeUncoalescedDiagnostics()
        if item.diagnosticCodeString != "TF_DIAGNOSTIC_STATUS_TYPE"
    ]
    articulation_roots = [
        str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    checks = [
        _check(
            "external_references_resolve",
            not unresolved and not diagnostics,
            unresolved=[_asset_path_text(item) for item in unresolved],
            composition_diagnostics=diagnostics,
            dependency_layer_count=len(layers),
            dependency_asset_count=len(assets),
        ),
        _check(
            "articulation_count",
            len(articulation_roots) == plan["expected_articulation_count"],
            expected=plan["expected_articulation_count"],
            actual=len(articulation_roots),
            roots=articulation_roots,
        ),
    ]
    expected_roots = {robot["articulation_prim"] for robot in plan["robots"]}
    checks.append(
        _check(
            "one_expected_articulation_root_per_robot",
            set(articulation_roots) == expected_roots,
            expected=sorted(expected_roots),
            actual=sorted(articulation_roots),
        )
    )

    robot_static = []
    for robot in plan["robots"]:
        joint_root = f"{robot['robot_prim']}/joints"
        dof_records = []
        actual_order = [
            prim.GetName()
            for prim in stage.Traverse()
            if str(prim.GetPath()).startswith(joint_root + "/")
            and (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint))
        ]
        # Stage traversal follows authored joint order for this imported asset.
        for name in robot["dof_order"]:
            prim = stage.GetPrimAtPath(f"{joint_root}/{name}")
            axis = "linear" if prim.IsA(UsdPhysics.PrismaticJoint) else "angular"
            applied = set(prim.GetAppliedSchemas())
            mimic = any(item.startswith("PhysxMimicJointAPI") for item in applied) or prim.HasAPI(
                PhysxSchema.PhysxMimicJointAPI
            )
            drive = UsdPhysics.DriveAPI(prim, axis)
            has_drive = bool(drive) and bool(drive.GetTypeAttr())
            state_api = PhysxSchema.JointStateAPI(prim, axis)
            state_position = state_api.GetPositionAttr().Get() if state_api and state_api.GetPositionAttr() else None
            target_position = (
                drive.GetTargetPositionAttr().Get() if has_drive and drive.GetTargetPositionAttr() else None
            )
            physx_joint = PhysxSchema.PhysxJointAPI(prim)
            max_velocity = physx_joint.GetMaxJointVelocityAttr().Get()
            max_force = drive.GetMaxForceAttr().Get() if has_drive and drive.GetMaxForceAttr() else None
            stiffness = drive.GetStiffnessAttr().Get() if has_drive and drive.GetStiffnessAttr() else 0.0
            damping = drive.GetDampingAttr().Get() if has_drive and drive.GetDampingAttr() else 0.0
            dof_records.append(
                {
                    "name": name,
                    "mimic": mimic,
                    "has_drive": has_drive,
                    "drive_type": (drive.GetTypeAttr().Get() if has_drive else None),
                    "state_position": state_position,
                    "target_position": target_position,
                    "initial_matches_target": (
                        mimic
                        or (
                            state_position is not None
                            and target_position is not None
                            and abs(state_position - target_position) < 1.0e-8
                        )
                    ),
                    "max_velocity": max_velocity,
                    "max_force": max_force,
                    "stiffness": stiffness,
                    "damping": damping,
                }
            )
        robot_static.append(
            {
                "name": robot["name"],
                "expected_dof_order": robot["dof_order"],
                "stage_traversal_nonfixed_joint_order": actual_order,
                "dofs": dof_records,
            }
        )
        checks.extend(
            [
                _check(
                    f"{robot['name']}_dof_names_present",
                    all(stage.GetPrimAtPath(f"{joint_root}/{name}") for name in robot["dof_order"]),
                    expected=robot["dof_order"],
                ),
                _check(
                    f"{robot['name']}_drive_or_mimic",
                    all(item["has_drive"] or item["mimic"] for item in dof_records),
                    dofs=dof_records,
                ),
                _check(
                    f"{robot['name']}_mimic_has_zero_active_drive",
                    all(
                        (not item["mimic"])
                        or (not item["has_drive"] and item["stiffness"] == 0.0 and item["damping"] == 0.0)
                        for item in dof_records
                    ),
                    dofs=dof_records,
                ),
                _check(
                    f"{robot['name']}_finite_positive_limits",
                    all(
                        math.isfinite(float(item["max_velocity"]))
                        and float(item["max_velocity"]) > 0
                        and (
                            item["mimic"]
                            or (
                                item["max_force"] is not None
                                and math.isfinite(float(item["max_force"]))
                                and float(item["max_force"]) > 0
                            )
                        )
                        for item in dof_records
                    ),
                    dofs=dof_records,
                ),
                _check(
                    f"{robot['name']}_initial_matches_drive_target",
                    all(item["initial_matches_target"] for item in dof_records),
                    dofs=dof_records,
                ),
            ]
        )

    mass_records = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        mass_api = UsdPhysics.MassAPI(prim)
        mass = mass_api.GetMassAttr().Get()
        inertia = mass_api.GetDiagonalInertiaAttr().Get()
        values = [float(mass)] + [float(item) for item in inertia]
        mass_records.append(
            {
                "prim": str(prim.GetPath()),
                "mass": float(mass),
                "diagonal_inertia": [float(item) for item in inertia],
                "finite_positive": all(math.isfinite(item) and item > 0 for item in values),
            }
        )
    checks.append(
        _check(
            "mass_and_inertia_finite_positive",
            bool(mass_records) and all(item["finite_positive"] for item in mass_records),
            rigid_bodies=mass_records,
        )
    )
    return stage, {
        "schema_version": 1,
        "checks": checks,
        "robots": robot_static,
        "mass_records": mass_records,
    }


def _runtime_suite(
    plan: Mapping[str, Any],
    *,
    curve_path: Path,
) -> dict[str, Any]:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import open_stage
    from isaacsim.core.utils.types import ArticulationAction

    World.clear_instance()
    if not open_stage(plan["workcell"]):
        raise RuntimeError(f"unable to open runtime stage: {plan['workcell']}")
    world = World(
        stage_units_in_meters=1.0,
        backend="numpy",
        device="cpu",
        physics_dt=plan["runtime"]["physics_dt_s"],
        rendering_dt=plan["runtime"]["physics_dt_s"],
    )
    robots = []
    for robot_plan in plan["robots"]:
        articulation = SingleArticulation(
            prim_path=robot_plan["articulation_prim"],
            name=robot_plan["name"],
            reset_xform_properties=False,
        )
        world.scene.add(articulation)
        robots.append((robot_plan, articulation))
    world.reset()
    runtime_checks = []
    runtime_robots = []
    curves: list[dict[str, Any]] = []
    for robot_plan, articulation in robots:
        actual_order = list(articulation.dof_names)
        runtime_checks.append(
            _check(
                f"{robot_plan['name']}_runtime_dof_order",
                actual_order == robot_plan["dof_order"],
                expected=robot_plan["dof_order"],
                actual=actual_order,
            )
        )
        initial = articulation.get_joint_positions().copy()
        world.step(render=False)
        after_first = articulation.get_joint_positions().copy()
        first_jump = float(np.max(np.abs(after_first - initial)))
        runtime_checks.append(
            _check(
                f"{robot_plan['name']}_first_frame_jump",
                first_jump <= plan["runtime"]["first_frame_jump_tolerance"],
                max_abs_jump=first_jump,
                tolerance=plan["runtime"]["first_frame_jump_tolerance"],
                before=initial.tolist(),
                after=after_first.tolist(),
            )
        )
        world.reset()
        hold_start = articulation.get_joint_positions().copy()
        for _ in range(plan["runtime"]["static_steps"]):
            world.step(render=False)
        hold_end = articulation.get_joint_positions().copy()
        hold_drift = float(np.max(np.abs(hold_end - hold_start)))
        runtime_checks.append(
            _check(
                f"{robot_plan['name']}_static_hold",
                hold_drift <= plan["runtime"]["static_position_tolerance"],
                max_abs_drift=hold_drift,
                tolerance=plan["runtime"]["static_position_tolerance"],
                start=hold_start.tolist(),
                end=hold_end.tolist(),
            )
        )

        properties = articulation.dof_properties
        joint_tests = []
        active_indices = list(range(len(actual_order) - 1))
        for joint_index in active_indices:
            world.reset()
            start = articulation.get_joint_positions().copy()
            target = start.copy()
            delta = (
                plan["runtime"]["prismatic_delta_m"]
                if actual_order[joint_index] == "left_finger"
                else plan["runtime"]["revolute_delta_rad"]
            )
            lower = float(properties[joint_index]["lower"])
            upper = float(properties[joint_index]["upper"])
            candidate = start[joint_index] + delta
            if math.isfinite(upper):
                candidate = min(candidate, upper - abs(delta) * 0.1)
            if math.isfinite(lower):
                candidate = max(candidate, lower + abs(delta) * 0.1)
            commanded_delta = candidate - start[joint_index]
            target[joint_index] = candidate
            articulation.get_articulation_controller().apply_action(ArticulationAction(joint_positions=target))
            for step in range(plan["runtime"]["one_joint_steps"]):
                world.step(render=False)
                position = articulation.get_joint_positions().copy()
                curves.append(
                    {
                        "robot": robot_plan["name"],
                        "commanded_joint": actual_order[joint_index],
                        "step": step,
                        **{name: float(position[index]) for index, name in enumerate(actual_order)},
                    }
                )
            end = articulation.get_joint_positions().copy()
            measured_delta = float(end[joint_index] - start[joint_index])
            direction_ok = measured_delta * commanded_delta > 0
            readback_ok = abs(measured_delta) >= plan["runtime"]["readback_minimum"]
            range_ok = all(
                (
                    not bool(properties[index]["hasLimits"])
                    or (
                        float(properties[index]["lower"]) - 1.0e-5
                        <= float(end[index])
                        <= float(properties[index]["upper"]) + 1.0e-5
                    )
                )
                for index in range(len(actual_order))
            )
            excluded = {joint_index}
            if actual_order[joint_index] == "left_finger":
                excluded.add(actual_order.index("right_finger"))
            unexpected = max(
                (abs(float(end[index] - start[index])) for index in range(len(actual_order)) if index not in excluded),
                default=0.0,
            )
            test_status = (
                direction_ok and readback_ok and range_ok and unexpected <= plan["runtime"]["static_position_tolerance"]
            )
            joint_tests.append(
                {
                    "joint": actual_order[joint_index],
                    "status": "PASS" if test_status else "FAIL",
                    "commanded_delta": float(commanded_delta),
                    "measured_delta": measured_delta,
                    "direction_ok": direction_ok,
                    "readback_ok": readback_ok,
                    "range_ok": range_ok,
                    "max_unexpected_delta": unexpected,
                    "start": start.tolist(),
                    "end": end.tolist(),
                }
            )
        runtime_checks.append(
            _check(
                f"{robot_plan['name']}_one_joint_at_a_time",
                all(item["status"] == "PASS" for item in joint_tests),
                tests=joint_tests,
            )
        )
        runtime_robots.append(
            {
                "name": robot_plan["name"],
                "dof_order": actual_order,
                "joint_tests": joint_tests,
            }
        )

    curve_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["robot", "commanded_joint", "step"] + plan["robots"][0]["dof_order"]
    with curve_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(curves)
    return {
        "schema_version": 1,
        "checks": runtime_checks,
        "robots": runtime_robots,
        "curve_csv": str(curve_path.resolve()),
    }


def run_validation(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    report_root = root / "reports/aloha1_mapping"
    summary_path = report_root / "validation_summary.json"
    previous_signature = None
    if summary_path.is_file():
        previous_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        previous_signature = previous_summary.get("determinism", {}).get("current_signature")
    plan = build_validation_plan(root)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
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
            raise RuntimeError(f"required extension disabled: {extension_id}")

        stage, static_report = _inspect_stage(plan)
        from pxr import Usd

        robot_rule_targets = [
            (
                robot["name"],
                Usd.Stage.Open(robot["source_robot_asset"]),
            )
            for robot in plan["robots"]
        ]
        if any(item[1] is None for item in robot_rule_targets):
            raise RuntimeError("unable to open follower source asset for RobotRules")
        asset_validator = _run_official_rules(
            {
                "IsaacSim.PhysicsRules": robot_rule_targets,
                "IsaacSim.RobotRules": robot_rule_targets,
                "IsaacSim.SimReadyAssetRules": [("workcell", stage)],
            }
        )
        _write_json(
            report_root / "asset_validator_report.json",
            asset_validator,
        )
        physicsrules_classification = build_physicsrules_report(asset_validator)
        _write_json(
            report_root / "physics_rules_classification.json",
            physicsrules_classification,
        )

        runtime_report = _runtime_suite(
            plan,
            curve_path=report_root / "one_joint_curves.csv",
        )
        gripper_check = load_required_machine_report(
            Path(plan["gripper_validation_report"]),
            name="Task5.GripperValidation",
            accepted_statuses=plan["required_task5_gripper_statuses"],
        )
        all_checks = static_report["checks"] + runtime_report["checks"] + [gripper_check]
        hard_blockers = [
            "measured follower mounting transforms",
            "tabletop/base and pipe fixture calibration",
            "bottle geometry and dynamics",
            "four-camera intrinsics/extrinsics/frame rates",
            "measured robot dynamics and fingertip friction",
            "motor response and Gain Tuner calibration for force drive",
            "real gripper motor-angle/aperture calibration",
        ]
        official_category_checks = [
            {
                "name": item["category"],
                "status": item["status"],
            }
            for item in asset_validator["categories"]
        ]
        overall = classify_validation(
            all_checks + official_category_checks,
            hard_blockers,
        )
        physics_report = {
            "schema_version": 1,
            "status": classify_validation(all_checks, hard_blockers),
            "static": static_report,
            "runtime_checks": runtime_report["checks"],
            "gripper_validation": gripper_check,
            "dynamics_calibration_status": "HARD_BLOCKER",
            "physics_rules_classification": physicsrules_classification,
        }
        control_report = {
            "schema_version": 1,
            "status": classify_validation(
                [
                    check
                    for check in runtime_report["checks"] + [gripper_check]
                    if "dof_order" in check["name"]
                    or "one_joint" in check["name"]
                    or check["name"] == "Task5.GripperValidation"
                ],
                ["real gripper motor-angle/aperture calibration"],
            ),
            "robots": runtime_report["robots"],
            "curve_csv": runtime_report["curve_csv"],
        }
        collision_issues = [
            issue
            for category in asset_validator["categories"]
            if category["category"] == "IsaacSim.PhysicsRules"
            for issue in category["issues"]
            if issue["rule"] == "NonAdjacentCollisionMeshesDoNotClash"
        ]
        collision_report = {
            "schema_version": 1,
            "status": "FAIL" if collision_issues else "PASS",
            "non_adjacent_collision_issues": collision_issues,
            "workcell_placeholder_collisions_enabled": False,
            "gripper_validation": gripper_check,
        }
        _write_json(report_root / "physics_report.json", physics_report)
        _write_json(report_root / "control_mapping_report.json", control_report)
        _write_json(report_root / "collision_report.json", collision_report)
        determinism_paths = [
            report_root / "asset_validator_report.json",
            report_root / "physics_rules_classification.json",
            report_root / "physics_report.json",
            report_root / "control_mapping_report.json",
            report_root / "collision_report.json",
            report_root / "one_joint_curves.csv",
            Path(plan["gripper_validation_report"]),
        ]
        artifact_hashes = {str(path.resolve()): _sha256(path) for path in determinism_paths}
        current_signature = hashlib.sha256(
            json.dumps(
                artifact_hashes,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        determinism = classify_repeat_determinism(
            previous_signature,
            current_signature,
        )
        determinism["artifact_hashes"] = artifact_hashes
        determinism_check = {
            "name": "headless_repeat_deterministic",
            "status": determinism["status"],
            "evidence": determinism,
        }
        final_checks = all_checks + official_category_checks + [determinism_check]
        overall = classify_validation(final_checks, hard_blockers)
        summary = {
            "schema_version": 1,
            "status": overall,
            "isaac_sim": "5.1.0.0",
            "checks": final_checks,
            "hard_blockers": hard_blockers,
            "physics_rules_classification_report": str((report_root / "physics_rules_classification.json").resolve()),
            "determinism": determinism,
            "optimization_gate": ("BLOCKED_UNTIL_VALIDATION_PASS" if overall != "PASS" else "OPEN"),
        }
        _write_json(summary_path, summary)
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _write_json(report_root / "validation_summary.json", failure)
        raise
    finally:
        app.close()
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args(argv)
    run_validation(arguments.project_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
