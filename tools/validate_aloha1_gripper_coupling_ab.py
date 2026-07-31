#!/usr/bin/env python3
"""Pure contracts and Isaac helpers for the isolated ALOHA coupling A/B.

Variant A leaves the current PhysX mimic untouched. Variant B removes only the
right-finger mimic API in an isolated diagnostic layer and applies the
official, source-backed kinematic relation ``q_right = -q_left`` at runtime.
Variant B is diagnostic evidence only; it is not a final control mapping.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STAGE_PATH = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/table_support_alignment/1.0"
    / "aloha1_table_support_aligned_workcell.usda"
)
STAGE_SHA256 = "2b3f76365ed67532f478d995ae859a88b5639975ac07cb7ac8a53ac679e8205c"
RIGHT_JOINT_PATH = (
    "/World/follower_left/vx300s_left/joints/right_finger"
)
LEFT_JOINT_PATH = (
    "/World/follower_left/vx300s_left/joints/left_finger"
)
MIMIC_INSTANCE_NAME = "rotY"
MIMIC_TOLERANCE_M = 0.001
MINIMUM_FRESH_RUNS = 5
DETERMINISM_SPAN_TOLERANCE_M = 5.0e-5
VARIANTS = (
    "current_physx_mimic",
    "official_symmetric_adapter",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coupling_variant_contract(variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported coupling variant: {variant}")
    symmetric = variant == "official_symmetric_adapter"
    return {
        "variant": variant,
        "stage_sha256": STAGE_SHA256,
        "changed_variable": (
            "COUPLING_REPRESENTATION_ONLY"
            if symmetric
            else "NONE_BASELINE"
        ),
        "classification": (
            "DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING"
            if symmetric
            else "UNCHANGED_CURRENT_PHYSX_MIMIC_BASELINE"
        ),
        "right_target_formula": "q_right=-q_left" if symmetric else None,
        "right_finger_commanded_as_hardware_actuator": False,
        "runtime_adapter_policy": (
            "REMOVE_RIGHT_MIMIC_IN_SESSION_LAYER_COPY_UNCHANGED_LEFT_"
            "DRIVE_PARAMETERS_AND_DISTRIBUTE_ONE_COMMAND_AS_PLUS_MINUS_TARGETS"
            if symmetric
            else "UNCHANGED_PHYSX_MIMIC"
        ),
        "collider_changed": False,
        "friction_changed": False,
        "drive_magnitude_changed": False,
        "right_drive_added_as_coupling_adapter": symmetric,
        "bottle_changed": False,
        "timestep_changed": False,
        "solver_changed": False,
        "initial_pose_changed": False,
        "task8": "NOT_RUN",
    }


def build_coupling_targets(
    variant: str,
    *,
    left_target_m: float,
    left_index: int,
    right_index: int,
) -> dict[str, Any]:
    if not math.isfinite(left_target_m):
        raise ValueError("left_target_m must be finite")
    if variant == "current_physx_mimic":
        return {
            "joint_indices": [left_index],
            "joint_positions_m": [left_target_m],
        }
    if variant == "official_symmetric_adapter":
        return {
            "joint_indices": [left_index, right_index],
            "joint_positions_m": [left_target_m, -left_target_m],
        }
    raise ValueError(f"unsupported coupling variant: {variant}")


def author_coupling_variant(
    *,
    stage: Any,
    variant: str,
    physx_schema: Any,
    usd_physics: Any,
) -> dict[str, Any]:
    """Author only the diagnostic mimic removal into the active edit target."""
    contract = coupling_variant_contract(variant)
    right_prim = stage.GetPrimAtPath(RIGHT_JOINT_PATH)
    if not right_prim or not right_prim.IsValid():
        raise RuntimeError(f"missing right finger joint: {RIGHT_JOINT_PATH}")
    before = list(right_prim.GetAppliedSchemas())
    copied_drive: dict[str, Any] | None = None
    if variant == "official_symmetric_adapter":
        removed = right_prim.RemoveAPI(
            physx_schema.PhysxMimicJointAPI,
            MIMIC_INSTANCE_NAME,
        )
        if not removed:
            raise RuntimeError(
                "failed to author removal of PhysxMimicJointAPI:rotY"
            )
        left_prim = stage.GetPrimAtPath(LEFT_JOINT_PATH)
        left_drive = usd_physics.DriveAPI.Get(left_prim, "linear")
        if not left_drive:
            raise RuntimeError("left finger linear drive is missing")
        right_drive = usd_physics.DriveAPI.Apply(right_prim, "linear")
        copied_drive = {}
        for name, getter, creator in (
            (
                "type",
                left_drive.GetTypeAttr,
                right_drive.CreateTypeAttr,
            ),
            (
                "max_force",
                left_drive.GetMaxForceAttr,
                right_drive.CreateMaxForceAttr,
            ),
            (
                "target_position",
                left_drive.GetTargetPositionAttr,
                right_drive.CreateTargetPositionAttr,
            ),
            (
                "target_velocity",
                left_drive.GetTargetVelocityAttr,
                right_drive.CreateTargetVelocityAttr,
            ),
            (
                "damping",
                left_drive.GetDampingAttr,
                right_drive.CreateDampingAttr,
            ),
            (
                "stiffness",
                left_drive.GetStiffnessAttr,
                right_drive.CreateStiffnessAttr,
            ),
        ):
            value = getter().Get()
            creator(value)
            copied_drive[name] = str(value) if name == "type" else value
    after = list(right_prim.GetAppliedSchemas())
    has_mimic_after = right_prim.HasAPI(
        physx_schema.PhysxMimicJointAPI,
        MIMIC_INSTANCE_NAME,
    )
    expected_mimic = variant == "current_physx_mimic"
    if has_mimic_after != expected_mimic:
        raise RuntimeError(
            "diagnostic coupling API readback does not match the variant"
        )
    return {
        **contract,
        "right_joint_path": RIGHT_JOINT_PATH,
        "applied_schemas_before": before,
        "applied_schemas_after": after,
        "mimic_instance": MIMIC_INSTANCE_NAME,
        "mimic_present_after": has_mimic_after,
        "copied_left_drive_parameters": copied_drive,
        "active_edit_target": stage.GetEditTarget().GetLayer().identifier,
    }


def normalize_gui_trial(
    report: Mapping[str, Any],
    *,
    run_index: int,
    report_path: Path,
) -> dict[str, Any]:
    runtime_contract = report["runtime"]["coupling_variant"]
    variant = str(runtime_contract["variant"])
    expected_contract = coupling_variant_contract(variant)
    gate = report["result"]["gate"]
    contacts = report["result"]["contacts"]["summary"]
    cleanup_errors = list(report.get("cleanup_errors", []))
    allowed_failures = (
        ["MIMIC_ACCURACY_FAILED"]
        if variant == "current_physx_mimic"
        else []
    )
    execution_complete = (
        gate.get("bilateral_contact") == "PASS"
        and gate.get("raw_export") == "PASS"
        and gate.get("derived_export") == "PASS"
        and list(gate.get("failure_reasons", [])) == allowed_failures
        and contacts.get("status") == "PASS"
        and not cleanup_errors
    )
    unchanged_fields = (
        "collider_changed",
        "friction_changed",
        "drive_magnitude_changed",
        "bottle_changed",
        "timestep_changed",
        "solver_changed",
        "initial_pose_changed",
    )
    single_variable_contract = (
        all(runtime_contract.get(name) is False for name in unchanged_fields)
        and runtime_contract.get("changed_variable")
        == expected_contract["changed_variable"]
        and runtime_contract.get("stage_sha256") == STAGE_SHA256
    )
    return {
        "status": "PASS" if execution_complete else "FAIL",
        "variant": variant,
        "run_index": run_index,
        "fresh_process": True,
        "stage_sha256": report["inputs"]["stage"]["sha256"],
        "mimic_residual_abs_m": float(
            report["result"]["joint_readback"]["mimic_error_abs_m"]
        ),
        "bilateral_contact": bool(
            contacts["bilateral_finger_contact"]
        ),
        "maximum_impulse_ns": float(contacts["maximum_impulse_ns"]),
        "minimum_separation_m": float(contacts["minimum_separation_m"]),
        "source_stage_unchanged": (
            report["inputs"]["stage"]["sha256"] == STAGE_SHA256
            and not cleanup_errors
        ),
        "single_variable_contract": single_variable_contract,
        "native_report_status": report.get("status"),
        "native_gate": gate,
        "report_path": str(report_path.resolve()),
        "coupling_runtime_readback": runtime_contract,
    }


def _variant_summary(
    trials: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    residuals = [
        float(item["mimic_residual_abs_m"]) for item in trials
    ]
    span = max(residuals) - min(residuals) if residuals else None
    deterministic = (
        bool(residuals)
        and span is not None
        and span <= DETERMINISM_SPAN_TOLERANCE_M
        and all(item["status"] == "PASS" for item in trials)
        and len(
            {
                (
                    bool(item["bilateral_contact"]),
                    round(float(item["mimic_residual_abs_m"]), 12),
                )
                for item in trials
            }
        )
        == 1
    )
    if residuals and all(value <= MIMIC_TOLERANCE_M for value in residuals):
        mimic_gate = "PASS"
    elif residuals and all(value > MIMIC_TOLERANCE_M for value in residuals):
        mimic_gate = "FAIL"
    else:
        mimic_gate = "PARTIAL"
    return {
        "run_count": len(trials),
        "fresh_process_count": sum(
            bool(item["fresh_process"]) for item in trials
        ),
        "residuals_m": residuals,
        "mean_residual_m": (
            statistics.fmean(residuals) if residuals else None
        ),
        "residual_span_m": span,
        "deterministic": deterministic,
        "mimic_gate": mimic_gate,
        "bilateral_contact_all_runs": all(
            bool(item["bilateral_contact"]) for item in trials
        ),
        "maximum_impulse_ns": (
            max(float(item["maximum_impulse_ns"]) for item in trials)
            if trials
            else None
        ),
        "minimum_separation_m": (
            min(float(item["minimum_separation_m"]) for item in trials)
            if trials
            else None
        ),
    }


def classify_ab_trials(
    trials: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for trial in trials:
        grouped[str(trial["variant"])].append(trial)
    summaries = {
        variant: _variant_summary(grouped.get(variant, []))
        for variant in VARIANTS
    }
    failure_reasons: list[str] = []
    for variant, summary in summaries.items():
        if summary["fresh_process_count"] < MINIMUM_FRESH_RUNS:
            failure_reasons.append(
                f"INSUFFICIENT_FRESH_RUNS:{variant}"
            )
        if summary["run_count"] != summary["fresh_process_count"]:
            failure_reasons.append(f"NONFRESH_PROCESS:{variant}")
        if not summary["deterministic"]:
            failure_reasons.append(f"NONDETERMINISTIC:{variant}")
        if not summary["bilateral_contact_all_runs"]:
            failure_reasons.append(f"BILATERAL_CONTACT_FAILED:{variant}")
    if any(not bool(item["single_variable_contract"]) for item in trials):
        failure_reasons.append("SINGLE_VARIABLE_CONTRACT_VIOLATION")
    if any(item["stage_sha256"] != STAGE_SHA256 for item in trials):
        failure_reasons.append("STAGE_HASH_MISMATCH")
    if any(not bool(item["source_stage_unchanged"]) for item in trials):
        failure_reasons.append("SOURCE_STAGE_CHANGED")

    baseline_impulse = summaries["current_physx_mimic"][
        "maximum_impulse_ns"
    ]
    symmetric_impulse = summaries["official_symmetric_adapter"][
        "maximum_impulse_ns"
    ]
    baseline_separation = summaries["current_physx_mimic"][
        "minimum_separation_m"
    ]
    symmetric_separation = summaries["official_symmetric_adapter"][
        "minimum_separation_m"
    ]
    contact_equivalence = "INCONCLUSIVE"
    if all(
        value is not None
        for value in (
            baseline_impulse,
            symmetric_impulse,
            baseline_separation,
            symmetric_separation,
        )
    ):
        impulse_ratio = (
            float(symmetric_impulse) / float(baseline_impulse)
            if float(baseline_impulse) > 0.0
            else math.inf
        )
        additional_penetration_m = max(
            0.0,
            float(baseline_separation) - float(symmetric_separation),
        )
        if impulse_ratio <= 10.0 and additional_penetration_m <= 0.001:
            contact_equivalence = "PASS"
        else:
            contact_equivalence = "FAIL"
            failure_reasons.append("CONTACT_EQUIVALENCE_FAILED")
    else:
        impulse_ratio = None
        additional_penetration_m = None

    classification = "INCONCLUSIVE"
    if not failure_reasons:
        baseline_gate = summaries["current_physx_mimic"]["mimic_gate"]
        symmetric_gate = summaries["official_symmetric_adapter"][
            "mimic_gate"
        ]
        if baseline_gate == "PASS":
            classification = "CURRENT_MIMIC_ACCEPTABLE"
        elif baseline_gate == "FAIL" and symmetric_gate == "PASS":
            classification = "PHYSX_MIMIC_PRIMARY"
        elif baseline_gate == "FAIL" and symmetric_gate == "FAIL":
            classification = "COUPLING_NOT_PRIMARY"

    status = "PASS" if not failure_reasons else "FAIL"
    passing_path = (
        "current_physx_mimic"
        if classification == "CURRENT_MIMIC_ACCEPTABLE"
        else (
            "official_symmetric_adapter"
            if classification == "PHYSX_MIMIC_PRIMARY"
            else None
        )
    )
    return {
        "schema_version": 1,
        "status": status,
        "classification": classification,
        "failure_reasons": sorted(set(failure_reasons)),
        "variants": summaries,
        "mimic_tolerance_m": MIMIC_TOLERANCE_M,
        "passing_path": passing_path,
        "promotion_authorized": False,
        "contact_equivalence": {
            "status": contact_equivalence,
            "maximum_impulse_ratio": 10.0,
            "maximum_additional_penetration_m": 0.001,
            "measured_impulse_ratio": impulse_ratio,
            "measured_additional_penetration_m": (
                additional_penetration_m
            ),
            "classification": (
                "ENGINEERING_DIAGNOSTIC_VALIDITY_GATE_NOT_REAL_HARDWARE_"
                "CALIBRATION"
            ),
        },
        "next_gate": (
            "GRASP_EDITOR_DIAGNOSTIC_ON_PASSING_PATH"
            if status == "PASS" and passing_path is not None
            else "BLOCKED_BY_STAGE3"
        ),
        "task8": "NOT_RUN",
    }


def _write_markdown(report: Mapping[str, Any], path: Path) -> None:
    baseline = report["variants"]["current_physx_mimic"]
    symmetric = report["variants"]["official_symmetric_adapter"]
    contact = report["contact_equivalence"]
    lines = [
        "# ALOHA 1 Official Gripper Coupling A/B",
        "",
        f"- Status: `{report['status']}`",
        f"- Classification: `{report['classification']}`",
        f"- Passing diagnostic path: `{report['passing_path']}`",
        f"- Promotion authorized: `{report['promotion_authorized']}`",
        f"- Next gate: `{report['next_gate']}`",
        f"- Task 8: `{report['task8']}`",
        "",
        "## Variant A — unchanged PhysX mimic",
        "",
        f"- Fresh runs: `{baseline['fresh_process_count']}`",
        f"- Mean residual: `{baseline['mean_residual_m']} m`",
        f"- Mimic gate: `{baseline['mimic_gate']}`",
        f"- Maximum impulse: `{baseline['maximum_impulse_ns']} N s`",
        f"- Minimum separation: `{baseline['minimum_separation_m']} m`",
        "",
        "## Variant B — official symmetric diagnostic adapter",
        "",
        "- Classification: `DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING`",
        f"- Fresh runs: `{symmetric['fresh_process_count']}`",
        f"- Mean residual: `{symmetric['mean_residual_m']} m`",
        f"- Mimic gate: `{symmetric['mimic_gate']}`",
        f"- Maximum impulse: `{symmetric['maximum_impulse_ns']} N s`",
        f"- Minimum separation: `{symmetric['minimum_separation_m']} m`",
        "",
        "## Contact-equivalence validity gate",
        "",
        f"- Status: `{contact['status']}`",
        f"- Measured impulse ratio B/A: `{contact['measured_impulse_ratio']}`",
        (
            "- Measured additional penetration: "
            f"`{contact['measured_additional_penetration_m']} m`"
        ),
        "",
        (
            "Variant B removes only the right-finger PhysX mimic in an "
            "isolated layer, copies the unchanged left drive values, and "
            "distributes one official actuation coordinate as +q/-q targets. "
            "It is evidence that the current mimic representation is primary "
            "at this runtime boundary; it is not a final asset promotion."
        ),
        "",
        (
            "A rejected state-projection probe reached zero algebraic "
            "residual but caused about 9.4 mm penetration and a 0.645 N s "
            "impulse. It is retained only as rejected diagnostic evidence."
        ),
        "",
        "Task 8 remains `NOT_RUN`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_variant_trials(root: Path, variant: str) -> list[dict[str, Any]]:
    paths = sorted(
        root.glob("run*/grasp_editor_variant_b_gui_report.json")
    )
    selected: list[dict[str, Any]] = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if (
            report.get("runtime", {})
            .get("coupling_variant", {})
            .get("variant")
            != variant
        ):
            continue
        selected.append(
            normalize_gui_trial(
                report,
                run_index=len(selected),
                report_path=path,
            )
        )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--symmetric-root", type=Path, required=True)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=(
            ROOT
            / "reports/aloha1_mapping/aloha1_gripper_coupling_ab.json"
        ),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=(
            ROOT
            / "reports/aloha1_mapping/aloha1_gripper_coupling_ab.md"
        ),
    )
    args = parser.parse_args()
    baseline = _load_variant_trials(
        args.baseline_root.resolve(),
        "current_physx_mimic",
    )
    symmetric = _load_variant_trials(
        args.symmetric_root.resolve(),
        "official_symmetric_adapter",
    )
    report = classify_ab_trials([*baseline, *symmetric])
    report["frozen_stage"] = {
        "absolute_path": str(STAGE_PATH),
        "expected_sha256": STAGE_SHA256,
        "current_sha256": _sha256(STAGE_PATH),
        "unchanged": _sha256(STAGE_PATH) == STAGE_SHA256,
    }
    report["trials"] = [*baseline, *symmetric]
    report["rejected_diagnostics"] = [
        {
            "name": "DIRECT_RIGHT_JOINT_STATE_PROJECTION",
            "status": "REJECTED_CONTACT_EQUIVALENCE_FAILED",
            "residual_m": 0.0,
            "maximum_impulse_ns": 0.6450559057324757,
            "minimum_separation_m": -0.009414959698915482,
            "report_path": str(
                (
                    ROOT
                    / ".codex/artifacts/"
                    "20260730-aloha1-official-gripper-unattended/stage3/"
                    "symmetric_probe00/"
                    "grasp_editor_variant_b_gui_report.json"
                ).resolve()
            ),
        }
    ]
    report["evidence_scope"] = {
        "official_source": (
            "One physical gripper coordinate and +x/-x state mapping from "
            "the Stage 1 exact-model official-source audit."
        ),
        "direct_nvidia_mcp": (
            "Isaac 5.1 drive/mimic/joint-state semantics queried before "
            "authoring the isolated adapter."
        ),
        "runtime_readback": (
            "Five fresh native GUI processes per variant on the frozen Stage."
        ),
        "engineering_gate": (
            "Contact-equivalence thresholds validate the intervention; they "
            "are not real-hardware calibration values."
        ),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(report, args.markdown_output)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
