#!/usr/bin/env python3
"""Compare baseline and max-force-only Task 5 numeric drive probes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASELINE = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_drive_probe_baseline.json"
)
MAX_FORCE = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_drive_probe_max_force_only.json"
)
OUTPUT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_drive_probe_comparison.json"
)
OUTPUT_MD = OUTPUT.with_suffix(".md")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(report: dict[str, Any]) -> dict[str, Any]:
    intended = [
        result
        for trajectory in report["trajectories"]
        for result in trajectory["intended_joint_results"]
    ]
    return {
        "status": report["status"],
        "max_force_n": {
            side: report["drive_readback"][side]["max_force"]
            for side in ("left", "right")
        },
        "all_intended_directions_correct": all(
            result["direction_correct"] for result in intended
        ),
        "mean_intended_final_error_m": (
            sum(result["final_error_m"] for result in intended)
            / len(intended)
        ),
        "maximum_intended_final_error_m": max(
            result["final_error_m"] for result in intended
        ),
        "maximum_base_translation_drift_m": max(
            trajectory["base_translation_drift_m"]
            for trajectory in report["trajectories"]
        ),
        "maximum_arm_dof_drift": max(
            trajectory["maximum_arm_dof_drift"]
            for trajectory in report["trajectories"]
        ),
    }


def main() -> int:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    max_force = json.loads(MAX_FORCE.read_text(encoding="utf-8"))
    if baseline["profile"] != "baseline":
        raise RuntimeError("baseline report identity mismatch")
    if max_force["profile"] != "max_force_only":
        raise RuntimeError("max-force report identity mismatch")
    baseline_metrics = _metrics(baseline)
    max_force_metrics = _metrics(max_force)
    direction_improved = (
        not baseline_metrics["all_intended_directions_correct"]
        and max_force_metrics["all_intended_directions_correct"]
    )
    error_improved = (
        max_force_metrics["mean_intended_final_error_m"]
        < baseline_metrics["mean_intended_final_error_m"]
    )
    root_failure_persists = (
        baseline_metrics["maximum_base_translation_drift_m"] > 0.01
        and max_force_metrics["maximum_base_translation_drift_m"] > 0.01
        and baseline_metrics["maximum_arm_dof_drift"] > 3.0
        and max_force_metrics["maximum_arm_dof_drift"] > 3.0
    )
    report = {
        "schema_version": 1,
        "status": "FAIL",
        "gate": "TASK5_NO_BOTTLE_DYNAMIC_STRUCTURE",
        "classification": (
            "MAX_FORCE_IMPROVES_FINGER_TRACKING_"
            "BUT_DISJOINT_ROOT_JOINT_BLOCKS_DYNAMIC_VALIDATION"
        ),
        "inputs": {
            "baseline": {
                "absolute_path": str(BASELINE.resolve()),
                "sha256": _sha256(BASELINE),
            },
            "max_force_only": {
                "absolute_path": str(MAX_FORCE.resolve()),
                "sha256": _sha256(MAX_FORCE),
            },
        },
        "baseline": baseline_metrics,
        "max_force_only": max_force_metrics,
        "comparisons": {
            "intended_direction_improved": direction_improved,
            "mean_finger_error_improved": error_improved,
            "root_and_arm_failure_persists": root_failure_persists,
            "max_force_only_passes_1mm_gate": max_force["gates"][
                "all_intended_final_errors_within_1mm"
            ],
        },
        "evidence_classification": {
            "runtime_readback": [
                "both baseline finger drives maxForce=0",
                "both diagnostic finger drives maxForce=5",
                "baseline and diagnostic base translation drift exceeds 75 mm",
                "baseline and diagnostic arm DOF drift exceeds 3.14",
            ],
            "official_source_direct": [
                "URDF effort limit for each follower finger is 5 N",
                (
                    "local USD Physics schema defines maxForce default as inf "
                    "and requires a non-negative value"
                ),
            ],
            "engineering_inference": [
                (
                    "maxForce=0 is causal for part of finger tracking failure"
                ),
                (
                    "disjoint root/assembly frames are an independent and "
                    "higher-priority dynamic blocker"
                ),
            ],
        },
        "next_gate": {
            "action": (
                "audit rootJoint_vx300s_left local frames and create a "
                "separate frame-only diagnostic layer from computed body "
                "transforms; do not tune finger drives further yet"
            ),
            "requires_parameter_guess": False,
            "bottle_test_allowed": False,
            "screenshots_required_for_next_runtime_test": True,
        },
        "scope": {
            "bottle_contact_grasp": "NOT_RUN",
            "task8": "NOT_RUN",
            "default_or_final_asset_modified": False,
        },
    }
    if not (direction_improved and error_improved and root_failure_persists):
        report["classification"] = "INCONCLUSIVE"
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# ALOHA ViperX CAD finger Task 5 drive-probe comparison",
        "",
        f"- Overall dynamic structure gate: `{report['status']}`",
        f"- Classification: `{report['classification']}`",
        "- Bottle/contact/grasp: `NOT_RUN`",
        "- Task 8: `NOT_RUN`",
        "",
        "| Metric | Baseline | Max-force-only |",
        "|---|---:|---:|",
        (
            "| maxForce left/right N | "
            f"`{baseline_metrics['max_force_n']}` | "
            f"`{max_force_metrics['max_force_n']}` |"
        ),
        (
            "| all intended directions correct | "
            f"`{baseline_metrics['all_intended_directions_correct']}` | "
            f"`{max_force_metrics['all_intended_directions_correct']}` |"
        ),
        (
            "| mean intended final error m | "
            f"`{baseline_metrics['mean_intended_final_error_m']:.9g}` | "
            f"`{max_force_metrics['mean_intended_final_error_m']:.9g}` |"
        ),
        (
            "| max base translation drift m | "
            f"`{baseline_metrics['maximum_base_translation_drift_m']:.9g}` | "
            f"`{max_force_metrics['maximum_base_translation_drift_m']:.9g}` |"
        ),
        (
            "| max arm DOF drift | "
            f"`{baseline_metrics['maximum_arm_dof_drift']:.9g}` | "
            f"`{max_force_metrics['maximum_arm_dof_drift']:.9g}` |"
        ),
        "",
        "The 5 N profile changes only `drive:linear:physics:maxForce`. It "
        "improves finger motion but does not make the approved review Stage "
        "dynamically valid. The next isolated variable is the computed "
        "`rootJoint_vx300s_left` frame relation. No bottle test is allowed "
        "until that dynamic structure gate passes.",
        "",
    ]
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"status={report['status']}")
    print(f"classification={report['classification']}")
    print(f"json={OUTPUT.resolve()}")
    print(f"markdown={OUTPUT_MD.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
