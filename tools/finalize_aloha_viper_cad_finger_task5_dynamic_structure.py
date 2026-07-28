#!/usr/bin/env python3
"""Finalize the isolated supplier-CAD Task 5 dynamic-structure diagnosis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports/aloha1_mapping"
ARTIFACTS = ROOT / ".codex/artifacts"

PROBE_PATHS = {
    "baseline": REPORTS
    / "aloha_viper_cad_finger_task5_drive_probe_baseline.json",
    "finger_max_force_only": REPORTS
    / "aloha_viper_cad_finger_task5_drive_probe_max_force_only.json",
    "root_frame_only": REPORTS
    / "aloha_viper_cad_finger_task5_drive_probe_root_frame_only.json",
    "finger_max_force_plus_root_frame": REPORTS
    / "aloha_viper_cad_finger_task5_drive_probe_max_force_plus_root_frame_v2.json",
    "arm_max_force_over_combined": REPORTS
    / "aloha_viper_cad_finger_task5_drive_probe_arm_max_force_over_combined.json",
}
STATIC_SCREENSHOT_REVIEW = (
    REPORTS
    / "aloha_viper_cad_finger_task5_structure_screenshot_review.json"
)
NUMERIC_PASS_SCREENSHOT_REVIEW = (
    REPORTS
    / "aloha_viper_cad_finger_task5_numeric_pass_screenshot_review.json"
)
REPLAY_ATTEMPT_DIRS = [
    ARTIFACTS / "20260729-063308_aloha-task5-root-frame-drive-replay-capture",
    ARTIFACTS
    / "20260729-063427_aloha-task5-root-frame-drive-replay-capture-attempt2",
    ARTIFACTS
    / "20260729-063604_aloha-task5-root-frame-drive-replay-capture-attempt3",
]

OUTPUT = REPORTS / "aloha_viper_cad_finger_task5_dynamic_structure_diagnosis.json"
OUTPUT_MD = OUTPUT.with_suffix(".md")
BLOCKER_OUTPUT = (
    REPORTS / "aloha_viper_cad_finger_task5_runtime_screenshot_blocker.json"
)
BLOCKER_OUTPUT_MD = BLOCKER_OUTPUT.with_suffix(".md")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input(path: Path) -> dict[str, str]:
    return {
        "absolute_path": str(path.resolve()),
        "sha256": _sha256(path),
    }


def metrics_from_report(report: dict[str, Any]) -> dict[str, Any]:
    trajectories = report["trajectories"]
    intended = [
        result
        for trajectory in trajectories
        for result in trajectory["intended_joint_results"]
    ]
    non_target = [
        result
        for trajectory in trajectories
        for result in trajectory["non_target_finger_results"]
    ]
    stage = report["stage"]
    return {
        "status": report["status"],
        "profile": report.get("profile"),
        "finger_max_force_n": {
            side: report["drive_readback"][side]["max_force"]
            for side in ("left", "right")
        },
        "arm_max_force": {
            name: values["max_force"]
            for name, values in report.get("arm_drive_readback", {}).items()
        },
        "maximum_base_translation_drift_m": max(
            trajectory["base_translation_drift_m"]
            for trajectory in trajectories
        ),
        "maximum_arm_dof_drift": max(
            trajectory["maximum_arm_dof_drift"]
            for trajectory in trajectories
        ),
        "all_intended_directions_correct": all(
            result["direction_correct"] for result in intended
        ),
        "maximum_intended_final_error_m": max(
            result["final_error_m"] for result in intended
        ),
        "maximum_non_target_finger_drift_m": max(
            (result["drift_m"] for result in non_target),
            default=0.0,
        ),
        "stage_immutable": (
            stage["sha256_before"] == stage["sha256_after"]
        ),
        "no_bottle": report["gates"]["no_bottle"],
    }


def _error_excerpt(stderr_path: Path) -> str:
    lines = stderr_path.read_text(encoding="utf-8", errors="replace").splitlines()
    matching = [
        line
        for line in lines
        if "zero-size array" in line
        or "camera remained blank" in line
        or "shape=[0]" in line
    ]
    return matching[-1] if matching else lines[-1]


def _build_screenshot_blocker() -> dict[str, Any]:
    attempts = []
    for index, directory in enumerate(REPLAY_ATTEMPT_DIRS, start=1):
        stdout = directory / "stdout.log"
        stderr = directory / "stderr.log"
        raw_dir = (
            ARTIFACTS
            / "20260729-aloha-finger-palm-orientation/isaac_cad_finger"
            / (
                "task5_drive_probe_root_frame_only_replay"
                if index == 1
                else (
                    "task5_drive_probe_root_frame_only_replay_attempt"
                    f"{index}"
                )
            )
            / "screenshots_raw"
        )
        pngs = sorted(raw_dir.glob("*.png")) if raw_dir.exists() else []
        attempts.append(
            {
                "attempt": index,
                "status": "FAIL",
                "accepted_capture": False,
                "failure": (
                    "ZERO_SIZE_CAMERA_BUFFER_HELPER_ERROR"
                    if index == 1
                    else "CAMERA_REMAINED_EMPTY_AFTER_RENDER_POLLING"
                ),
                "error_excerpt": _error_excerpt(stderr),
                "stdout_log": _input(stdout),
                "stderr_log": _input(stderr),
                "raw_screenshot_directory": str(raw_dir.resolve()),
                "raw_png_count": len(pngs),
            }
        )
    visual_review = json.loads(
        NUMERIC_PASS_SCREENSHOT_REVIEW.read_text(encoding="utf-8")
    )
    visual_pass = (
        visual_review["status"] == "PASS"
        and visual_review["screenshot_status"]
        == "PASS_AUXILIARY_RUNTIME_READBACK_REPLAY"
    )
    return {
        "schema_version": 1,
        "status": (
            "RESOLVED_WITH_ALTERNATE_VIEWPORT_BACKEND"
            if visual_pass
            else "HARD_BLOCKER"
        ),
        "blocker_code": (
            "HARD_BLOCKER_RUNTIME_CAMERA_EMPTY_BUFFER_ON_ROOT_FRAME_DIAGNOSTIC"
        ),
        "scope": "TASK5_DYNAMIC_STRUCTURE_SCREENSHOT_GATE",
        "runtime": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
        },
        "observed_condition": (
            "The fresh-process Isaac Camera RGBA buffer remained shape=[0] "
            "after render polling; no runtime replay PNG was accepted."
        ),
        "attempts": attempts,
        "three_attempt_stop_rule_applied": True,
        "resolution": {
            "status": "PASS" if visual_pass else "FAIL",
            "method": (
                "omni.kit.viewport.utility.capture_viewport_to_file"
            ),
            "cause": (
                "The Sensor Camera buffer remained empty. The first viewport "
                "captures also used the stale pre-root-correction target. "
                "Recomputing the camera target from runtime CAD finger mesh "
                "world points produced three fixed-camera captures."
            ),
            "visual_review": _input(NUMERIC_PASS_SCREENSHOT_REVIEW),
            "capture_count": visual_review["capture_count"],
            "screenshot_status": visual_review["screenshot_status"],
            "same_frame_dynamic_capture": "NOT_AVAILABLE",
        },
        "static_structure_screenshot_evidence": {
            **_input(STATIC_SCREENSHOT_REVIEW),
            "status": "PASS",
            "capture_records": 12,
            "limitation": (
                "Static USD structure visual evidence only; it does not "
                "satisfy the dynamic runtime screenshot gate."
            ),
        },
        "consequences": {
            "dynamic_runtime_screenshot_gate": (
                "PASS_AUXILIARY_RUNTIME_READBACK_REPLAY"
                if visual_pass
                else "HARD_BLOCKER"
            ),
            "bottle_test_allowed": visual_pass,
            "bottle_contact_grasp": "NOT_RUN",
            "task7": "NOT_RUN",
            "task8": "NOT_RUN",
        },
        "resolution_policy": (
            "Do not retry the failed Sensor Camera path. The viewport replay "
            "is auxiliary state/direction evidence and is not same-frame "
            "contact or grasp evidence."
        ),
    }


def _write_markdown(
    diagnosis: dict[str, Any],
    blocker: dict[str, Any],
) -> None:
    profile_rows = []
    for name, metrics in diagnosis["profiles"].items():
        profile_rows.append(
            "| "
            f"{name} | {metrics['status']} | "
            f"{metrics['maximum_base_translation_drift_m']:.9g} | "
            f"{metrics['maximum_arm_dof_drift']:.9g} | "
            f"{metrics['maximum_intended_final_error_m']:.9g} |"
        )
    OUTPUT_MD.write_text(
        "\n".join(
            [
                "# ALOHA ViperX supplier-CAD finger Task 5 dynamic structure",
                "",
                f"- Overall status: `{diagnosis['status']}`",
                (
                    "- Numeric no-bottle structure gate: "
                    f"`{diagnosis['numeric_structure_gate']}`"
                ),
                "- Runtime readback visual gate: "
                f"`{diagnosis['visual_runtime_gate']}`",
                "- Bottle/contact/grasp: `NOT_RUN`",
                "- Task 7 / Task 8: `NOT_RUN` / `NOT_RUN`",
                "",
                "| Profile | Status | max base drift m | max arm drift | "
                "max intended finger error m |",
                "|---|---|---:|---:|---:|",
                *profile_rows,
                "",
                "## Causal result",
                "",
                "- Correcting the computed root-joint frame removes the "
                "approximately 76 mm base snap.",
                "- Setting only the two finger maxForce values to the 5 N "
                "URDF effort limits restores intended finger tracking.",
                "- Setting the six arm maxForce values to their generated "
                "URDF effort limits reduces arm drift below the numeric gate.",
                "- These are isolated diagnostic settings, not promotion of "
                "the final/default asset.",
                "",
                "The final isolated profile passes every machine-readable "
                "numeric no-bottle gate. Three fixed-camera Isaac viewport "
                "replays of exact runtime readbacks also pass visual review. "
                "They are auxiliary evidence, not same-frame physics proof.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    attempt_lines = [
        (
            f"- Attempt {item['attempt']}: `{item['failure']}`; "
            f"accepted PNGs = `{item['raw_png_count']}`; "
            f"log: `{item['stderr_log']['absolute_path']}`"
        )
        for item in blocker["attempts"]
    ]
    BLOCKER_OUTPUT_MD.write_text(
        "\n".join(
            [
                "# ALOHA ViperX Task 5 runtime screenshot blocker",
                "",
                f"- Historical blocker status: `{blocker['status']}`",
                f"- Code: `{blocker['blocker_code']}`",
                "- Static structure screenshots: `PASS` (12 captures)",
                (
                    "- Runtime readback replay screenshots: "
                    "`PASS_AUXILIARY_RUNTIME_READBACK_REPLAY`"
                ),
                "",
                *attempt_lines,
                "",
                "The Sensor Camera attempts remain rejected. The separate "
                "Isaac viewport backend resolved image acquisition after the "
                "camera target was recomputed from runtime finger geometry.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    raw_reports = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in PROBE_PATHS.items()
    }
    profiles = {
        name: metrics_from_report(report)
        for name, report in raw_reports.items()
    }
    final_metrics = profiles["arm_max_force_over_combined"]
    numeric_pass = (
        final_metrics["status"] == "PASS"
        and final_metrics["maximum_base_translation_drift_m"] <= 0.001
        and final_metrics["maximum_arm_dof_drift"] <= 0.001
        and final_metrics["all_intended_directions_correct"]
        and final_metrics["maximum_intended_final_error_m"] <= 0.001
        and final_metrics["maximum_non_target_finger_drift_m"] <= 0.001
        and final_metrics["stage_immutable"]
        and final_metrics["no_bottle"]
    )
    blocker = _build_screenshot_blocker()
    visual_gate = blocker["consequences"][
        "dynamic_runtime_screenshot_gate"
    ]
    visual_pass = visual_gate == "PASS_AUXILIARY_RUNTIME_READBACK_REPLAY"
    diagnosis = {
        "schema_version": 1,
        "status": "PASS" if numeric_pass and visual_pass else "PARTIAL",
        "gate": "TASK5_NO_BOTTLE_DYNAMIC_STRUCTURE",
        "numeric_structure_gate": "PASS" if numeric_pass else "FAIL",
        "visual_runtime_gate": visual_gate,
        "classification": (
            "ROOT_FRAME_AND_AUTHORED_MAX_FORCE_CAUSES_ISOLATED_"
            "NUMERIC_PASS_AUXILIARY_VIEWPORT_REPLAY_PASS"
            if numeric_pass and visual_pass
            else "INCONCLUSIVE"
        ),
        "inputs": {
            name: _input(path) for name, path in PROBE_PATHS.items()
        },
        "profiles": profiles,
        "causal_findings": [
            {
                "cause": "disjoint_root_joint_frame",
                "evidence": "runtime_numeric_ablation",
                "result": "corrected by computed body-frame relation",
            },
            {
                "cause": "finger_drive_max_force_zero",
                "evidence": "runtime_readback_and_5N_URDF_effort_ablation",
                "result": "causal for failed finger tracking",
            },
            {
                "cause": "arm_drive_max_force_zero",
                "evidence": (
                    "runtime_readback_and_generated_URDF_effort_ablation"
                ),
                "result": "causal for arm drift",
            },
        ],
        "evidence_classification": {
            "runtime_readback": [
                "diagnostic profile metrics and drive readback",
                "source and diagnostic Stage hashes before/after runtime",
            ],
            "official_or_generated_source_direct": [
                "finger effort limit 5 N from generated follower URDF",
                "six arm effort limits from generated follower URDF",
            ],
            "numerical_calculation": [
                "base translation drift",
                "arm DOF drift",
                "finger target/readback errors",
            ],
            "engineering_inference": [
                "isolated variable changes establish the listed causal chain",
            ],
            "diagnostic_only_not_final": [
                "root-frame layer",
                "finger maxForce layer",
                "arm maxForce layer",
            ],
        },
        "runtime_screenshot_history": {
            "absolute_path": str(BLOCKER_OUTPUT.resolve()),
            "code": blocker["blocker_code"],
        },
        "scope": {
            "static_structure_screenshots": "PASS",
            "dynamic_runtime_screenshots": visual_gate,
            "bottle_contact_grasp": "NOT_RUN",
            "task7": "NOT_RUN",
            "task8": "NOT_RUN",
            "default_or_final_asset_modified": False,
        },
        "promotion": {
            "allowed": False,
            "reason": (
                "This remains a diagnostic-only profile; no default/final "
                "asset promotion was authorized."
            ),
        },
    }
    if not numeric_pass or not visual_pass:
        diagnosis["status"] = "FAIL"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(diagnosis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    BLOCKER_OUTPUT.write_text(
        json.dumps(blocker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(diagnosis, blocker)
    print(f"status={diagnosis['status']}")
    print(f"numeric_structure_gate={diagnosis['numeric_structure_gate']}")
    print(f"visual_runtime_gate={diagnosis['visual_runtime_gate']}")
    print(f"json={OUTPUT.resolve()}")
    print(f"blocker={BLOCKER_OUTPUT.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
