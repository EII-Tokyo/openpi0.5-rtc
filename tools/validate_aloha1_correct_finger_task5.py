#!/usr/bin/env python3
"""Rerun frozen Task 5 against the user-confirmed custom ALOHA fingers."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import traceback
from typing import Any

import numpy as np

from tools.aloha1_mapping.correct_finger_asset import (
    EXPECTED_RESTART_BOUNDARY,
)
from tools.aloha1_mapping.correct_finger_asset import load_correct_finger_profile
from tools.aloha1_mapping.correct_finger_asset import sha256_file
from tools.aloha1_mapping.correct_finger_asset import verify_correct_finger_sources
from tools.aloha1_mapping.gripper_collider_ab import (
    classify_decomposition_status,
)
from tools.aloha1_mapping.gripper_collider_ab import summarize_ab_trials
from tools.aloha1_mapping.gripper_validation import build_gripper_validation_plan
from tools.aloha1_mapping.screenshot_manifest import build_screenshot_manifest
import tools.validate_aloha1_gripper_collider_ab as frozen_runtime


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(
                json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                    default=_json_default,
                )
                + "\n"
            )
    temporary.replace(path)


def _gate(status: bool, **evidence: Any) -> dict[str, Any]:
    return {"status": "PASS" if status else "FAIL", **evidence}


def _required_group_screenshots(
    robot: str,
    profile_name: str,
) -> dict[str, list[str]]:
    token = "hull" if profile_name == "convex_hull" else "decomposition"
    return {
        "runtime_open": [
            f"{robot}_{token}_open_with_bottle_isometric",
        ],
        "bilateral_contact": [
            f"{robot}_{token}_bilateral_contact_established_closing_axis",
            f"{robot}_{token}_bilateral_contact_established_isometric",
        ],
        "release_hold": [
            f"{robot}_{token}_release_isometric",
            f"{robot}_{token}_hold_end_isometric",
        ],
    }


def _task5_gates(
    trials: Sequence[Mapping[str, Any]],
    *,
    screenshot_status: str,
) -> dict[str, Any]:
    motion = [
        (
            float(trial["states"]["open_fingers"]["left_finger_m"])
            > float(trial["states"]["start_fingers"]["left_finger_m"])
            and float(trial["states"]["open_fingers"]["right_finger_m"])
            < float(trial["states"]["start_fingers"]["right_finger_m"])
            and float(
                trial["states"]["closed_against_fixed_bottle"]["left_finger_m"]
            )
            < float(trial["states"]["open_fingers"]["left_finger_m"])
            and float(
                trial["states"]["closed_against_fixed_bottle"]["right_finger_m"]
            )
            > float(trial["states"]["open_fingers"]["right_finger_m"])
        )
        for trial in trials
    ]
    aperture = [
        float(trial["aperture"]["open"]["surface_gap_m"])
        > float(
            trial["aperture"]["closed_against_fixed_bottle"]["surface_gap_m"]
        )
        for trial in trials
    ]
    residuals = [
        float(trial["states"]["maximum_sampled_control_residual_m"])
        for trial in trials
    ]
    bilateral = [
        bool(trial["metrics"]["bilateral_contact_before_release"])
        for trial in trials
    ]
    persistence = [
        bool(trial["contacts"]["left"].get("contact"))
        and bool(trial["contacts"]["right"].get("contact"))
        and float(trial["contacts"]["left"].get("contact_duration_s", 0.0))
        > 0.0
        and float(trial["contacts"]["right"].get("contact_duration_s", 0.0))
        > 0.0
        for trial in trials
    ]
    no_penetration = [
        not bool(trial["metrics"]["persistent_penetration"])
        for trial in trials
    ]
    no_internal = [
        not bool(trial["metrics"]["unexpected_gripper_collision"])
        for trial in trials
    ]
    hold = [
        bool(trial["metrics"]["held_for_required_steps"])
        for trial in trials
    ]
    signature_groups: dict[tuple[str, str], set[str]] = {}
    for trial in trials:
        key = (str(trial["profile"]), str(trial["robot"]))
        signature_groups.setdefault(key, set()).add(
            str(trial["deterministic_signature"])
        )
    deterministic = all(
        len(signatures) == 1 for signatures in signature_groups.values()
    )
    return {
        "finger_motion_direction": _gate(
            bool(motion) and all(motion),
            passed_trials=sum(motion),
            trial_count=len(motion),
        ),
        "aperture_monotonicity": _gate(
            bool(aperture) and all(aperture),
            passed_trials=sum(aperture),
            trial_count=len(aperture),
        ),
        "mimic_accuracy": _gate(
            bool(residuals) and max(residuals) <= 0.001,
            tolerance_m=0.001,
            maximum_sampled_residual_m=max(residuals) if residuals else None,
        ),
        "bilateral_contact_establishment": _gate(
            bool(bilateral) and all(bilateral),
            passed_trials=sum(bilateral),
            trial_count=len(bilateral),
        ),
        "contact_persistence": _gate(
            bool(persistence) and all(persistence),
            passed_trials=sum(persistence),
            trial_count=len(persistence),
            semantics="nonzero reported duration; not equivalent to static hold",
        ),
        "penetration": _gate(
            bool(no_penetration) and all(no_penetration),
            passed_trials=sum(no_penetration),
            trial_count=len(no_penetration),
        ),
        "unexpected_internal_collision": _gate(
            bool(no_internal) and all(no_internal),
            passed_trials=sum(no_internal),
            trial_count=len(no_internal),
        ),
        "static_bottle_hold": _gate(
            bool(hold) and all(hold),
            passed_trials=sum(hold),
            trial_count=len(hold),
            hold_interval_s=2.0,
            maximum_drop_m=0.010,
        ),
        "determinism": _gate(
            deterministic,
            unique_signature_count_by_profile_robot={
                f"{key[0]}/{key[1]}": len(values)
                for key, values in sorted(signature_groups.items())
            },
        ),
        "screenshots": _gate(
            screenshot_status == "PASS",
            screenshot_manifest_status=screenshot_status,
        ),
    }


def run(
    *,
    project_root: Path,
    repeats: int,
    smoke: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    profile_path = (
        project_root / "configs/aloha1_gripper_correct_finger_profiles.yaml"
    )
    profile = load_correct_finger_profile(profile_path, project_root)
    repeats_per_robot = 20
    if repeats < repeats_per_robot and not smoke:
        raise ValueError(
            "acceptance Task 5 rerun requires at least 20 fresh resets per robot"
        )
    source_before = verify_correct_finger_sources(profile, project_root)
    if source_before["status"] != "PASS":
        raise RuntimeError("correct-finger source/protected baseline preflight failed")
    base_plan = build_gripper_validation_plan(project_root)
    frozen = profile["frozen"]
    if (
        base_plan["physics"]["physics_dt_s"]
        != 1.0 / float(frozen["physics_frequency_hz"])
        or base_plan["bottle_proxy"]["mass_kg"] != frozen["bottle_mass_kg"]
        or base_plan["bottle_proxy"]["diameter_m"]
        != frozen["bottle_diameter_m"]
        or base_plan["released_hold"]["hold_time_s"]
        != frozen["hold_interval_s"]
        or base_plan["released_hold"]["max_drop_m"] != frozen["drop_gate_m"]
    ):
        raise RuntimeError("runtime plan no longer matches correct-finger frozen values")
    robots = {item["name"]: item for item in base_plan["robots"]}
    output_dir = (
        project_root
        / "reports/aloha1_mapping/"
        / (
            "gripper_correct_finger_task5_smoke_trials"
            if smoke
            else "gripper_correct_finger_task5_trials"
        )
    )
    screenshot_root = (
        project_root / profile["diagnostic_directories"]["screenshots"]
    ).resolve()
    if smoke:
        screenshot_root = screenshot_root / "smoke"
    screenshot_root.mkdir(parents=True, exist_ok=True)
    captures: list[dict[str, Any]] = []
    groups: dict[str, Any] = {}
    all_trials: list[dict[str, Any]] = []
    group_trials: dict[str, list[dict[str, Any]]] = {}
    for profile_name, variant in profile["profiles"].items():
        approximation = variant["approximation"]
        group_name = (
            "hull_current"
            if profile_name == "convex_hull"
            else "decomposition_current"
        )
        robot_results = {}
        combined = []
        for robot in ("follower_left", "follower_right"):
            asset = (
                project_root
                / profile["diagnostic_directories"][profile_name]
                / robot
                / f"{robot}_{profile_name}.usd"
            ).resolve(strict=True)
            trials = []
            for trial_index in range(repeats):
                print(
                    json.dumps(
                        {
                            "correct_finger_task5_event": "trial_start",
                            "group": group_name,
                            "robot": robot,
                            "trial_index": trial_index,
                            "repeat_count": repeats,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                screenshot_context = (
                    {
                        "artifact_root": screenshot_root,
                        "captures": captures,
                        "asset": str(asset),
                        "robot": robot,
                        "profile": profile_name,
                        "trial_index": trial_index,
                    }
                    if trial_index == 0
                    else None
                )
                trial = frozen_runtime._run_trial(
                    robot_plan=robots[robot],
                    base_plan=base_plan,
                    asset=asset,
                    profile_name=profile_name,
                    approximation=approximation,
                    control_mode="current_mimic",
                    trial_index=trial_index,
                    screenshot_context=screenshot_context,
                )
                trials.append(trial)
                print(
                    json.dumps(
                        {
                            "correct_finger_task5_event": "trial_complete",
                            "group": group_name,
                            "robot": robot,
                            "trial_index": trial_index,
                            "status": trial["status"],
                            "drop_m": trial["released_hold"]["drop_m"],
                            "bilateral": trial["metrics"][
                                "bilateral_contact_before_release"
                            ],
                            "signature": trial["deterministic_signature"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            trial_path = output_dir / group_name / f"{robot}.jsonl"
            _write_jsonl(trial_path, trials)
            minimum = 1 if smoke else repeats_per_robot
            summary = summarize_ab_trials(
                trials,
                minimum_repeats=minimum,
            )
            group_capture_names = {
                name
                for names in _required_group_screenshots(
                    robot,
                    profile_name,
                ).values()
                for name in names
            }
            group_captures = [
                item
                for item in captures
                if item["capture_name"] in group_capture_names
            ]
            group_screenshots = build_screenshot_manifest(
                captures=group_captures,
                required_captures=_required_group_screenshots(
                    robot,
                    profile_name,
                ),
                artifact_root=screenshot_root,
            )
            robot_results[robot] = {
                "summary": summary,
                "trial_file": str(trial_path.resolve()),
                "trial_file_sha256": sha256_file(trial_path),
                "asset": str(asset),
                "asset_sha256": sha256_file(asset),
                "screenshots": group_screenshots,
            }
            combined.extend(trials)
        group_trials[group_name] = combined
        all_trials.extend(combined)
        groups[group_name] = {
            "profile": profile_name,
            "approximation": approximation,
            "control_mode": "current_mimic",
            "robots": robot_results,
            "combined": summarize_ab_trials(
                combined,
                minimum_repeats=(
                    2 if smoke else repeats_per_robot * 2
                ),
            ),
            "diagnostic_metrics": frozen_runtime._diagnostic_group_metrics(
                combined
            ),
            "screenshots": _gate(
                all(
                    item["screenshots"]["status"] == "PASS"
                    for item in robot_results.values()
                ),
                robots={
                    name: item["screenshots"]["status"]
                    for name, item in robot_results.items()
                },
            ),
        }

    preflight_screenshots = json.loads(
        (
            project_root
            / "reports/aloha1_mapping/"
            "gripper_correct_finger_preflight_screenshots.json"
        ).read_text(encoding="utf-8")
    )["manifest"]["captures"]
    aggregate_captures = preflight_screenshots + captures
    required = {
        phase: profile["screenshots"]["required_captures"][phase]
        for phase in (
            "asset_preflight",
            "runtime_open",
            "bilateral_contact",
            "release_hold",
        )
    }
    screenshot_manifest = build_screenshot_manifest(
        captures=aggregate_captures,
        required_captures=required,
        artifact_root=(
            Path(preflight_screenshots[0]["artifact_root"])
            if not smoke
            else screenshot_root
        ),
    )
    if smoke:
        # Smoke images use an isolated root and are intentionally not combined
        # with the acceptance preflight manifest.
        screenshot_manifest = build_screenshot_manifest(
            captures=captures,
            required_captures={
                phase: required[phase]
                for phase in (
                    "runtime_open",
                    "bilateral_contact",
                    "release_hold",
                )
            },
            artifact_root=screenshot_root,
        )

    gates = _task5_gates(
        all_trials,
        screenshot_status=screenshot_manifest["status"],
    )
    experiment_complete = all(
        group["combined"]["complete"] for group in groups.values()
    )
    physical_pass = all(gate["status"] == "PASS" for gate in gates.values())
    decomposition = classify_decomposition_status(
        [
            bool(trial["metrics"]["held_for_required_steps"])
            for trial in group_trials["hull_current"]
        ],
        [
            bool(trial["metrics"]["held_for_required_steps"])
            for trial in group_trials["decomposition_current"]
        ],
        minimum_repeats=(2 if smoke else repeats_per_robot * 2),
    )
    source_after = verify_correct_finger_sources(profile, project_root)
    report = {
        "schema_version": 1,
        "status": (
            "PARTIAL"
            if smoke
            else ("PASS" if physical_pass else "FAIL")
        ),
        "experiment_execution_status": (
            "PASS" if experiment_complete else "FAIL"
        ),
        "run_mode": "NON_ACCEPTANCE_SMOKE" if smoke else "ACCEPTANCE",
        "restart_boundary": EXPECTED_RESTART_BOUNDARY,
        "scope": "ALOHA follower correct-custom-finger Task 5 only",
        "repeats_per_robot": repeats,
        "fresh_reset_per_trial": True,
        "frozen_manifest": str(profile_path),
        "frozen_values": frozen,
        "groups": groups,
        "task5_gates": gates,
        "CONVEX_DECOMPOSITION_STATUS": decomposition["status"],
        "decomposition_evidence": decomposition,
        "source_protection": {
            "before": source_before,
            "after": source_after,
        },
        "screenshot_manifest": str(
            project_root
            / "reports/aloha1_mapping/"
            "gripper_correct_finger_screenshot_manifest.json"
        ),
        "screenshot_root_absolute": str(screenshot_root),
        "default_asset_collider_modified": False,
        "historical_generic_finger_reports_modified": False,
        "task8": "NOT_RUN",
    }
    return report, screenshot_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the correct-finger frozen Task 5 A/B."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="allow fewer than 20 repeats and keep outputs non-acceptance",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.project_root.resolve(strict=True)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "width": 1280, "height": 900})
    report_path = (
        root
        / "reports/aloha1_mapping/"
        / (
            "gripper_correct_finger_task5_smoke.json"
            if args.smoke
            else "gripper_correct_finger_task5.json"
        )
    )
    manifest_path = (
        root
        / "reports/aloha1_mapping/"
        / (
            "gripper_correct_finger_screenshot_manifest_smoke.json"
            if args.smoke
            else "gripper_correct_finger_screenshot_manifest.json"
        )
    )
    failure_path = (
        root
        / "reports/aloha1_mapping/gripper_correct_finger_task5_failure.json"
    )
    try:
        report, screenshot_manifest = run(
            project_root=root,
            repeats=args.repeats,
            smoke=args.smoke,
        )
        _write_json(report_path, report)
        _write_json(manifest_path, screenshot_manifest)
        failure_path.unlink(missing_ok=True)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "experiment_execution_status": report[
                        "experiment_execution_status"
                    ],
                    "report": str(report_path),
                    "screenshot_manifest": str(manifest_path),
                    "screenshot_root": report["screenshot_root_absolute"],
                },
                indent=2,
            ),
            flush=True,
        )
    except BaseException as error:
        _write_json(
            failure_path,
            {
                "schema_version": 1,
                "status": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        traceback.print_exc()
        raise
    finally:
        app.close()
    return 0 if report["experiment_execution_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
