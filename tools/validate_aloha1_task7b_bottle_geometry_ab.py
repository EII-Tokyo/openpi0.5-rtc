#!/usr/bin/env python3
"""Combine isolated Task 7B profile reports without launching Isaac."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from tools.aloha1_mapping.task7b_bottle_geometry_ab import compare_geometry_groups
from tools.aloha1_mapping.task7b_bottle_geometry_ab import render_comparison_markdown
from tools.aloha1_mapping.task7b_bottle_geometry_ab import validate_single_geometry_variable

REQUIRED_PHASES = {
    "open",
    "bilateral_contact",
    "release",
    "hold_end",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _validate_input_report(
    report: dict[str, Any],
    trials: list[dict[str, Any]],
    *,
    expected_profile: str,
) -> list[str]:
    failures = []
    if report.get("run_mode") != "ACCEPTANCE":
        failures.append("run_mode")
    if report.get("bottle_profile") != expected_profile:
        failures.append("bottle_profile")
    if len(trials) != 20:
        failures.append("trial_count")
    if not report.get("baseline_protection", {}).get(
        "protected_assets_immutable"
    ):
        failures.append("protected_assets_immutable")
    captures = report.get("screenshots", {}).get("captures", [])
    capture_names = {
        str(capture.get("capture_name")) for capture in captures
    }
    if (
        report.get("screenshots", {}).get("status") != "PASS"
        or capture_names != REQUIRED_PHASES
    ):
        failures.append("screenshots")
    if report.get("boundaries", {}).get("task8") != "NOT_RUN":
        failures.append("task8")
    return failures


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--baseline-trials", type=Path, required=True)
    parser.add_argument("--project-report", type=Path, required=True)
    parser.add_argument("--project-trials", type=Path, required=True)
    parser.add_argument("--screenshot-review", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--output-trials", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = {
        name: value.resolve(strict=True)
        for name, value in (
            ("config", args.config),
            ("baseline_report", args.baseline_report),
            ("baseline_trials", args.baseline_trials),
            ("project_report", args.project_report),
            ("project_trials", args.project_trials),
            ("screenshot_review", args.screenshot_review),
        )
    }
    config = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    baseline_report = _load_json(paths["baseline_report"])
    project_report = _load_json(paths["project_report"])
    screenshot_review = _load_json(paths["screenshot_review"])
    baseline_trials = _load_jsonl(paths["baseline_trials"])
    project_trials = _load_jsonl(paths["project_trials"])

    input_failures = {
        "procedural_cylinder": _validate_input_report(
            baseline_report,
            baseline_trials,
            expected_profile="procedural_cylinder",
        ),
        "project_bottle500": _validate_input_report(
            project_report,
            project_trials,
            expected_profile="project_bottle500",
        ),
    }
    configured_audit = validate_single_geometry_variable(
        config["profiles"]["procedural_cylinder"],
        config["profiles"]["project_bottle500"],
        allowed_differences=config["allowed_profile_differences"],
    )
    runtime_audit = validate_single_geometry_variable(
        baseline_report["causal_profile"],
        project_report["causal_profile"],
        allowed_differences=config["allowed_profile_differences"],
    )
    runtime_profiles_match_config = (
        baseline_report["causal_profile"]
        == config["profiles"]["procedural_cylinder"]
        and project_report["causal_profile"]
        == config["profiles"]["project_bottle500"]
    )
    comparison = compare_geometry_groups(
        baseline_report["summary"],
        project_report["summary"],
    )
    screenshot_review_pass = (
        screenshot_review.get("status") == "PASS"
        and screenshot_review.get("reviewed_raw_image_count") == 8
        and screenshot_review.get("reviewed_annotated_image_count") == 8
    )
    contract_pass = (
        configured_audit["status"] == "PASS"
        and runtime_audit["status"] == "PASS"
        and runtime_profiles_match_config
        and not any(input_failures.values())
        and screenshot_review_pass
    )
    report = {
        "schema_version": 1,
        **comparison,
        "status": (
            comparison["status"] if contract_pass else "FAIL"
        ),
        "single_variable_audit": {
            "status": "PASS" if contract_pass else "FAIL",
            "configured_profiles": configured_audit,
            "runtime_profiles": runtime_audit,
            "runtime_profiles_match_config": runtime_profiles_match_config,
            "input_failures": input_failures,
        },
        "inputs": {
            name: {
                "absolute_path": str(path),
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
        "screenshot_review": {
            "status": (
                "PASS" if screenshot_review_pass else "FAIL"
            ),
            "absolute_path": str(paths["screenshot_review"]),
            "sha256": _sha256(paths["screenshot_review"]),
            "reviewed_raw_image_count": screenshot_review.get(
                "reviewed_raw_image_count"
            ),
            "reviewed_annotated_image_count": screenshot_review.get(
                "reviewed_annotated_image_count"
            ),
            "visual_model_review": screenshot_review.get(
                "visual_model_review"
            ),
        },
        "asset_promotion": "PARTIAL",
        "boundaries": {
            "static_hold_is_pickup": False,
            "support_to_lift": "NOT_RUN",
            "task8": "NOT_RUN",
        },
    }
    combined_trials = []
    for profile_name, trials in (
        ("procedural_cylinder", baseline_trials),
        ("project_bottle500", project_trials),
    ):
        combined_trials.extend(
            {"bottle_profile": profile_name, **trial}
            for trial in trials
        )

    _write_text(
        args.output_json.resolve(),
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    _write_text(
        args.output_markdown.resolve(),
        render_comparison_markdown(report),
    )
    _write_text(
        args.output_trials.resolve(),
        "".join(
            json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
            for item in combined_trials
        ),
    )
    print(f"status={report['status']}")
    print(f"conclusion={report['conclusion']}")
    print(f"report={args.output_json.resolve()}")
    return 0 if contract_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
