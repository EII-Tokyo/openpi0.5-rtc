#!/usr/bin/env python3
"""Finalize the Task 7 gripper JointStateAPI PhysicsRules A/B report."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/aloha1_mapping"
ARTIFACT_ROOT = ROOT / ".codex/artifacts/20260731-aloha1-post-grasp-task7"
BUILD_REPORT = REPORT_ROOT / "aloha1_task7_joint_state_physics_candidate_build.json"
BASELINE_REPORT = REPORT_ROOT / "aloha1_signal_correspondence_official_rules.json"
OUTPUT_JSON = REPORT_ROOT / "aloha1_task7_joint_state_physics_candidate.json"
OUTPUT_MD = REPORT_ROOT / "aloha1_task7_joint_state_physics_candidate.md"

RULE_REPORTS = {
    "follower_left": {
        "primary": REPORT_ROOT
        / "aloha1_task7_joint_state_physics_follower_left_official.json",
        "repeat": ARTIFACT_ROOT / "joint_state_left_official_repeat.json",
    },
    "follower_right": {
        "primary": REPORT_ROOT
        / "aloha1_task7_joint_state_physics_follower_right_official.json",
        "repeat": ARTIFACT_ROOT / "joint_state_right_official_repeat.json",
    },
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _issue_key(issue: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(issue.get("rule")),
        str(issue.get("at")),
        str(issue.get("message")),
    )


def _signature(report: dict[str, Any]) -> str:
    keys = (
        "category",
        "official_status",
        "rules",
        "issues",
        "blocking_issue_count",
        "warning_count",
        "target_sha256_before",
        "target_sha256_after",
        "target_immutable",
    )
    payload = {key: report[key] for key in keys}
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _baseline_target(
    report: dict[str, Any], name: str
) -> dict[str, Any]:
    matches = [
        target
        for target in report["targets"]
        if target["category"] == "IsaacSim.PhysicsRules"
        and target["target_name"] == name
    ]
    if len(matches) != 1:
        raise ValueError(f"baseline PhysicsRules target mismatch: {name}")
    return matches[0]


def _candidate_result(
    *,
    name: str,
    build: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    primary_path = RULE_REPORTS[name]["primary"]
    repeat_path = RULE_REPORTS[name]["repeat"]
    primary = _load(primary_path)
    repeat = _load(repeat_path)
    baseline_target = _baseline_target(baseline, name)
    build_candidate = build["candidates"][name]

    primary_signature = _signature(primary)
    repeat_signature = _signature(repeat)
    baseline_keys = {_issue_key(issue) for issue in baseline_target["issues"]}
    candidate_keys = {_issue_key(issue) for issue in primary["issues"]}
    removed = [
        issue
        for issue in baseline_target["issues"]
        if _issue_key(issue) in baseline_keys - candidate_keys
    ]
    added = [
        issue
        for issue in primary["issues"]
        if _issue_key(issue) in candidate_keys - baseline_keys
    ]
    remaining_counts = Counter(issue["rule"] for issue in primary["issues"])
    checks = {
        "baseline_blocking_five": baseline_target["blocking_issue_count"] == 5,
        "candidate_blocking_four": primary["blocking_issue_count"] == 4,
        "candidate_warning_zero": primary["warning_count"] == 0,
        "only_joint_state_removed": [item["rule"] for item in removed]
        == ["JointHasJointStateAPI"],
        "no_issue_added": not added,
        "remaining_rules_exact": dict(sorted(remaining_counts.items()))
        == {"MimicAPICheck": 1, "RigidBodyHasCollider": 3},
        "deterministic_repeat": primary_signature == repeat_signature,
        "candidate_immutable": primary["target_immutable"] is True,
        "source_stage_unchanged": build_candidate["source_stage"]["modified"]
        is False,
        "joint_state_api_readback": build_candidate[
            "joint_state_api_readback"
        ]
        is True,
        "no_authored_state_values": build_candidate[
            "authored_state_values"
        ]
        is False,
        "no_authored_drive_values": build_candidate[
            "authored_drive_values"
        ]
        is False,
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{name} candidate checks failed: {failed}")

    return {
        "status": "PASS",
        "wrapper": build_candidate["wrapper"],
        "physics_layer": build_candidate["physics_layer"],
        "joint_path": build_candidate["joint_path"],
        "joint_type": build_candidate["joint_type"],
        "joint_state_axis": build_candidate["joint_state_axis"],
        "joint_state_api_readback": build_candidate[
            "joint_state_api_readback"
        ],
        "authored_state_values": build_candidate["authored_state_values"],
        "authored_drive_values": build_candidate["authored_drive_values"],
        "source_stage": build_candidate["source_stage"],
        "source_stage_modified": False,
        "baseline": {
            "official_status": baseline_target["official_status"],
            "blocking_issue_count": baseline_target["blocking_issue_count"],
            "issue_rules": [item["rule"] for item in baseline_target["issues"]],
        },
        "official_rules": {
            "official_status": primary["official_status"],
            "blocking_issue_count": primary["blocking_issue_count"],
            "warning_count": primary["warning_count"],
            "deterministic_repeat": primary_signature == repeat_signature,
            "deterministic_signature": primary_signature,
            "primary_report": {
                "absolute_path": str(primary_path.resolve()),
                "sha256": _sha256(primary_path),
            },
            "repeat_report": {
                "absolute_path": str(repeat_path.resolve()),
                "sha256": _sha256(repeat_path),
            },
        },
        "removed_issue_rules": [item["rule"] for item in removed],
        "added_issues": added,
        "remaining_issue_rule_counts": dict(sorted(remaining_counts.items())),
        "checks": checks,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ALOHA1 Task 7 JointStateAPI physics candidate",
        "",
        f"- Candidate status: `{report['status']}`",
        f"- Task 7: `{report['task7']}`",
        f"- Task 8: `{report['task8']}`",
        "",
    ]
    for name, candidate in report["candidates"].items():
        lines.extend(
            [
                f"## {name}",
                "",
                "- Gripper joint: " f"`{candidate['joint_path']}`",
                "- Joint type/axis: "
                f"`{candidate['joint_type']} / {candidate['joint_state_axis']}`",
                "- Baseline blocking findings: `5`",
                "- Candidate blocking findings: `4`",
                "- Removed rule: `JointHasJointStateAPI`",
                "- Remaining: `MimicAPICheck x1`, "
                "`RigidBodyHasCollider x3`",
                "- Fresh-process repeat: `PASS`",
                "",
            ]
        )
    lines.extend(
        [
            "The candidate authors only `PhysicsJointStateAPI:angular` in a "
            "dedicated `_physics.usd` layer. It authors no state or drive "
            "values and does not change geometry, colliders, mimic, drives, "
            "the source Stage, or final/default assets.",
            "",
            "The literal PhysicsRules result remains `FAIL` because the two "
            "version-specific mimic findings and six missing-source-collider "
            "findings remain unsuppressed. Therefore Task 7 remains "
            "`PARTIAL` and Task 8 remains `NOT_RUN`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    build = _load(BUILD_REPORT)
    baseline = _load(BASELINE_REPORT)
    candidates = {
        name: _candidate_result(
            name=name,
            build=build,
            baseline=baseline,
        )
        for name in RULE_REPORTS
    }
    report = {
        "schema_version": 1,
        "status": "PASS",
        "task7": "PARTIAL",
        "task8": "NOT_RUN",
        "candidates": candidates,
        "local_runtime": build["local_runtime"],
        "direct_nvidia_mcp_probe": build["direct_nvidia_mcp_probe"],
        "literal_official_status_after_candidate": "FAIL",
        "remaining_boundaries": {
            "ISAAC_5_1_VALIDATOR_SCHEMA_CONFLICT": 2,
            "HARD_BLOCKER_NO_SOURCE_COLLIDER_GEOMETRY": 6,
        },
        "final_or_default_asset_modified": False,
        "geometry_modified": False,
        "collider_modified": False,
        "drive_modified": False,
        "mimic_modified": False,
        "real_robot_connected": False,
        "remote_192_168_1_103_accessed": False,
        "input_manifest": {
            "build_report": {
                "absolute_path": str(BUILD_REPORT.resolve()),
                "sha256": _sha256(BUILD_REPORT),
            },
            "baseline_report": {
                "absolute_path": str(BASELINE_REPORT.resolve()),
                "sha256": _sha256(BASELINE_REPORT),
            },
        },
    }
    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    OUTPUT_MD.write_text(_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
