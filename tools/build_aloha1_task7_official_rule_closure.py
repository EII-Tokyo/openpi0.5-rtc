#!/usr/bin/env python3
"""Build the machine-readable ALOHA1 Task 7 official-rule closure plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tools.aloha1_mapping.task7_official_rule_closure import classify_official_rule_closure

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/aloha1_mapping"
TRIAGE = REPORT_ROOT / "aloha1_task7a_rule_triage.json"
OFFICIAL = REPORT_ROOT / "aloha1_signal_correspondence_official_rules.json"
STAGE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/1.0"
    / "aloha1_signal_correspondence_workcell.usda"
)
OUTPUT_JSON = REPORT_ROOT / "aloha1_task7_official_rule_closure.json"
OUTPUT_MD = REPORT_ROOT / "aloha1_task7_official_rule_closure.md"
RIGHT_SCHEMA_ASSET = (
    REPORT_ROOT
    / "aloha_viper_supplier_cad_follower_right_robot_schema_asset.json"
)
RIGHT_SCHEMA_RULES = (
    REPORT_ROOT / "aloha1_task7_right_schema_official_robot_rules.json"
)
RIGHT_SCHEMA_RULES_REPEAT = (
    ROOT
    / ".codex/artifacts/20260731-aloha1-post-grasp-task7/"
    "follower_right_schema_robot_rules_repeat.json"
)

FROZEN = {
    TRIAGE: "59c45343616cda2017344b1b0170697ee6b16c615fe4d7cfaa09b91bf373d53c",
    OFFICIAL: "a7acb1e7363d1306b01b7f9609f9a5250f0b535771a3de8523246ae3cd31756f",
    STAGE: "d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf",
    RIGHT_SCHEMA_ASSET: (
        "1bb8b2495a9872538d40d028d70536e73b6fd1e75b89abdd7c75c7074ee68ab0"
    ),
    RIGHT_SCHEMA_RULES: (
        "7b5688acbac079b0a6514b1fe31890ff780c5bf1b9425f307804be8f38b0d34c"
    ),
    RIGHT_SCHEMA_RULES_REPEAT: (
        "b7898402e1044e6e04c5de6d778deadbf037511a187629856bd00b3a4a98e2f0"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def _rule_signature(report: dict[str, Any]) -> str:
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


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ALOHA1 Task 7 official-rule closure audit",
        "",
        f"- Status: `{report['status']}`",
        f"- Literal official status: `{report['official_status']}`",
        f"- Findings: `{report['issue_count']}`",
        f"- Stage mutated: `{report['stage_mutated']}`",
        f"- Task 8: `{report['task8']}`",
        "",
        "## Evidence partition",
        "",
    ]
    lines.extend(
        f"- `{name}`: `{count}`"
        for name, count in report["classification_counts"].items()
    )
    lines.extend(["", "## Authorized next action", ""])
    lines.extend(
        f"- `{name}`: `{count}`"
        for name, count in report["action_counts"].items()
    )
    lines.extend(
        [
            "",
            "Only the 28 package/layer findings may be tested in a new "
            "isolated promotion candidate. The source Stage, robot geometry, "
            "joints, drives, mimic, collisions and final/default assets remain "
            "unchanged.",
            "",
            "The six helper-link findings remain "
            "`HARD_BLOCKER_NO_SOURCE_GEOMETRY`; no collider is invented and "
            "RigidBodyAPI is not removed. The two mimic findings remain visible "
            "as literal Isaac Sim 5.1 errors even though the opposed-axis "
            "runtime probe passed.",
            "",
            "The direct NVIDIA MCP probe was reachable, but its Asset "
            "Validation catalog reported 1.2.1. Exact rule behavior therefore "
            "uses the installed Isaac Sim 5.1 Asset Validation 1.1.0 source as "
            "the version authority.",
            "",
            "## Isolated candidate result",
            "",
            "The follower-right schema-only candidate passed "
            "`IsaacSim.RobotRules` twice in fresh processes with zero issues. "
            "Its physical diagnostic Stage was not composed or modified. This "
            "closes the right-side Robot Schema/package boundary only; it does "
            "not clear the six missing-source-collider findings or the two "
            "literal mimic-rule conflicts.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    manifests = []
    for path, expected in FROZEN.items():
        actual = _sha256(path.resolve(strict=True))
        if actual != expected:
            raise ValueError(f"frozen input changed: {path}")
        manifests.append(
            {
                "absolute_path": str(path.resolve()),
                "sha256": actual,
            }
        )

    triage = _load(TRIAGE)
    right_schema_asset = _load(RIGHT_SCHEMA_ASSET)
    right_rules = _load(RIGHT_SCHEMA_RULES)
    right_rules_repeat = _load(RIGHT_SCHEMA_RULES_REPEAT)
    first_signature = _rule_signature(right_rules)
    second_signature = _rule_signature(right_rules_repeat)
    right_candidate_checks = {
        "asset_status": right_schema_asset["status"] == "PASS",
        "scope": right_schema_asset["scope"]
        == "ROBOTRULES_SCHEMA_ONLY_DIAGNOSTIC",
        "official_status": right_rules["official_status"] == "PASS",
        "blocking_issue_count": right_rules["blocking_issue_count"] == 0,
        "warning_count": right_rules["warning_count"] == 0,
        "target_immutable": right_rules["target_immutable"] is True,
        "repeat": first_signature == second_signature,
        "physical_stage_excluded": (
            right_schema_asset["physical_right_stage_included"] is False
        ),
        "physical_stage_unchanged": right_schema_asset[
            "physical_right_stage"
        ]["modified"]
        is False,
    }
    failed_candidate = [
        name for name, passed in right_candidate_checks.items() if not passed
    ]
    if failed_candidate:
        raise ValueError(
            f"right schema candidate checks failed: {failed_candidate}"
        )
    report = classify_official_rule_closure(triage["issues"])
    report.update(
        {
            "schema_version": 1,
            "status": "PARTIAL",
            "local_runtime": {
                "isaac_sim": "5.1.0.0",
                "kit": "107.3.3",
                "physx": "107.3.26",
                "asset_validation": "1.1.0",
            },
            "direct_nvidia_mcp_probe": {
                "status": "PASS",
                "transport": "DIRECT_NOT_MCPJUNGLE",
                "tool": "get_isaac_sim_extension_details",
                "asset_validation": "1.2.1",
                "robot_schema": "5.1.0",
                "version_match_for_asset_validation": False,
            },
            "version_authority": "LOCAL_ISAAC_SIM_5_1_SOURCE",
            "input_manifest": manifests,
            "isolated_candidate_results": {
                "follower_right_robot_schema": {
                    "status": "PASS",
                    "official_status": right_rules["official_status"],
                    "blocking_issue_count": right_rules[
                        "blocking_issue_count"
                    ],
                    "warning_count": right_rules["warning_count"],
                    "deterministic_repeat": first_signature
                    == second_signature,
                    "deterministic_signature": first_signature,
                    "candidate_stage": right_schema_asset["wrapper"],
                    "physical_stage": right_schema_asset[
                        "physical_right_stage"
                    ],
                    "physical_stage_modified": False,
                    "checks": right_candidate_checks,
                    "scope": "ROBOTRULES_SCHEMA_ONLY_DIAGNOSTIC",
                }
            },
            "stage_mutated": False,
            "final_or_default_asset_modified": False,
            "real_robot_connected": False,
            "remote_192_168_1_103_accessed": False,
        }
    )
    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    OUTPUT_MD.write_text(_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
