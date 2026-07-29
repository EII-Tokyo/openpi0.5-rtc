#!/usr/bin/env python3
"""Build evidence-linked Task 7 runtime and promotion acceptance reports."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from tools.aloha1_mapping.task7_acceptance_matrix import classify_asset_promotion_readiness
from tools.aloha1_mapping.task7_acceptance_matrix import classify_runtime_control
from tools.aloha1_mapping.task7_acceptance_matrix import classify_workcell_physics
from tools.aloha1_mapping.task7_acceptance_matrix import combine_task7a_layers
from tools.aloha1_mapping.task7_acceptance_matrix import file_sha256
from tools.aloha1_mapping.task7_acceptance_matrix import verify_file_sha256

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/aloha1_mapping"
ARTIFACT_ROOT = ROOT / ".codex/artifacts/20260729-aloha1-task7a-acceptance-separation"
STAGE = (
    ROOT / "assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/1.0/aloha1_signal_correspondence_workcell.usda"
)
EXPECTED_STAGE_SHA256 = "d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf"
INPUTS = {
    "task7a_summary": (REPORT_ROOT / "aloha1_task7a_7b_validation_summary.json"),
    "swept_collision": (REPORT_ROOT / "aloha1_task7a_swept_collision.json"),
    "official_rules": (REPORT_ROOT / "aloha1_signal_correspondence_official_rules.json"),
    "rule_triage": REPORT_ROOT / "aloha1_task7a_rule_triage.json",
    "helper_link_audit": (REPORT_ROOT / "aloha1_task7a_helper_link_semantics.json"),
}
RUNTIME_JSON = REPORT_ROOT / "aloha1_task7_runtime_acceptance.json"
RUNTIME_MD = REPORT_ROOT / "aloha1_task7_runtime_acceptance.md"
PROMOTION_JSON = REPORT_ROOT / "aloha1_task7_asset_promotion_readiness.json"
PROMOTION_MD = REPORT_ROOT / "aloha1_task7_asset_promotion_readiness.md"
APPLICABILITY_JSON = REPORT_ROOT / "aloha1_task7_official_rule_applicability.json"
APPLICABILITY_MD = REPORT_ROOT / "aloha1_task7_official_rule_applicability.md"
INPUT_MANIFEST = ARTIFACT_ROOT / "input_manifest.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _input_manifest() -> dict[str, Any]:
    stage = STAGE.resolve(strict=True)
    verify_file_sha256(stage, EXPECTED_STAGE_SHA256)
    return {
        "schema_version": 1,
        "stage": {
            "absolute_path": str(stage),
            "sha256": EXPECTED_STAGE_SHA256,
        },
        "reports": {
            name: {
                "absolute_path": str(path.resolve(strict=True)),
                "sha256": file_sha256(path.resolve(strict=True)),
                "size_bytes": path.resolve(strict=True).stat().st_size,
            }
            for name, path in INPUTS.items()
        },
        "real_robot_connected": False,
        "remote_192_168_1_103_accessed": False,
        "task_7b": "NOT_RUN",
        "task_8": "NOT_RUN",
    }


def _assert_linked_stage(
    summary: dict[str, Any],
    swept: dict[str, Any],
    helper: dict[str, Any],
) -> None:
    observed = {
        summary["stage"]["sha256_after"],
        swept["stage"]["sha256_after"],
        helper["stage"]["sha256_after"],
    }
    if observed != {EXPECTED_STAGE_SHA256}:
        raise ValueError(f"input reports do not share frozen Stage hash: {observed}")
    if summary["stage"].get("immutable") is not True:
        raise ValueError("Task 7A summary did not preserve its Stage")
    if swept["stage"].get("immutable") is not True:
        raise ValueError("swept-collision report did not preserve its Stage")
    if helper["stage"].get("stage_modified") is not False:
        raise ValueError("helper-link audit modified its Stage")


def _applicability(triage: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    for item in triage["issues"]:
        applicability = item["task7a_applicability"]
        if applicability in {
            "DIRECT_RUNTIME_CROSS_CHECK",
            "DIRECT_LAYER_BOUNDARY",
        }:
            primary_layer = "ASSET_PROMOTION_READINESS"
            runtime_cross_check = True
        elif applicability in {
            "PACKAGE_NOT_CONTROL_SIGNAL",
            "OUT_OF_SCOPE_GEOMETRY_BOUNDARY",
        }:
            primary_layer = "ASSET_PROMOTION_READINESS"
            runtime_cross_check = False
        elif applicability == "NON_BLOCKING":
            primary_layer = "INFORMATIONAL"
            runtime_cross_check = False
        else:
            primary_layer = "TASK7A_RUNTIME_OR_WORKCELL_UNRESOLVED"
            runtime_cross_check = True
        issues.append(
            {
                "issue_key": item["issue_key"],
                "rule": item["rule"],
                "at": item["at"],
                "official_severity": item["official_severity"],
                "official_result_suppressed": item["official_result_suppressed"],
                "classification": item["classification"],
                "task7a_applicability": applicability,
                "primary_layer": primary_layer,
                "runtime_cross_check": runtime_cross_check,
                "closure": item["closure"],
            }
        )
    layer_counts = Counter(item["primary_layer"] for item in issues)
    return {
        "schema_version": 1,
        "status": (
            "PASS"
            if not any(item["primary_layer"] == "TASK7A_RUNTIME_OR_WORKCELL_UNRESOLVED" for item in issues)
            else "FAIL"
        ),
        "issue_count": len(issues),
        "official_status": triage["official_status"],
        "official_status_suppressed": triage["official_status_suppressed"],
        "unclassified_issue_count": triage["unclassified_issue_count"],
        "layer_counts": dict(sorted(layer_counts.items())),
        "issues": issues,
        "task_7b": "NOT_RUN",
        "task_8": "NOT_RUN",
    }


def _runtime_markdown(report: dict[str, Any]) -> str:
    runtime = report["runtime_control"]
    workcell = report["workcell_physics"]
    return "\n".join(
        [
            "# ALOHA1 Task 7 runtime acceptance",
            "",
            f"- Task 7A aggregate: `{report['task_7a_aggregate']}`",
            f"- Runtime control: `{runtime['status']}`",
            f"- Workcell physics: `{workcell['status']}`",
            f"- Asset promotion: `{report['asset_promotion_status']}`",
            f"- Task 7B: `{report['task_7b']}`",
            f"- Task 8: `{report['task_8']}`",
            "",
            "Runtime control and workcell physics pass independently of "
            "package-only Asset Validation findings. The aggregate remains "
            "PARTIAL because the current package is not promotion-ready.",
            "",
            "The four allowed supplier-CAD finger/table contacts are recorded "
            "as reachable-workcell limits. They do not waive generic "
            "robot/environment, non-adjacent self, or cross-follower contact.",
            "",
        ]
    )


def _promotion_markdown(report: dict[str, Any]) -> str:
    counts = report["classification_counts"]
    lines = [
        "# ALOHA1 Task 7 asset-promotion readiness",
        "",
        f"- Status: `{report['status']}`",
        f"- Ready for promotion: `{report['ready_for_promotion']}`",
        f"- Literal official status: `{report['official_status']}`",
        (f"- Official status suppressed: `{report['official_status_suppressed']}`"),
        "",
        "## Classified official findings",
        "",
    ]
    lines.extend(f"- `{name}`: {count}" for name, count in sorted(counts.items()))
    lines.extend(
        [
            "",
            "The current Stage is suitable for the measured Task 7A runtime "
            "diagnostics but is not a promoted SimReady robot package. "
            "Packaging findings require an isolated package candidate. The "
            "six helper links have no source collider geometry, so no "
            "collider is invented and no rigid-body schema is removed.",
            "",
            "- Task 7B: `NOT_RUN`.",
            "- Task 8: `NOT_RUN`.",
            "",
        ]
    )
    return "\n".join(lines)


def _applicability_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ALOHA1 Task 7 official-rule applicability",
        "",
        f"- Applicability audit: `{report['status']}`",
        f"- Literal official status: `{report['official_status']}`",
        f"- Issues: `{report['issue_count']}`",
        f"- Unclassified: `{report['unclassified_issue_count']}`",
        "",
        "| Acceptance layer | Findings |",
        "|---|---:|",
    ]
    lines.extend(f"| `{name}` | {count} |" for name, count in sorted(report["layer_counts"].items()))
    lines.extend(
        [
            "",
            "Applicability changes neither the NVIDIA severity nor the "
            "literal result; it only prevents package-only findings from "
            "being mislabeled as measured controller failures.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    manifest = _input_manifest()
    summary = _load(INPUTS["task7a_summary"])
    swept = _load(INPUTS["swept_collision"])
    official = _load(INPUTS["official_rules"])
    triage = _load(INPUTS["rule_triage"])
    helper = _load(INPUTS["helper_link_audit"])
    _assert_linked_stage(summary, swept, helper)

    runtime = classify_runtime_control(summary["task_7a"])
    workcell = classify_workcell_physics(swept)
    promotion = classify_asset_promotion_readiness(
        official=official,
        triage=triage,
        helper_audit=helper,
    )
    aggregate = combine_task7a_layers(
        runtime_status=runtime["status"],
        workcell_status=workcell["status"],
        promotion_status=promotion["status"],
    )
    runtime_report = {
        "schema_version": 1,
        "task_7a_aggregate": aggregate,
        "runtime_control": runtime,
        "workcell_physics": workcell,
        "asset_promotion_status": promotion["status"],
        "stage": manifest["stage"],
        "input_manifest": str(INPUT_MANIFEST.resolve()),
        "task_7b": "NOT_RUN",
        "task_8": "NOT_RUN",
        "real_robot_connected": False,
        "remote_192_168_1_103_accessed": False,
    }
    promotion.update(
        {
            "schema_version": 1,
            "stage": manifest["stage"],
            "input_manifest": str(INPUT_MANIFEST.resolve()),
            "task_7b": "NOT_RUN",
            "task_8": "NOT_RUN",
            "real_robot_connected": False,
            "remote_192_168_1_103_accessed": False,
            "promotion_candidate": ("NOT_CREATED_EVIDENCE_INSUFFICIENT_FOR_HELPER_LINK_MUTATION"),
        }
    )
    applicability = _applicability(triage)

    _write_json(INPUT_MANIFEST, manifest)
    _write_json(RUNTIME_JSON, runtime_report)
    RUNTIME_MD.write_text(
        _runtime_markdown(runtime_report),
        encoding="utf-8",
    )
    _write_json(PROMOTION_JSON, promotion)
    PROMOTION_MD.write_text(
        _promotion_markdown(promotion),
        encoding="utf-8",
    )
    _write_json(APPLICABILITY_JSON, applicability)
    APPLICABILITY_MD.write_text(
        _applicability_markdown(applicability),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "task_7a_aggregate": aggregate,
                "runtime_control": runtime["status"],
                "workcell_physics": workcell["status"],
                "asset_promotion": promotion["status"],
                "official_status": promotion["official_status"],
                "task_7b": "NOT_RUN",
                "task_8": "NOT_RUN",
            },
            sort_keys=True,
        )
    )
    return 0 if aggregate in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
