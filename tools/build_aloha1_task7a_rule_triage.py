#!/usr/bin/env python3
"""Build the ALOHA1 Task 7A official-rule triage reports."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from tools.aloha1_mapping.task7a_rule_triage import build_rule_triage

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha1_signal_correspondence_official_rules.json"
)
DEFAULT_OUTPUT = (
    ROOT / "reports/aloha1_mapping/aloha1_task7a_rule_triage.json"
)
DEFAULT_MIMIC_PROBE = (
    ROOT
    / ".codex/artifacts/20260729-aloha1-task7a-rules-sweep/"
    "mimic_rule_runtime_probe.json"
)


def _write_markdown(report: dict, path: Path) -> None:
    counts = Counter(item["classification"] for item in report["issues"])
    rows = [
        f"| {name} | {counts[name]} |"
        for name in sorted(counts)
    ]
    blockers = [
        item
        for item in report["issues"]
        if item["official_severity"] in {"ERROR", "FAILURE"}
    ]
    path.write_text(
        "\n".join(
            [
                "# ALOHA1 Task 7A official-rule triage",
                "",
                f"- Triage status: `{report['status']}`",
                f"- Literal NVIDIA status: `{report['official_status']}`",
                "- Official findings suppressed: `false`",
                (
                    "- Runtime: Isaac Sim `5.1.0.0`, Kit `107.3.3`, "
                    "PhysX `107.3.26`"
                ),
                f"- Source findings: `{report['source_issue_count']}`",
                f"- Blocking findings: `{len(blockers)}`",
                f"- Inconclusive findings: `{report['unclassified_issue_count']}`",
                "",
                "| Classification | Count |",
                "|---|---:|",
                *rows,
                "",
                "## Boundary",
                "",
                (
                    "The gripper JointStateAPI exists twice in the frozen "
                    "workcell home layer; the official finding is produced "
                    "when the child robot asset is validated without that "
                    "workcell layer."
                ),
                "",
                (
                    "The read-only mimic probe loaded the installed 5.1 "
                    "MimicAPICheck and confirmed positive limits on the "
                    "active finger, negative limits on the opposite local "
                    "finger axis, and positive gearing. The installed rule "
                    "compares those raw local-axis intervals and its "
                    "positive-gearing error message labels the self upper "
                    "limit as a lower limit. This is recorded as a "
                    "version-specific validator/schema conflict; the two "
                    "literal errors are not suppressed."
                ),
                "",
                (
                    "Mass-only helper links remain missing-source-evidence "
                    "blockers. No collider, density, mass, or inertia was "
                    "invented. The literal NVIDIA failures remain visible."
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mimic-probe",
        type=Path,
        default=DEFAULT_MIMIC_PROBE,
    )
    args = parser.parse_args()
    report = build_rule_triage(
        ROOT,
        args.input,
        mimic_probe_path=args.mimic_probe,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(report, output.with_suffix(".md"))
    print(
        json.dumps(
            {
                "status": report["status"],
                "official_status": report["official_status"],
                "triaged_issue_count": report["triaged_issue_count"],
                "unclassified_issue_count": report[
                    "unclassified_issue_count"
                ],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0 if report["unclassified_issue_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
