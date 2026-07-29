#!/usr/bin/env python3
"""Combine fresh single-target Isaac official-rule reports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from tools.aloha1_mapping.signal_correspondence_rules import combine_official_rule_fragments


def _write_markdown(report: dict, path: Path) -> None:
    rows = [
        (
            "| {category} | {target} | {official} | {issues} |".format(
                category=target["category"],
                target=target["target_name"],
                official=target["official_status"],
                issues=len(target["issues"]),
            )
        )
        for target in report["targets"]
    ]
    path.write_text(
        "\n".join(
            [
                "# ALOHA1 signal correspondence official rules",
                "",
                f"- Official status: `{report['official_status']}`",
                (f"- Task 7A applicable status: `{report['task7a_applicable_status']}`"),
                "- Official outcomes suppressed: `false`",
                "",
                "| Category | Target | Official | Issues |",
                "|---|---|---|---:|",
                *rows,
                "",
                (
                    "Task 7A classification does not replace NVIDIA's "
                    "official result. Packaging and uncalibrated geometry "
                    "remain explicit Task 7B/PARTIAL evidence."
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    fragments = [json.loads(path.resolve(strict=True).read_text(encoding="utf-8")) for path in args.input]
    report = combine_official_rule_fragments(fragments)
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
                "official_status": report["official_status"],
                "task7a_applicable_status": report["task7a_applicable_status"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
