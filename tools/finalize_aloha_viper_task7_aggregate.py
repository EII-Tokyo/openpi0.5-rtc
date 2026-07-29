#!/usr/bin/env python3
"""Combine left and right robot-local Task 7 results without inventing placement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LEFT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task7_validation.json"
)
RIGHT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_follower_right_task7_validation.json"
)
IDENTITY = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_follower_left_right_cad_identity.json"
)
OUTPUT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_task7_aggregate_validation.json"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    left = _load(LEFT)
    right = _load(RIGHT)
    identity = _load(IDENTITY)
    cad_available = identity["classification"] in {
        "VERIFIED_IDENTICAL_ROBOT_PRODUCT_INSTANCES",
        "VERIFIED_SINGLE_REUSABLE_ROBOT_PRODUCT",
    }
    status = (
        "FAIL"
        if "FAIL" in {left["status"], right["status"]}
        else "PARTIAL"
    )
    report = {
        "schema_version": 1,
        "status": status,
        "scope": (
            "TWO_FOLLOWER_TASK7_AGGREGATE_WITH_ROBOT_LOCAL_AND_WORKCELL_"
            "BOUNDARIES"
        ),
        "follower_left": {
            "status": left["status"],
            "scope": left["scope"],
            "report": str(LEFT.resolve()),
            "report_sha256": _sha256(LEFT),
        },
        "follower_right_robot_local": {
            "status": right["status"],
            "scope": right["scope"],
            "cad_available": cad_available,
            "cad_identity": identity["classification"],
            "arm_one_joint": right["robot_local"]["arm_one_joint"],
            "mimic_accuracy": right["robot_local"]["mimic_accuracy"],
            "official_rules": right["official_rules"]["status"],
            "report": str(RIGHT.resolve()),
            "report_sha256": _sha256(RIGHT),
        },
        "dual_arm_workcell_placement": {
            "status": "PARTIAL",
            "verified": False,
            "hard_blocker": (
                "HARD_BLOCKER_FOLLOWER_RIGHT_WORKCELL_INSTALL_TRANSFORM"
            ),
        },
        "corrected_interpretation": (
            "The approved follower_left review Stage omits follower_right, "
            "but the supplier CAD is a verified reusable ViperX robot "
            "product. The right Stage is generated and validated in robot-"
            "local coordinates; only its workcell installation transform is "
            "blocked."
        ),
        "hard_blockers": sorted(
            (
                set(left["hard_blockers"])
                | set(right["hard_blockers"])
            )
            - {
                "HARD_BLOCKER_APPROVED_STAGE_CONTAINS_FOLLOWER_LEFT_ONLY"
            }
        ),
        "task8": "NOT_RUN",
    }
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    OUTPUT.with_suffix(".md").write_text(
        "\n".join(
            [
                "# ALOHA Viper Task 7 aggregate",
                "",
                f"- Overall: `{report['status']}`",
                f"- follower_left: `{report['follower_left']['status']}`",
                (
                    "- follower_right robot-local: "
                    f"`{report['follower_right_robot_local']['status']}`"
                ),
                (
                    "- follower_right arm one-joint / mimic: "
                    f"`{report['follower_right_robot_local']['arm_one_joint']}`"
                    " / "
                    f"`{report['follower_right_robot_local']['mimic_accuracy']}`"
                ),
                "- Dual-arm workcell placement: `PARTIAL` / unverified",
                f"- Task 8: `{report['task8']}`",
                "",
                report["corrected_interpretation"],
                "",
                "## HARD_BLOCKER",
                "",
                *[
                    f"- `{item}`"
                    for item in report["hard_blockers"]
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"status={status}")
    print(f"output={OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
