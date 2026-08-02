#!/usr/bin/env python3
"""Aggregate the isolated Task 7 PhysicsRules candidate matrix."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "reports/aloha1_mapping/task7_physicsrules_root_cause_matrix"
OUTPUT = ROOT / "reports/aloha1_mapping/aloha1_task7_physicsrules_root_cause_matrix.json"
OUTPUT_MD = OUTPUT.with_suffix(".md")
FROZEN_SHA256 = "327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9"
HELPER_MASS_AUDIT = (
    ROOT / "reports/aloha1_mapping/aloha1_task7_virtual_helper_mass_audit.json"
)
PROFILES = (
    "joint_state_zero",
    "baseline_gripper_fixed_group_split",
    "virtual_helpers_without_rigid_body",
    "virtual_helper_topology_collapse",
    "combined_topology_joint_state",
)
FOLLOWERS = ("follower_left", "follower_right")
DECISIONS = {
    "joint_state_zero": "TARGETED_FIX_VERIFIED_RUNTIME_EQUIVALENT",
    "baseline_gripper_fixed_group_split": (
        "TARGETED_FIX_VERIFIED_RUNTIME_STABLE_GRASP_REGRESSION_REQUIRED"
    ),
    "virtual_helpers_without_rigid_body": "REJECTED_REPEATABLE_REGRESSION",
    "virtual_helper_topology_collapse": (
        "TARGETED_TOPOLOGY_FIX_VERIFIED_PHYSICS_EQUIVALENCE_BLOCKED"
    ),
    "combined_topology_joint_state": (
        "VALIDATOR_REDUCED_TO_KNOWN_MIMIC_CONFLICT_"
        "PHYSICS_EQUIVALENCE_BLOCKED"
    ),
}
RUNTIME_ROOT = ROOT / "reports/aloha1_mapping/task7_physicsrules_root_cause_runtime"
RUNTIME_PROFILES = (
    "baseline",
    "joint_state_zero",
    "baseline_gripper_fixed_group_split",
    "virtual_helper_topology_collapse",
    "combined_topology_joint_state",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def _issue_signature(report: dict[str, Any]) -> str:
    normalized = sorted(
        (
            {
                "severity": str(issue["severity"]),
                "rule": str(issue["rule"]),
                "at": str(issue["at"]),
                "message": str(issue["message"]),
            }
            for issue in report["issues"]
        ),
        key=lambda item: (
            item["severity"], item["rule"], item["at"], item["message"]
        ),
    )
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _rule_counts(report: dict[str, Any]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                str(issue["rule"])
                for issue in report["issues"]
                if issue["severity"] in {"ERROR", "FAILURE"}
            ).items()
        )
    )


def _runtime_profile(profile: str) -> dict[str, Any]:
    followers: dict[str, Any] = {}
    for follower in FOLLOWERS:
        repeats = []
        for repeat in (1, 2):
            path = RUNTIME_ROOT / f"{profile}_{follower}_repeat{repeat}.json"
            payload = _load(path)
            repeats.append(
                {
                    "repeat": repeat,
                    "absolute_path": str(path.resolve()),
                    "sha256": _sha256(path),
                    "status": payload["status"],
                    "deterministic_signature": payload["deterministic_signature"],
                    "collision_count": payload["collision_count"],
                    "first_frame_arm_jump_max_abs_rad": payload["summary"][
                        "first_frame_arm_jump_max_abs_rad"
                    ],
                    "static_arm_drift_max_abs_rad": payload["summary"][
                        "static_arm_drift_max_abs_rad"
                    ],
                }
            )
        followers[follower] = {
            "fresh_process_count": 2,
            "all_pass": all(item["status"] == "PASS" for item in repeats),
            "repeat_signatures_identical": (
                repeats[0]["deterministic_signature"]
                == repeats[1]["deterministic_signature"]
            ),
            "repeats": repeats,
        }
    return {
        "all_pass": all(item["all_pass"] for item in followers.values()),
        "all_repeat_signatures_identical": all(
            item["repeat_signatures_identical"] for item in followers.values()
        ),
        "followers": followers,
    }


def build() -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    validator_fresh_process_count = 0
    for profile in PROFILES:
        followers: dict[str, Any] = {}
        unique_new_counts: Counter[str] = Counter()
        for follower in FOLLOWERS:
            repeats = []
            for repeat in (1, 2):
                path = INPUT_ROOT / f"{profile}_{follower}_repeat{repeat}.json"
                payload = _load(path)
                signature = _issue_signature(payload)
                repeats.append(
                    {
                        "repeat": repeat,
                        "absolute_path": str(path.resolve()),
                        "sha256": _sha256(path),
                        "stage_absolute_path": payload["target_absolute_path"],
                        "stage_sha256": payload["target_sha256_before"],
                        "official_status": payload["official_status"],
                        "blocking_issue_count": payload["blocking_issue_count"],
                        "rule_counts": _rule_counts(payload),
                        "deterministic_signature": signature,
                    }
                )
                validator_fresh_process_count += 1
            first_counts = repeats[0]["rule_counts"]
            for rule, count in first_counts.items():
                if rule == "NonAdjacentCollisionMeshesDoNotClash":
                    unique_new_counts[rule] += count
            followers[follower] = {
                "fresh_process_count": 2,
                "repeat_signatures_identical": (
                    repeats[0]["deterministic_signature"]
                    == repeats[1]["deterministic_signature"]
                ),
                "repeats": repeats,
            }
        aggregate_counts: Counter[str] = Counter()
        for follower in followers.values():
            aggregate_counts.update(follower["repeats"][0]["rule_counts"])
        profiles[profile] = {
            "decision": DECISIONS[profile],
            "followers": followers,
            "blocking_rule_counts": dict(sorted(aggregate_counts.items())),
            "new_rule_counts": dict(sorted(unique_new_counts.items())),
        }

    runtime_profiles = {
        profile: _runtime_profile(profile) for profile in RUNTIME_PROFILES
    }
    for profile in PROFILES:
        if profile in runtime_profiles:
            profiles[profile]["runtime"] = runtime_profiles[profile]
        else:
            profiles[profile]["runtime"] = {
                "status": "NOT_RUN_REJECTED_BY_VALIDATOR_REGRESSION"
            }

    review_reports = []
    for side in ("left", "right"):
        path = (
            ROOT
            / "reports/aloha1_mapping"
            / f"aloha1_task7_virtual_helper_failure_screenshot_review_{side}.json"
        )
        review = _load(path)
        review_reports.append(
            {
                "follower": f"follower_{side}",
                "absolute_path": str(path.resolve()),
                "sha256": _sha256(path),
                "status": review["status"],
                "captures": [
                    {
                        "view": capture["view"],
                        "raw_absolute_path": capture["raw_absolute_path"],
                        "raw_sha256": capture["raw_sha256"],
                        "annotated_absolute_path": capture["annotated_absolute_path"],
                        "annotated_sha256": capture["annotated_sha256"],
                    }
                    for capture in review["captures"]
                ],
            }
        )
    profiles["virtual_helpers_without_rigid_body"]["screenshot_escalation"] = {
        "status": (
            "PASS"
            if all(item["status"] == "PASS" for item in review_reports)
            else "FAIL"
        ),
        "trigger": "TWO_IDENTICAL_FRESH_PROCESS_FAILURES_PER_FOLLOWER",
        "retake_history": [
            {
                "attempt": 1,
                "status": "REJECTED",
                "reasons": [
                    "WHOLE_ARM_UPPER_GEOMETRY_CROPPED",
                    "CLOSEUP_BASE_BOX_PROJECTED_OUTSIDE_IMAGE",
                    "HELPER_AND_GRIPPER_LABELS_OVERLAPPED",
                ],
                "artifact_root": str(
                    (
                        ROOT
                        / ".codex/artifacts/20260802-aloha1-task7-physicsrules-root-cause/screenshots"
                    ).resolve()
                ),
            },
            {
                "attempt": 2,
                "status": "PASS",
                "reasons": [],
                "artifact_root": str(
                    (
                        ROOT
                        / ".codex/artifacts/20260802-aloha1-task7-physicsrules-root-cause/screenshots_attempt2"
                    ).resolve()
                ),
            },
        ],
        "review_reports": review_reports,
    }
    profile_decisions = {name: data["decision"] for name, data in profiles.items()}
    mass_audit = _load(HELPER_MASS_AUDIT)
    removed_mass_values = [
        float(item["total_helper_mass_kg"]) for item in mass_audit["followers"]
    ]
    removed_mass_per_follower = max(removed_mass_values)
    report = {
        "schema_version": 1,
        "status": "PARTIAL",
        "scope": "ISOLATED_TASK7_PHYSICSRULES_ROOT_CAUSE_MATRIX",
        "runtime": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
            "asset_validation": "1.1.0",
        },
        "frozen_stage_sha256": FROZEN_SHA256,
        "fresh_process_count": validator_fresh_process_count,
        "validator_fresh_process_count": validator_fresh_process_count,
        "runtime_fresh_process_count": len(RUNTIME_PROFILES) * len(FOLLOWERS) * 2,
        "profile_decisions": profile_decisions,
        "profiles": profiles,
        "runtime_profiles": runtime_profiles,
        "helper_mass_semantics": {
            "audit_absolute_path": str(HELPER_MASS_AUDIT.resolve()),
            "audit_sha256": _sha256(HELPER_MASS_AUDIT),
            "removed_mass_per_follower_kg": removed_mass_per_follower,
            "physically_calibrated": False,
            "calibration_status": mass_audit["physical_calibration_status"],
            "uncompensated_collapse_allowed": mass_audit[
                "uncompensated_collapse_allowed"
            ],
        },
        "mimic_decision": (
            "KEEP_VALID_PHYSX_107_3_AUTHORING_VALIDATOR_1_1_0_FORMULA_MISMATCH"
        ),
        "next_gate": "HELPER_MASS_INERTIA_AGGREGATION_OR_AUTHORIZED_REMOVAL",
        "final_or_default_asset_modified": False,
        "task8": "NOT_RUN",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# ALOHA1 Task 7 PhysicsRules root-cause matrix",
        "",
        f"- Status: `{report['status']}`",
        f"- Validator fresh Isaac processes: `{validator_fresh_process_count}`",
        f"- Runtime fresh Isaac processes: `{report['runtime_fresh_process_count']}`",
        f"- Frozen Stage SHA-256: `{FROZEN_SHA256}`",
        "- Final/default assets modified: `false`",
        "- Task 8: `NOT_RUN`",
        "",
        "| Candidate | Validator result | Decision |",
        "|---|---|---|",
    ]
    for profile in PROFILES:
        left = profiles[profile]["followers"]["follower_left"]["repeats"][0]
        right = profiles[profile]["followers"]["follower_right"]["repeats"][0]
        lines.append(
            f"| `{profile}` | left `{left['blocking_issue_count']}`, "
            f"right `{right['blocking_issue_count']}` blockers | "
            f"`{DECISIONS[profile]}` |"
        )
    lines.extend(
        [
            "",
            "The helper-body removal candidate is rejected: it removes the six original "
            "helper missing-collider findings but creates 57 deterministic "
            "`NonAdjacentCollisionMeshesDoNotClash` findings per follower. Two fresh "
            "processes per follower reproduce the same signature.",
            "",
            "Raw and annotated failure evidence was visually reviewed after one rejected "
            "capture/annotation attempt. Absolute paths and hashes are stored in the JSON report.",
            "",
            "The joint-state-zero candidate is runtime-equivalent to the frozen baseline in "
            "two fresh processes per follower. The fixed-group split is deterministic and "
            "stable but changes active collider paths, so accepted-grasp regression remains "
            "required before promotion.",
            "",
            "The frame-preserving topology collapse removes the six helper findings without "
            "creating the 57 clash errors. The combined candidate leaves only the known "
            "Asset Validation 1.1.0 mimic-formula conflict. However, collapse also removes "
            f"{removed_mass_per_follower:.9g} kg of source-authored, physically uncalibrated "
            "helper mass per follower. It remains diagnostic and non-promotable until mass, "
            "COM and inertia semantics are preserved or explicitly authorized for removal.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    report = build()
    print(
        json.dumps(
            {
                "status": report["status"],
                "validator_fresh_process_count": report[
                    "validator_fresh_process_count"
                ],
                "runtime_fresh_process_count": report["runtime_fresh_process_count"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
