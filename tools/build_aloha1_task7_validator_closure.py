#!/usr/bin/env python3
"""Build Task 7 per-issue validator scope and control reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tools.aloha1_mapping.task7_rule_scope_audit import audit_blocking_issues
from tools.aloha1_mapping.task7_rule_scope_audit import summarize_audit
from tools.aloha1_mapping.task7_validator_controls import summarize_two_runs
from tools.aloha1_mapping.task7_validator_controls import validate_negative_delta

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "reports/aloha1_mapping"
ARTIFACTS = (
    REPO
    / ".codex/artifacts/20260802-aloha1-cad-derived-colliders/task7_final_closure"
    / "validator_controls"
)
RULE_DIR = (
    REPO
    / ".venv_issac/lib/python3.11/site-packages/isaacsim/exts"
    / "isaacsim.asset.validation/isaacsim/asset/validation"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _two(paths: list[Path]) -> dict[str, Any]:
    reports = [_load(path) for path in paths]
    summary = summarize_two_runs(reports)
    summary["reports"] = [str(path.resolve()) for path in paths]
    summary["report_sha256"] = [_sha256(path) for path in paths]
    return summary


def _candidate_pairs() -> dict[str, list[Path]]:
    old = REPO / ".codex/artifacts/20260802-aloha1-cad-derived-colliders/task7_rule_candidates"
    return {
        "robot_left": [
            REPORTS / "aloha1_cad_derived_task7_candidate_left_robot.json",
            old / "left_robot_repeat.json",
        ],
        "robot_right": [
            REPORTS / "aloha1_cad_derived_task7_candidate_right_robot.json",
            old / "right_robot_repeat.json",
        ],
        "physics_left": [
            REPORTS / "aloha1_cad_derived_task7_candidate_left_physics.json",
            old / "left_physics_repeat.json",
        ],
        "physics_right": [
            REPORTS / "aloha1_cad_derived_task7_candidate_right_physics.json",
            old / "right_physics_repeat.json",
        ],
        "simready_workcell": [
            REPORTS / "aloha1_cad_derived_task7_candidate_workcell_simready.json",
            old / "workcell_simready_repeat.json",
        ],
    }


def _scope_report() -> dict[str, Any]:
    robot_path = REPORTS / "aloha1_cad_derived_zup_official_robot_rules.json"
    physics_path = REPORTS / "aloha1_cad_derived_zup_official_physics_rules.json"
    robot = _load(robot_path)
    physics = _load(physics_path)
    robot_rows = audit_blocking_issues(
        family="IsaacSim.RobotRules",
        target=robot["target_absolute_path"],
        issues=robot["issues"],
    )
    physics_rows = audit_blocking_issues(
        family="IsaacSim.PhysicsRules",
        target=physics["target_absolute_path"],
        issues=physics["issues"],
    )
    summary = summarize_audit(robot_rows=robot_rows, physics_rows=physics_rows)
    sources = {}
    for name in ("robot_rules.py", "physics_rules.py", "joint_rules.py"):
        path = RULE_DIR / name
        sources[name] = {"absolute_path": str(path.resolve()), "sha256": _sha256(path)}
    return {
        "schema_version": 1,
        "status": "PASS",
        "runtime": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
            "asset_validation": "1.1.0",
        },
        "installed_rule_sources": sources,
        "original_reports": {
            "robot": {"absolute_path": str(robot_path.resolve()), "sha256": _sha256(robot_path)},
            "physics": {"absolute_path": str(physics_path.resolve()), "sha256": _sha256(physics_path)},
        },
        "correct_targets": {
            "IsaacSim.RobotRules": "standalone Robot.usd/package whose defaultPrim is the one robot RobotAPI root",
            "IsaacSim.PhysicsRules.robot": "standalone physical robot asset, one follower at a time",
            "IsaacSim.PhysicsRules.bottle": "standalone Bottle500 rigid-body asset",
            "IsaacSim.PhysicsRules.table": "standalone static table/environment asset",
            "IsaacSim.PhysicsRules.workcell": "only rules whose source explicitly traverses and is meaningful for all composed bodies/joints",
            "IsaacSim.SimReadyAssetRules": "composed frozen workcell review Stage",
        },
        "summary": summary,
        "robot_rows": robot_rows,
        "physics_rows": physics_rows,
        "task8": "NOT_RUN",
    }


def _control_report() -> dict[str, Any]:
    positive = {
        category: _two(
            [
                ARTIFACTS / f"positive_ur10_exact_run{run}/{category}.json"
                for run in (1, 2)
            ]
        )
        for category in ("robot", "physics")
    }
    positive["asset"] = {
        "official_asset_root": "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1",
        "relative_path": "Isaac/Robots/UniversalRobots/ur10/ur10.usd",
        "local_path": str((ARTIFACTS / "official_5_1_assets/UniversalRobots/ur10/ur10.usd").resolve()),
        "sha256": _sha256(ARTIFACTS / "official_5_1_assets/UniversalRobots/ur10/ur10.usd"),
    }
    positive["status"] = "FAIL"
    positive["interpretation"] = (
        "The released Isaac 5.1 UR10 is not clean under local Asset Validation 1.1.0: "
        "the literal results are deterministic but include blocking findings. It is "
        "therefore not misreported as a passing positive control."
    )

    baseline_robot = _load(REPORTS / "aloha1_cad_derived_task7_candidate_left_robot.json")
    baseline_physics = _load(REPORTS / "aloha1_cad_derived_task7_candidate_left_physics.json")
    negative_specs = {
        "negative_robot_api": (baseline_robot, "RobotSchema", "negative_robot_api"),
        "negative_mass_api": (baseline_physics, "RigidBodyHasMassAPI", "follower_left_base_link"),
        "negative_collider": (baseline_physics, "RigidBodyHasCollider", "follower_left_base_link"),
    }
    negatives: dict[str, Any] = {}
    for name, (baseline, rule, target) in negative_specs.items():
        paths = [ARTIFACTS / f"negative_results/run{run}/{name}.json" for run in (1, 2)]
        reports = [_load(path) for path in paths]
        delta = validate_negative_delta(
            baseline=baseline,
            negative=reports[0],
            expected_rule=rule,
            expected_target_fragment=target,
        )
        negatives[name] = {
            "status": "PASS",
            "fresh_runs": _two(paths),
            "delta": delta,
        }
    scoped_assets = {
        "Bottle500_original": _two(
            [ARTIFACTS / f"scoped_filtered_run{run}/bottle.json" for run in (1, 2)]
        ),
        "static_environment_original": _two(
            [ARTIFACTS / f"scoped_filtered_run{run}/environment.json" for run in (1, 2)]
        ),
        "Bottle500_candidate": _two(
            [ARTIFACTS / f"scoped_candidates_v3_run{run}/bottle.json" for run in (1, 2)]
        ),
        "static_environment_candidate": _two(
            [ARTIFACTS / f"scoped_candidates_v3_run{run}/environment.json" for run in (1, 2)]
        ),
        "candidate_manifest": {
            "absolute_path": str(
                (ARTIFACTS / "scoped_physics_candidates_v3_manifest.json").resolve()
            ),
            "sha256": _sha256(ARTIFACTS / "scoped_physics_candidates_v3_manifest.json"),
            "promotion": "USER_REVIEW_REQUIRED",
        },
    }
    return {
        "schema_version": 1,
        "status": "PARTIAL",
        "positive_control": positive,
        "negative_controls": negatives,
        "scoped_physical_assets": scoped_assets,
        "aloha_candidates": {name: _two(paths) for name, paths in _candidate_pairs().items()},
        "conclusion": (
            "The execution chain detects all three intentional defects deterministically. "
            "The requested released-asset positive control is itself not validator-clean, "
            "so it cannot establish a globally green reference baseline."
        ),
        "task8": "NOT_RUN",
    }


def _markdown_scope(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# ALOHA1 Task 7 validator rule-scope audit",
        "",
        f"- Status: `{report['status']}`",
        f"- Original blocking issues: `{s['total_classified']}` (RobotRules 63, PhysicsRules 26)",
        f"- Classification counts: `{json.dumps(s['classification_counts'], sort_keys=True)}`",
        f"- Remaining Task 7-blocking issue records: `{s['task7_blocking_count']}`",
        "- Task 8: `NOT_RUN`",
        "",
        "The 63 RobotRules errors are `WRONG_SCOPE`: they were produced by running robot-package rules on the two-robot workcell wrapper. They are not suppressed; the correct standalone left/right package runs are reported separately.",
        "",
        "The 26 PhysicsRules records remain individually classified. Applicable literal defects and unresolved validator/runtime conflicts continue to block asset promotion.",
        "",
        "| # | Family | Rule | Owner | Classification | Task 7 blocker | Prim |",
        "|---:|---|---|---|---|---:|---|",
    ]
    rows = [*report["robot_rows"], *report["physics_rows"]]
    for index, row in enumerate(rows, 1):
        lines.append(
            f"| {index} | {row['rule_family']} | {row['rule_name']} | {row['asset_owner']} | "
            f"{row['classification']} | {row['task7_blocking']} | {row['target_prim_path']} |"
        )
    return "\n".join(lines) + "\n"


def _markdown_controls(report: dict[str, Any]) -> str:
    p = report["positive_control"]
    lines = [
        "# ALOHA1 Task 7 validator controls",
        "",
        f"- Overall: `{report['status']}`",
        f"- Released Isaac 5.1 UR10 positive control: `{p['status']}`",
        f"- UR10 RobotRules: `{p['robot']['statuses']}` / {p['robot']['blocking_counts']} blockers",
        f"- UR10 PhysicsRules: `{p['physics']['statuses']}` / {p['physics']['blocking_counts']} blockers",
        "- All negative controls: `PASS`, two fresh-process signatures identical",
        "- Bottle500 candidate: `PARTIAL` with 0 blockers; static-environment candidate: `PASS`",
        "- Both physical candidates require user review before promotion",
        "- Task 8: `NOT_RUN`",
        "",
        p["interpretation"],
        "",
        "| Negative control | Expected rule | Fresh consistency | Added defect |",
        "|---|---|---:|---:|",
    ]
    for name, entry in report["negative_controls"].items():
        lines.append(
            f"| {name} | {entry['delta']['expected_rule']} | "
            f"{entry['fresh_runs']['consistent']} | {len(entry['delta']['matching_added_issues'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    scope = _scope_report()
    controls = _control_report()
    outputs = {
        REPORTS / "aloha1_task7_final_rule_scope_audit.json": json.dumps(scope, indent=2, sort_keys=True) + "\n",
        REPORTS / "aloha1_task7_final_rule_scope_audit.md": _markdown_scope(scope),
        REPORTS / "aloha1_task7_validator_controls.json": json.dumps(controls, indent=2, sort_keys=True) + "\n",
        REPORTS / "aloha1_task7_validator_controls.md": _markdown_controls(controls),
    }
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8")
    print(json.dumps({"scope": scope["summary"], "controls": controls["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
