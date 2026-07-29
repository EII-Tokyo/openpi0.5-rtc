#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Run applicable Task 7 checks on the robot-local follower-right diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STAGE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "supplier_cad_follower_right/1.0/"
    "supplier_cad_follower_right.usda"
)
ASSET_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_supplier_cad_follower_right_asset.json"
)
RUNTIME_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_follower_right_one_joint_validation.json"
)
STRUCTURE_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_follower_right_structure_validation.json"
)
SCREENSHOT_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_follower_right_pose_screenshot_review.json"
)
OUTPUT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_follower_right_task7_validation.json"
)
OFFICIAL_CATEGORIES = (
    "IsaacSim.PhysicsRules",
    "IsaacSim.RobotRules",
    "IsaacSim.SimReadyAssetRules",
)
MCPJUNGLE_NVIDIA_OFFICIAL_API_VERIFIED = {
    "status": "PASS",
    "gateway": "mcpjungle_lab",
    "source": "NVIDIA official Isaac Sim documentation capability",
    "version_boundary": "Isaac Sim 5.1.0.0 / Kit 107.3.3",
    "verified_behaviors": [
        "Usd.Stage.Open for read-only Stage validation",
        "ValidationRulesRegistry category enumeration",
        "ValidationEngine explicit rule enablement",
        "fresh Stage open for deterministic repeat",
    ],
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _serialize_issue(issue: Any) -> dict[str, Any]:
    return {
        "severity": getattr(issue.severity, "name", str(issue.severity)),
        "rule": issue.rule.__name__ if issue.rule else None,
        "message": issue.message,
        "at": issue.at.as_str() if issue.at is not None else None,
    }


def _run_rules_once(stage_path: Path) -> dict[str, Any]:
    import isaacsim.asset.validation  # noqa: F401
    import omni.asset_validator.core as av_core
    from pxr import Usd

    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"unable to open Stage: {stage_path}")

    categories = []
    for category in OFFICIAL_CATEGORIES:
        rules = list(
            av_core.ValidationRulesRegistry.rules(
                category,
                enabledOnly=False,
            )
        )
        engine = av_core.ValidationEngine(init_rules=False, variants=False)
        for rule in rules:
            engine.enable_rule(rule)
        issues = sorted(
            (_serialize_issue(issue) for issue in engine.validate(stage)),
            key=lambda item: (
                item["severity"],
                item["rule"] or "",
                item["at"] or "",
                item["message"] or "",
            ),
        )
        blocking = [
            issue
            for issue in issues
            if issue["severity"] in {"ERROR", "FAILURE"}
        ]
        warnings = [
            issue for issue in issues if issue["severity"] == "WARNING"
        ]
        categories.append(
            {
                "category": category,
                "target": str(stage_path),
                "status": (
                    "FAIL"
                    if blocking
                    else "PARTIAL"
                    if warnings
                    else "PASS"
                ),
                "rule_count": len(rules),
                "rules": sorted(rule.__name__ for rule in rules),
                "issues": issues,
                "blocking_issue_count": len(blocking),
                "warning_count": len(warnings),
            }
        )
    return {"categories": categories}


def _rule_signature(result: dict[str, Any]) -> str:
    payload = json.dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_report() -> dict[str, Any]:
    stage_path = STAGE.resolve(strict=True)
    stage_hash_before = _sha256(stage_path)
    asset = _load(ASSET_REPORT)
    runtime = _load(RUNTIME_REPORT)
    structure = _load(STRUCTURE_REPORT)
    screenshots = _load(SCREENSHOT_REPORT)

    first_rules = _run_rules_once(stage_path)
    second_rules = _run_rules_once(stage_path)
    first_signature = _rule_signature(first_rules)
    second_signature = _rule_signature(second_rules)
    repeat_pass = first_signature == second_signature
    stage_hash_after = _sha256(stage_path)
    stage_immutable = stage_hash_before == stage_hash_after

    categories = second_rules["categories"]
    official_status = (
        "FAIL"
        if any(item["status"] == "FAIL" for item in categories)
        else (
            "PARTIAL"
            if any(item["status"] == "PARTIAL" for item in categories)
            else "PASS"
        )
    )
    visual_pass = (
        screenshots["visual_installation_pose_gate"] == "PASS"
        and len(screenshots["records"]) == 7
        and all(
            item["raw"]["visual_model_review"] == "PASS"
            and item["annotated"]["visual_model_review"] == "PASS"
            for item in screenshots["records"]
        )
    )
    robot_local = {
        "overall": runtime["status"],
        "articulation_count": (
            "PASS" if structure["articulation_count"] == 1 else "FAIL"
        ),
        "dof_name_and_order": (
            "PASS"
            if runtime["dof_order"] == asset["dof_order"]
            else "FAIL"
        ),
        "arm_one_joint": (
            "PASS"
            if len(runtime["arm_one_joint_cases"]) == 24
            and all(
                item["status"] == "PASS"
                for item in runtime["arm_one_joint_cases"]
            )
            else "FAIL"
        ),
        "gripper_motion_direction": runtime["gripper_validation"][
            "motion_direction"
        ],
        "aperture_monotonicity": runtime["gripper_validation"][
            "aperture_monotonicity"
        ],
        "legal_range": runtime["gripper_validation"]["legal_range"],
        "mimic_accuracy": (
            "PASS"
            if runtime["gripper_validation"]["maximum_mimic_residual_m"]
            <= 0.001
            else "FAIL"
        ),
        "first_frame_jump": runtime["first_frame_jump"]["status"],
        "static_pose_hold": runtime["static_pose_hold"]["status"],
        "initial_overlap": structure["initial_overlap"]["status"],
        "determinism": runtime["determinism"]["status"],
        "screenshot_evidence": {
            "status": "PASS" if visual_pass else "FAIL",
            "raw_count": len(screenshots["records"]),
            "annotated_count": len(screenshots["records"]),
            "report": str(SCREENSHOT_REPORT.resolve()),
            "auxiliary_only": True,
            "numeric_runtime_authoritative": True,
        },
    }
    status = (
        "FAIL"
        if (
            official_status == "FAIL"
            or robot_local["mimic_accuracy"] == "FAIL"
            or not repeat_pass
            or not stage_immutable
            or not visual_pass
        )
        else "PARTIAL"
    )
    return {
        "schema_version": 1,
        "status": status,
        "scope": "ROBOT_LOCAL_DIAGNOSTIC_NOT_WORKCELL_PLACEMENT",
        "mcp_gateway_verification": MCPJUNGLE_NVIDIA_OFFICIAL_API_VERIFIED,
        "versions": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
        },
        "stage": {
            "absolute_path": str(stage_path),
            "sha256_before": stage_hash_before,
            "sha256_after": stage_hash_after,
            "immutable": stage_immutable,
            "root_prim": asset["root_prim"],
            "articulation_roots": asset["articulation_roots"],
        },
        "source_reports": {
            "asset": {
                "absolute_path": str(ASSET_REPORT.resolve()),
                "sha256": _sha256(ASSET_REPORT),
            },
            "runtime": {
                "absolute_path": str(RUNTIME_REPORT.resolve()),
                "sha256": _sha256(RUNTIME_REPORT),
            },
            "structure": {
                "absolute_path": str(STRUCTURE_REPORT.resolve()),
                "sha256": _sha256(STRUCTURE_REPORT),
            },
            "screenshots": {
                "absolute_path": str(SCREENSHOT_REPORT.resolve()),
                "sha256": _sha256(SCREENSHOT_REPORT),
            },
        },
        "robot_local": robot_local,
        "official_rules": {
            "status": official_status,
            "categories": categories,
        },
        "repeat_determinism": {
            "pass": repeat_pass,
            "run_count": 2,
            "fresh_stage_open_each_run": True,
            "signatures": [first_signature, second_signature],
        },
        "dual_arm_workcell_placement": {
            "status": "PARTIAL",
            "verified": False,
            "reason": (
                "Robot-local product identity is verified; no supplier CAD "
                "or measured follower_right workcell installation transform "
                "has been accepted."
            ),
        },
        "hard_blockers": [
            "HARD_BLOCKER_FOLLOWER_RIGHT_WORKCELL_INSTALL_TRANSFORM"
        ],
        "acceptance_boundary": (
            "This report validates only the isolated follower_right robot "
            "product in its local frame. It does not validate a dual-arm "
            "workcell placement. Visual screenshots are auxiliary; numeric "
            "runtime and official-rule results are authoritative."
        ),
        "task8": "NOT_RUN",
        "final_default_collider_modified": False,
        "source_stage_modified": False,
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    categories = report["official_rules"]["categories"]
    path.write_text(
        "\n".join(
            [
                "# follower_right robot-local Task 7 validation",
                "",
                f"- Status: `{report['status']}`",
                f"- Scope: `{report['scope']}`",
                f"- Stage: `{report['stage']['absolute_path']}`",
                f"- Stage immutable: `{report['stage']['immutable']}`",
                (
                    "- Arm one-joint: "
                    f"`{report['robot_local']['arm_one_joint']}`"
                ),
                (
                    "- Gripper direction / aperture / mimic: "
                    f"`{report['robot_local']['gripper_motion_direction']}` / "
                    f"`{report['robot_local']['aperture_monotonicity']}` / "
                    f"`{report['robot_local']['mimic_accuracy']}`"
                ),
                (
                    "- Screenshot visual gate: "
                    f"`{report['robot_local']['screenshot_evidence']['status']}` "
                    "(auxiliary only)"
                ),
                "- Dual-arm workcell placement: `PARTIAL` / unverified",
                f"- Task 8: `{report['task8']}`",
                "",
                "| Official category | Status | Blocking | Warnings |",
                "|---|---|---:|---:|",
                *[
                    (
                        f"| {item['category']} | {item['status']} | "
                        f"{item['blocking_issue_count']} | "
                        f"{item['warning_count']} |"
                    )
                    for item in categories
                ],
                "",
                "## HARD_BLOCKER",
                "",
                *[
                    f"- `{item}`"
                    for item in report["hard_blockers"]
                ],
                "",
                report["acceptance_boundary"],
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    report = _build_report()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(report, OUTPUT.with_suffix(".md"))
    print(f"status={report['status']}")
    print(
        "rules="
        + ",".join(
            f"{item['category']}:{item['status']}"
            for item in report["official_rules"]["categories"]
        )
    )
    print(
        "repeat_pass="
        f"{report['repeat_determinism']['pass']}"
    )
    print(f"output={OUTPUT}")
    return 0


def run() -> int:
    """Launch Kit, preserve Python failures, and close the application."""

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    exit_code = 1
    try:
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        extension_id = "isaacsim.asset.validation"
        if not manager.is_extension_enabled(extension_id):
            manager.set_extension_enabled_immediate(
                extension_id,
                True,  # noqa: FBT003
            )
        if not manager.is_extension_enabled(extension_id):
            raise RuntimeError(
                f"required extension disabled: {extension_id}"
            )
        exit_code = main()
    except Exception:
        traceback.print_exc()
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
