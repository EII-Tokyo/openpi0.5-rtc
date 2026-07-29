#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Audit Task 7A helper-link source and composed USD semantics read-only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import traceback
from typing import Any

from tools.aloha1_mapping.task7a_helper_link_audit import audit_urdf_helper_links

ROOT = Path(__file__).resolve().parents[1]
STAGE = (
    ROOT / "assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/1.0/aloha1_signal_correspondence_workcell.usda"
)
EXPECTED_STAGE_SHA256 = "d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf"
REPORT_ROOT = ROOT / "reports/aloha1_mapping"
JSON_OUTPUT = REPORT_ROOT / "aloha1_task7a_helper_link_semantics.json"
MD_OUTPUT = REPORT_ROOT / "aloha1_task7a_helper_link_semantics.md"
ARTIFACT_ROOT = ROOT / ".codex/artifacts/20260729-aloha1-task7a-acceptance-separation"
XACRO = (
    ROOT / "external/ros2-essentials/aloha_ws/src/"
    "interbotix_ros_manipulators/interbotix_ros_xsarms/"
    "interbotix_xsarm_descriptions/urdf/aloha_vx300s.urdf.xacro"
)
SOURCE_REPOSITORY = ROOT / "external/ros2-essentials/aloha_ws/src/interbotix_ros_manipulators"
OFFICIAL_REPORT = REPORT_ROOT / "aloha1_signal_correspondence_official_rules.json"
CAD_MAPPING_REPORT = REPORT_ROOT / "aloha_public_cad_gripper_mapping.json"
PHYSICS_RULE_SOURCE = (
    ROOT
    / ".venv_issac/lib/python3.11/site-packages/isaacsim/exts/"
    "isaacsim.asset.validation/isaacsim/asset/validation/physics_rules.py"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(SOURCE_REPOSITORY), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _line_numbers(path: Path, needles: tuple[str, ...]) -> dict[str, int]:
    result: dict[str, int] = {}
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        for needle in needles:
            if needle in line and needle not in result:
                result[needle] = number
    return result


def _references(prim: Any) -> list[dict[str, str]]:
    if not prim.HasAuthoredReferences():
        return []
    reference_list = prim.GetMetadata("references")
    items = reference_list.GetAddedOrExplicitItems()
    return [
        {
            "asset_path": str(item.assetPath),
            "prim_path": str(item.primPath),
        }
        for item in items
    ]


def _usd_helper_record(stage: Any, prim_path: str) -> dict[str, Any]:
    from pxr import Usd
    from pxr import UsdPhysics

    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"missing composed helper prim {prim_path}")
    descendants = list(Usd.PrimRange(prim))
    colliders = [str(item.GetPath()) for item in descendants if item.HasAPI(UsdPhysics.CollisionAPI)]
    return {
        "prim_path": prim_path,
        "type_name": prim.GetTypeName(),
        "applied_schemas": list(prim.GetAppliedSchemas()),
        "has_rigid_body_api": prim.HasAPI(UsdPhysics.RigidBodyAPI),
        "descendant_collision_api_paths": colliders,
        "descendant_collision_api_count": len(colliders),
        "child_prim_names": [item.GetName() for item in prim.GetChildren()],
        "prim_stack": [
            {
                "layer": str(spec.layer.identifier),
                "path": str(spec.path),
                "specifier": str(spec.specifier),
            }
            for spec in prim.GetPrimStack()
        ],
    }


def _official_findings() -> dict[str, dict[str, Any]]:
    report = json.loads(OFFICIAL_REPORT.read_text(encoding="utf-8"))
    findings: dict[str, dict[str, Any]] = {}
    for target in report["targets"]:
        for issue in target.get("issues", []):
            if issue.get("rule") != "RigidBodyHasCollider":
                continue
            location = str(issue["at"]).removeprefix("Prim <").removesuffix(">")
            findings[location] = {
                "severity": issue["severity"],
                "message": issue["message"],
                "target_name": target["target_name"],
                "category": target["category"],
            }
    return findings


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ALOHA1 Task 7A helper-link semantics",
        "",
        f"- Status: `{report['status']}`",
        f"- Frozen Stage unchanged: `{report['stage']['stage_modified'] is False}`",
        f"- Official findings covered: `{report['coverage']['official_findings']}`",
        f"- Helper records: `{report['coverage']['helper_records']}`",
        "",
        "## Result",
        "",
        "The six findings are real literal Isaac Sim 5.1 "
        "`RigidBodyHasCollider` failures. The pinned Xacro and generated URDFs "
        "define no visual or collision geometry for these links. "
        "`ee_arm_link` and `fingers_link` are geometry-free kinematic helper "
        "frames; `ee_gripper_link` is a fixed end-effector frame alias. Their "
        "1 g inertial blocks do not define a physical shape.",
        "",
        "No collider was invented and no `RigidBodyAPI` was removed. Either "
        "change could alter articulation semantics and requires a separate "
        "source-backed promotion candidate plus regression.",
        "",
        "## Per-link evidence",
        "",
        "| Prim | Source semantic class | RigidBodyAPI | Descendant colliders |",
        "|---|---|---:|---:|",
    ]
    lines.extend(
        (
            "| `{prim}` | `{semantic}` | `{rigid}` | `{colliders}` |".format(
                prim=record["usd"]["prim_path"],
                semantic=record["urdf"]["semantic_class"],
                rigid=record["usd"]["has_rigid_body_api"],
                colliders=record["usd"]["descendant_collision_api_count"],
            )
        )
        for record in report["helpers"]
    )
    lines.extend(
        [
            "",
            "## Acceptance boundary",
            "",
            "- Runtime control: these findings do not invalidate measured "
            "DOF motion, target/readback, or deterministic swept-path results.",
            "- Asset promotion: remains `PARTIAL`; literal official failures remain unsuppressed.",
            "- Supplier CAD maps geometry to the handed finger links, not to these six abstract helper frames.",
            "- Task 7B: `NOT_RUN`.",
            "- Task 8: `NOT_RUN`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    from pxr import Usd

    stage_path = STAGE.resolve(strict=True)
    stage_hash_before = _sha256(stage_path)
    if stage_hash_before != EXPECTED_STAGE_SHA256:
        raise RuntimeError(f"approved Stage hash mismatch: {stage_hash_before} != {EXPECTED_STAGE_SHA256}")
    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"could not open {stage_path}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim or str(default_prim.GetPath()) != "/World":
        raise RuntimeError("approved Stage default prim is not /World")

    official_findings = _official_findings()
    helpers: list[dict[str, Any]] = []
    for robot in ("follower_left", "follower_right"):
        urdf_path = ROOT / "generated/urdf" / f"{robot}.urdf"
        urdf_records = audit_urdf_helper_links(urdf_path, robot)
        robot_root = "vx300s_left" if robot == "follower_left" else "vx300s_right"
        for name, urdf_record in urdf_records.items():
            asset_prim_path = f"/{robot}/{robot_root}/{name}"
            composed_prim_path = f"/World{asset_prim_path}"
            usd_record = _usd_helper_record(stage, composed_prim_path)
            if asset_prim_path not in official_findings:
                raise RuntimeError(f"official finding missing for helper {asset_prim_path}")
            helpers.append(
                {
                    "robot": robot,
                    "asset_prim_path": asset_prim_path,
                    "composed_prim_path": composed_prim_path,
                    "urdf": urdf_record,
                    "usd": usd_record,
                    "official_rule": official_findings[asset_prim_path],
                    "supplier_cad_geometry_mapped_to_helper": False,
                    "invent_collider_allowed": False,
                    "remove_rigid_body_api_allowed": False,
                }
            )

    source_lines = _line_numbers(
        XACRO,
        tuple(
            f"/{suffix}"
            for suffix in (
                "ee_arm_link",
                "fingers_link",
                "ee_gripper_link",
            )
        ),
    )
    stage_hash_after = _sha256(stage_path)
    source_cad = json.loads(CAD_MAPPING_REPORT.read_text(encoding="utf-8"))
    report = {
        "schema_version": 1,
        "status": (
            "PASS"
            if len(helpers) == 6 and len(official_findings) == 6 and stage_hash_before == stage_hash_after
            else "FAIL"
        ),
        "stage": {
            "absolute_path": str(stage_path),
            "sha256_before": stage_hash_before,
            "sha256_after": stage_hash_after,
            "stage_modified": stage_hash_before != stage_hash_after,
            "default_prim": str(default_prim.GetPath()),
            "root_sublayers": list(stage.GetRootLayer().subLayerPaths),
            "root_references": {
                str(prim.GetPath()): _references(prim) for prim in stage.Traverse() if prim.HasAuthoredReferences()
            },
        },
        "runtime": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
            "asset_validation_extension": "1.1.0",
        },
        "source_xacro": {
            "absolute_path": str(XACRO.resolve(strict=True)),
            "sha256": _sha256(XACRO),
            "repository": {
                "root": str(SOURCE_REPOSITORY.resolve(strict=True)),
                "commit": _git_output("rev-parse", "HEAD"),
                "branch": _git_output("branch", "--show-current"),
                "remote": _git_output("remote", "get-url", "origin"),
                "dirty": bool(_git_output("status", "--short")),
            },
            "first_matching_lines": source_lines,
        },
        "supplier_cad_crosscheck": {
            "report": str(CAD_MAPPING_REPORT.resolve(strict=True)),
            "report_sha256": _sha256(CAD_MAPPING_REPORT),
            "mapped_links": {
                "CAD +X": source_cad["cad_to_urdf_frame_mapping"]["positive_cad_x_link"],
                "CAD -X": source_cad["cad_to_urdf_frame_mapping"]["negative_cad_x_link"],
            },
            "helper_geometry_mapping": "NONE_RECORDED",
        },
        "official_rule_source": {
            "report": str(OFFICIAL_REPORT.resolve(strict=True)),
            "report_sha256": _sha256(OFFICIAL_REPORT),
            "installed_python_source": {
                "absolute_path": str(
                    PHYSICS_RULE_SOURCE.resolve(strict=True)
                ),
                "sha256": _sha256(PHYSICS_RULE_SOURCE),
                "class_name": "RigidBodyHasCollider",
                "class_first_line": _line_numbers(
                    PHYSICS_RULE_SOURCE,
                    ("class RigidBodyHasCollider",),
                )["class RigidBodyHasCollider"],
            },
            "literal_status": "FAIL",
            "finding_count": len(official_findings),
            "suppressed": False,
        },
        "coverage": {
            "official_findings": len(official_findings),
            "helper_records": len(helpers),
            "expected_each": 6,
        },
        "helpers": helpers,
        "decision": {
            "runtime_control_effect": "NON_BLOCKING_SOURCE_GEOMETRY_BOUNDARY",
            "asset_promotion_effect": "PARTIAL_LITERAL_RULE_FAILURE_REMAINS",
            "promotion_candidate_action": ("DO_NOT_INVENT_COLLIDER_OR_REMOVE_RIGID_BODY_API"),
        },
        "real_robot_connected": False,
        "remote_192_168_1_103_accessed": False,
        "task_7b": "NOT_RUN",
        "task_8": "NOT_RUN",
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MD_OUTPUT.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "helpers": len(helpers),
                "stage_modified": report["stage"]["stage_modified"],
                "json": str(JSON_OUTPUT.resolve()),
                "markdown": str(MD_OUTPUT.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if report["status"] == "PASS" else 1


def run() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "create_new_stage": False,
            "disable_viewport_updates": True,
        }
    )
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
