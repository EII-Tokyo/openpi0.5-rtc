#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Read-only source/composition audit for Task 7 missing-collider findings."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import traceback
from typing import Any
import xml.etree.ElementTree as ET

from tools.aloha1_mapping.task7_physicsrules_root_cause import classify_collider_finding

ROOT = Path(__file__).resolve().parents[1]
FROZEN_STAGE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0"
    / "aloha1_cad_derived_full_body_collider_gripper_decomposition_"
    "tabletop_zero_z_up_meters_diagnostic.usda"
)
FROZEN_SHA256 = (
    "327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9"
)
CANDIDATES = {
    "follower_left": (
        ROOT
        / "assets/Trossen/ALOHA1/1.0/diagnostics/"
        "cad_derived_task7_rule_candidates/1.0/Trossen/vx300s_left/1.0/"
        "vx300s_left.usda",
        "/vx300s_left",
    ),
    "follower_right": (
        ROOT
        / "assets/Trossen/ALOHA1/1.0/diagnostics/"
        "cad_derived_task7_rule_candidates/1.0/Trossen/vx300s_right/1.0/"
        "vx300s_right.usda",
        "/vx300s_right",
    ),
}
SUFFIXES = (
    "ee_arm_link",
    "ee_gripper_link",
    "fingers_link",
    "gripper_bar_link",
)
OUTPUT_JSON = (
    ROOT
    / "reports/aloha1_mapping/aloha1_task7_helper_link_collider_audit.json"
)
OUTPUT_MD = OUTPUT_JSON.with_suffix(".md")
COLLIDER_STAGE_REPORT = (
    ROOT / "reports/aloha1_mapping/aloha1_cad_derived_collider_stage.json"
)
RULE_SOURCE = (
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


def _joint_inventory(
    root: ET.Element,
) -> tuple[
    dict[str, list[dict[str, str]]],
    dict[str, list[dict[str, str]]],
]:
    incoming: dict[str, list[dict[str, str]]] = {}
    outgoing: dict[str, list[dict[str, str]]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        record = {
            "name": str(joint.get("name")),
            "type": str(joint.get("type")),
            "parent": str(parent.get("link")),
            "child": str(child.get("link")),
        }
        incoming.setdefault(record["child"], []).append(record)
        outgoing.setdefault(record["parent"], []).append(record)
    return incoming, outgoing


def _urdf_records(robot: str) -> dict[str, dict[str, Any]]:
    urdf = (ROOT / "generated/urdf" / f"{robot}.urdf").resolve(strict=True)
    tree = ET.parse(urdf)
    root = tree.getroot()
    links = {str(link.get("name")): link for link in root.findall("link")}
    incoming, outgoing = _joint_inventory(root)
    records: dict[str, dict[str, Any]] = {}
    for suffix in SUFFIXES:
        name = f"{robot}_{suffix}"
        link = links[name]
        visuals = link.findall("visual")
        collisions = link.findall("collision")
        inertials = link.findall("inertial")
        collision_meshes = [
            str(mesh.get("filename"))
            for collision in collisions
            for mesh in collision.findall("./geometry/mesh")
        ]
        visual_meshes = [
            str(mesh.get("filename"))
            for visual in visuals
            for mesh in visual.findall("./geometry/mesh")
        ]
        incoming_records = incoming.get(name, [])
        records[name] = {
            "absolute_path": str(urdf),
            "sha256": _sha256(urdf),
            "link_name": name,
            "visual_count": len(visuals),
            "collision_count": len(collisions),
            "inertial_count": len(inertials),
            "visual_meshes": visual_meshes,
            "collision_meshes": collision_meshes,
            "incoming_joints": incoming_records,
            "outgoing_joints": outgoing.get(name, []),
            "classification": classify_collider_finding(
                visual_count=len(visuals),
                collision_count=len(collisions),
                incoming_joint_types=[item["type"] for item in incoming_records],
            ),
        }
    return records


def _prim_record(stage: Any, prim_path: str) -> dict[str, Any]:
    from pxr import Usd
    from pxr import UsdPhysics

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"missing candidate prim: {prim_path}")
    descendants = list(Usd.PrimRange(prim))
    collision_paths = sorted(
        str(item.GetPath())
        for item in descendants
        if item.IsActive() and item.HasAPI(UsdPhysics.CollisionAPI)
    )
    return {
        "prim_path": prim_path,
        "type_name": prim.GetTypeName(),
        "active": prim.IsActive(),
        "applied_schemas": list(prim.GetAppliedSchemas()),
        "has_rigid_body_api": prim.HasAPI(UsdPhysics.RigidBodyAPI),
        "descendant_collision_paths": collision_paths,
        "descendant_collision_count": len(collision_paths),
        "children": [
            {"path": str(child.GetPath()), "active": child.IsActive()}
            for child in prim.GetAllChildren()
        ],
        "prim_stack": [
            {
                "layer": str(spec.layer.identifier),
                "path": str(spec.path),
                "specifier": str(spec.specifier),
            }
            for spec in prim.GetPrimStack()
        ],
    }


def _official_issues(side: str) -> dict[str, dict[str, Any]]:
    report_path = (
        ROOT
        / "reports/aloha1_mapping"
        / f"aloha1_cad_derived_task7_candidate_{side}_physics.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    issues = {}
    for item in report["issues"]:
        if item.get("rule") != "RigidBodyHasCollider":
            continue
        prim_path = str(item["at"]).removeprefix("Prim <").removesuffix(">")
        issues[prim_path] = dict(item)
    return issues


def build_report() -> dict[str, Any]:
    from pxr import Usd

    frozen = FROZEN_STAGE.resolve(strict=True)
    frozen_before = _sha256(frozen)
    if frozen_before != FROZEN_SHA256:
        raise RuntimeError("frozen Stage hash mismatch")
    group_report = json.loads(COLLIDER_STAGE_REPORT.read_text(encoding="utf-8"))
    group_coverage = group_report["gripper_bar_fixed_group_coverage"]
    findings: list[dict[str, Any]] = []
    candidate_hashes: dict[str, Any] = {}
    for robot, (candidate_path, root_path) in CANDIDATES.items():
        candidate = candidate_path.resolve(strict=True)
        before = _sha256(candidate)
        stage = Usd.Stage.Open(str(candidate), Usd.Stage.LoadAll)
        if stage is None:
            raise RuntimeError(f"cannot open {candidate}")
        source_records = _urdf_records(robot)
        side = robot.removeprefix("follower_")
        issues = _official_issues(side)
        for suffix in SUFFIXES:
            link_name = f"{robot}_{suffix}"
            prim_path = f"{root_path}/{link_name}"
            if prim_path not in issues:
                raise RuntimeError(f"official finding missing: {prim_path}")
            source = source_records[link_name]
            findings.append(
                {
                    "follower": robot,
                    "suffix": suffix,
                    "prim_path": prim_path,
                    "classification": source["classification"],
                    "source_urdf": source,
                    "usd": _prim_record(stage, prim_path),
                    "official_finding": issues[prim_path],
                    "fixed_group_coverage": (
                        {
                            "applies": True,
                            "owner_link_suffix": "gripper_link",
                            "owner_collider_path": next(
                                path
                                for path in group_coverage["owner_collider_paths"]
                                if f"/{robot}/" in path
                            ),
                            "source_rule": group_coverage["rule"],
                            "interpretation": (
                                "The source link is physical, but the isolated CAD "
                                "diagnostic intentionally assigns the supplier fixed-group "
                                "compound collider to gripper_link and deactivates the "
                                "baseline gripper_bar collider to avoid duplicate coverage."
                            ),
                        }
                        if suffix == "gripper_bar_link"
                        else {"applies": False}
                    ),
                    "candidate_action": (
                        "TEST_COLLIDER_OWNERSHIP_OR_FIXED_BODY_TOPOLOGY_IN_ISOLATION"
                        if suffix == "gripper_bar_link"
                        else "TEST_REMOVE_RIGID_BODY_API_WITH_ARTICULATION_REGRESSION"
                    ),
                    "usd_modified": False,
                }
            )
        after = _sha256(candidate)
        candidate_hashes[robot] = {
            "absolute_path": str(candidate),
            "sha256_before": before,
            "sha256_after": after,
            "modified": before != after,
        }
    frozen_after = _sha256(frozen)
    counts = dict(sorted(Counter(item["classification"] for item in findings).items()))
    all_literal = all(
        item["usd"]["has_rigid_body_api"]
        and item["usd"]["descendant_collision_count"] == 0
        for item in findings
    )
    return {
        "schema_version": 1,
        "status": "PASS" if len(findings) == 8 and all_literal else "PARTIAL",
        "finding_count": len(findings),
        "classification_counts": counts,
        "stage": {
            "absolute_path": str(frozen),
            "sha256_before": frozen_before,
            "sha256_after": frozen_after,
            "modified": frozen_before != frozen_after,
        },
        "candidate_stages": candidate_hashes,
        "findings": findings,
        "fixed_group_evidence": {
            "report": str(COLLIDER_STAGE_REPORT.resolve(strict=True)),
            "report_sha256": _sha256(COLLIDER_STAGE_REPORT),
            "readback": group_coverage,
        },
        "local_rule_source": {
            "absolute_path": str(RULE_SOURCE.resolve(strict=True)),
            "sha256": _sha256(RULE_SOURCE.resolve(strict=True)),
            "class": "RigidBodyHasCollider",
        },
        "decision": {
            "invent_collider_without_geometry_allowed": False,
            "remove_rigid_body_api_without_regression_allowed": False,
            "gripper_bar_duplicate_collider_allowed": False,
            "next_step": "SINGLE_VARIABLE_ISOLATED_CANDIDATES",
        },
        "final_or_default_asset_modified": False,
        "task8": "NOT_RUN",
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ALOHA1 Task 7 helper-link collider audit",
        "",
        f"- Status: `{report['status']}`",
        f"- Findings: `{report['finding_count']}`",
        f"- Classes: `{json.dumps(report['classification_counts'], sort_keys=True)}`",
        "- Frozen/default assets modified: `false`",
        "- Task 8: `NOT_RUN`",
        "",
        "| Follower | Link | Source geometry | Class | Existing active collider | Fixed-group coverage |",
        "|---|---|---:|---|---:|---:|",
    ]
    lines.extend(
        "| {follower} | `{suffix}` | V={visual}/C={collision} | `{kind}` | {active} | {group} |".format(
            follower=item["follower"],
            suffix=item["suffix"],
            visual=item["source_urdf"]["visual_count"],
            collision=item["source_urdf"]["collision_count"],
            kind=item["classification"],
            active=item["usd"]["descendant_collision_count"],
            group=item["fixed_group_coverage"]["applies"],
        )
        for item in report["findings"]
    )
    lines.extend(
        [
            "",
            "The six empty fixed-frame links have no source visual/collision geometry; "
            "inventing colliders is prohibited. The two gripper-bar links are different: "
            "the pinned URDF contains a real bar mesh and collider, while the CAD diagnostic "
            "deactivates that collider because supplier Part__Feature006 is already authored "
            "as one compound collider for the fixed gripper+bar group. The literal validator "
            "finding is therefore reproduced, but the correct repair cannot be chosen until "
            "collider ownership and fixed-body topology are tested separately.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    report = build_report()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    OUTPUT_MD.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {"status": report["status"], "output": str(OUTPUT_JSON.resolve())},
            sort_keys=True,
        )
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
