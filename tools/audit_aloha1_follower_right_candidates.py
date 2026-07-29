#!/usr/bin/env python3
"""Read-only audit of bounded follower-right USD candidates.

Run with the USD Python libraries bundled with the local Isaac Sim 5.1
installation.  This tool never authors or saves a Stage.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pxr import Usd
from pxr import UsdGeom
from pxr import UsdPhysics

from tools.aloha1_mapping.follower_right_stage_audit import classify_candidate
from tools.aloha1_mapping.follower_right_stage_audit import summarize_audit

ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = [
    ROOT / "assets/Trossen/ALOHA1",
    *sorted((ROOT / "local_eval_assets").glob("aloha_isaac*")),
    ROOT / "aloha_isaac_rebuild",
]
OUTPUT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_follower_right_stage_audit.json"
)
OUTPUT_MD = OUTPUT.with_suffix(".md")
APPROVED_STAGE = (
    ROOT / "local_eval_assets/aloha_isaac_assets/aloha_viperx.usd"
)
TASK5_ASSET_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_asset.json"
)
CAD_IDENTITY_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_follower_left_right_cad_identity.json"
)
EXPECTED_APPROVED_STAGE_HASH = (
    "b24afe3678155654892c69517fc58ecd970108d68cec56b02dc6fdcb8bf4493e"
)
SUPPLIER_HASHES = {
    "left": (
        "c6710d0fe5b2030a32722d9df5c0b553c771c9d61d92b8ddaec36c94c5963488"
    ),
    "right": (
        "b0979c5d55fee448dab512dc75b1251bab17d94892decd01de9a6e76c01482d1"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_named_values(value: Any, name: str) -> list[str]:
    found: list[str] = []
    if hasattr(value, "items"):
        for key, nested in value.items():
            if str(key) == name:
                found.append(str(nested))
            else:
                found.extend(_collect_named_values(nested, name))
    elif isinstance(value, list | tuple):
        for nested in value:
            found.extend(_collect_named_values(nested, name))
    return found


def _ancestor_source_hashes(prim: Usd.Prim) -> set[str]:
    hashes: set[str] = set()
    current = prim
    while current and current.IsValid() and not current.IsPseudoRoot():
        hashes.update(
            _collect_named_values(
                current.GetCustomData(),
                "sourceObjSha256",
            )
        )
        current = current.GetParent()
    return hashes


def _is_joint(prim: Usd.Prim) -> bool:
    return (
        prim.IsA(UsdPhysics.RevoluteJoint)
        or prim.IsA(UsdPhysics.PrismaticJoint)
        or prim.IsA(UsdPhysics.FixedJoint)
    )


def _probe(path: Path) -> dict[str, Any]:
    before = _sha256(path)
    open_error = None
    try:
        stage = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
    except Exception as error:  # USD raises Tf.ErrorException for bad layers.
        stage = None
        open_error = f"{type(error).__name__}: {error}"
    if stage is None:
        return {
            "absolute_path": str(path),
            "relative_path": str(path.relative_to(ROOT)),
            "sha256_before": before,
            "sha256_after": _sha256(path),
            "open_status": "FAIL",
            "open_error": open_error or "Usd.Stage.Open returned None",
            "articulation_roots": [],
            "right_prim_count": 0,
            "finger_source_hashes": [],
            "finger_mesh_signatures": [],
            "used_layers": [],
        }

    articulation_roots: list[str] = []
    right_prims: list[str] = []
    joints: list[str] = []
    source_hashes: set[str] = set()
    mesh_signatures: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        lower = prim_path.lower()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(prim_path)
        if _is_joint(prim):
            joints.append(prim_path)
        if (
            "follower_right" in lower
            or "vx300s_right" in lower
            or "right_follower" in lower
        ):
            right_prims.append(prim_path)
        if prim.IsA(UsdGeom.Mesh) and (
            "finger" in lower or "gripper" in lower
        ):
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get() or []
            faces = mesh.GetFaceVertexCountsAttr().Get() or []
            mesh_hashes = _ancestor_source_hashes(prim)
            source_hashes.update(mesh_hashes)
            mesh_signatures.append(
                {
                    "path": prim_path,
                    "point_count": len(points),
                    "face_count": len(faces),
                    "source_obj_sha256": sorted(mesh_hashes),
                    "prim_stack_layers": sorted(
                        {
                            spec.layer.realPath or spec.layer.identifier
                            for spec in prim.GetPrimStack()
                        }
                    ),
                }
            )

    default_prim = stage.GetDefaultPrim()
    return {
        "absolute_path": str(path),
        "relative_path": str(path.relative_to(ROOT)),
        "sha256_before": before,
        "sha256_after": _sha256(path),
        "open_status": "PASS",
        "default_prim": (
            str(default_prim.GetPath()) if default_prim.IsValid() else None
        ),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "articulation_roots": sorted(articulation_roots),
        "joint_count": len(joints),
        "joints": sorted(joints),
        "right_prim_count": len(right_prims),
        "finger_source_hashes": sorted(source_hashes),
        "finger_mesh_signatures": mesh_signatures,
        "used_layers": sorted(
            str(Path(layer.realPath).resolve())
            for layer in stage.GetUsedLayers()
            if layer.realPath
        ),
    }


def _discover() -> list[Path]:
    suffixes = {".usd", ".usda", ".usdc"}
    paths: set[Path] = set()
    for search_root in SEARCH_ROOTS:
        if not search_root.exists():
            continue
        paths.update(
            path.resolve()
            for path in search_root.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes
        )
    return sorted(paths)


def _candidate_probe(record: dict[str, Any]) -> bool:
    lower_path = record["absolute_path"].lower()
    return bool(
        record["right_prim_count"]
        or any(
            "follower_right" in root.lower()
            or "vx300s_right" in root.lower()
            for root in record["articulation_roots"]
        )
        or "rejected_phantom_right_branch" in lower_path
        or "failed_attempt" in lower_path
    )


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ALOHA Viper follower_right Stage audit",
        "",
        f"- Status: `{report['status']}`",
        f"- Candidate Stages: `{report['candidate_count']}`",
        f"- Eligible current supplier-CAD Stages: `{report['eligible_count']}`",
        f"- CAD availability: `{report['cad_availability']}`",
        f"- Next action: `{report['next_action']}`",
        f"- Protected inputs unchanged: `{report['protected_inputs_unchanged']}`",
        "- Scope: read-only, local Isaac Sim 5.1 USD composition evidence.",
        "- Task 8: `NOT_RUN`",
        "",
        "## Result",
        "",
    ]
    if report["eligible_count"]:
        lines.append(
            "At least one independent follower_right Stage contains both "
            "handed supplier-v2 mesh source hashes."
        )
    else:
        lines.extend(
            [
                "No eligible follower_right Stage was found. Existing right "
                "candidates are generic, historical, ALOHA2/legacy, rejected "
                "phantom branches, or lack current supplier-CAD provenance.",
                "",
                "This means the robot-local right Stage has not yet been "
                "generated and validated. It does not mean the supplier CAD "
                "lacks a right-arm product: the CAD identity audit verifies "
                "one reusable ViperX robot product for both followers.",
            ]
        )
    lines.extend(["", "## Classification counts", ""])
    for name, count in report["classification_counts"].items():
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "## HARD_BLOCKER", ""])
    lines.extend(f"- `{blocker}`" for blocker in report["hard_blockers"])
    lines.extend(
        [
            "",
            "The approved review Stage remains immutable and contains no "
            "`/workcell/vx300s_right`; that absence is scoped only to the "
            "approved left review Stage. No rejected or historical asset was "
            "promoted. The workcell placement transform remains a separate "
            "HARD_BLOCKER from robot-local asset generation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    approved_before = _sha256(APPROVED_STAGE.resolve(strict=True))
    task5_report = json.loads(
        TASK5_ASSET_REPORT.read_text(encoding="utf-8")
    )
    cad_identity_report = json.loads(
        CAD_IDENTITY_REPORT.read_text(encoding="utf-8")
    )
    probed = [_probe(path) for path in _discover()]
    candidates = [
        classify_candidate(record, SUPPLIER_HASHES)
        for record in probed
        if _candidate_probe(record)
    ]
    summary = summarize_audit(
        candidates,
        cad_identity_classification=cad_identity_report["classification"],
        workcell_placement_verified=cad_identity_report[
            "workcell_placement_verified"
        ],
    )
    approved_after = _sha256(APPROVED_STAGE)
    protected_unchanged = (
        approved_before == approved_after == EXPECTED_APPROVED_STAGE_HASH
        and all(
            item["sha256_before"] == item["sha256_after"]
            for item in candidates
        )
    )
    hard_blockers = []
    if not task5_report["source_follower_presence"]["follower_right"]:
        hard_blockers.append(
            "HARD_BLOCKER_APPROVED_STAGE_MISSING_FOLLOWER_RIGHT"
        )
    hard_blockers.extend(summary["hard_blockers"])
    blocker_definitions = {
        "HARD_BLOCKER_APPROVED_STAGE_MISSING_FOLLOWER_RIGHT": {
            "meaning": (
                "The user-approved follower_left review Stage does not "
                "contain a complete follower_right articulation."
            ),
            "does_not_mean": (
                "Supplier CAD lacks the right-arm robot product."
            ),
            "scope": "APPROVED_LEFT_REVIEW_STAGE_ONLY",
        },
        **summary["blocker_definitions"],
    }
    report = {
        "schema_version": 1,
        "scope": "CURRENT_SUPPLIER_CAD_FOLLOWER_RIGHT_STAGE_READ_ONLY_AUDIT",
        **summary,
        "status": (
            "FAIL"
            if not protected_unchanged
            else summary["status"]
        ),
        "hard_blockers": hard_blockers,
        "blocker_definitions": blocker_definitions,
        "approved_stage": {
            "absolute_path": str(APPROVED_STAGE.resolve()),
            "sha256_before": approved_before,
            "sha256_after": approved_after,
            "expected_sha256": EXPECTED_APPROVED_STAGE_HASH,
            "follower_right_present": task5_report[
                "source_follower_presence"
            ]["follower_right"],
            "absence_scope": "APPROVED_LEFT_REVIEW_STAGE_ONLY",
        },
        "cad_identity_report": {
            "absolute_path": str(CAD_IDENTITY_REPORT.resolve()),
            "classification": cad_identity_report["classification"],
            "robot_local_identity_verified": cad_identity_report[
                "robot_local_identity_verified"
            ],
            "workcell_placement_verified": cad_identity_report[
                "workcell_placement_verified"
            ],
        },
        "supplier_mesh_source_hashes": SUPPLIER_HASHES,
        "search_roots": [str(path.resolve()) for path in SEARCH_ROOTS],
        "discovered_usd_count": len(probed),
        "protected_inputs_unchanged": protected_unchanged,
        "eligibility_rule": (
            "both handed supplier-v2 sourceObjSha256 values + exactly one "
            "independent follower_right/vx300s_right Articulation Root + "
            "non-rejected, non-ALOHA2 provenance"
        ),
        "candidates": candidates,
        "task8": "NOT_RUN",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    OUTPUT_MD.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "discovered_usd_count": len(probed),
                "candidate_count": len(candidates),
                "eligible_count": report["eligible_count"],
                "output": str(OUTPUT.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
