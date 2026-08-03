#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback
from typing import Any

from tools.aloha1_mapping.task8_optimization import build_material_dedup_plan
from tools.audit_aloha1_task8_baseline import audit
from tools.audit_aloha1_task8_baseline import start_usd_runtime_if_needed

ROOT = Path(__file__).resolve().parents[1]
BASELINE_REPORT = ROOT / "reports/aloha1_mapping/aloha1_task8_baseline_inventory.json"
DEFAULT_OUTPUT_DIR = ROOT / (
    "assets/Trossen/ALOHA1/1.0/diagnostics/task8_visual_material_dedup/1.0"
)
DEFAULT_REPORT = ROOT / "reports/aloha1_mapping/aloha1_task8_visual_material_candidate.json"
DEFAULT_MARKDOWN = ROOT / "reports/aloha1_mapping/aloha1_task8_visual_material_candidate.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _bound_visual_material_count(inventory: dict[str, Any]) -> int:
    summary = inventory["summary"]
    if "distinct_bound_visual_material_count" in summary:
        return int(summary["distinct_bound_visual_material_count"])
    return len(
        {
            str(mesh["material_path"])
            for mesh in inventory["meshes"]
            if not bool(mesh["is_collision"]) and mesh.get("material_path")
        }
    )


def _markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# ALOHA1 Task 8 visual-material candidate",
            "",
            f"Status: `{report['status']}`",
            "",
            f"- Candidate Stage: `{report['candidate']['root_stage']}`",
            f"- Candidate SHA-256: `{report['candidate']['root_sha256']}`",
            f"- Shared bindings authored: {report['candidate']['binding_count']}",
            f"- baseline/candidate effective visual materials: "
            f"{report['comparison']['baseline_bound_visual_material_count']} / "
            f"{report['comparison']['candidate_bound_visual_material_count']}",
            f"- protected physics signature unchanged: "
            f"`{report['comparison']['protected_physics_signature_unchanged']}`",
            f"- instanceable prim count unchanged: "
            f"`{report['comparison']['instanceable_prim_count_unchanged']}`",
            "",
            "This isolated candidate changes only visual material binding at robot visual-instance "
            "roots. It does not change mesh geometry, collision, mass, joints, drives, timestep or "
            "solver settings and is not promoted to a final/default asset.",
            "",
        ]
    )


def build(output_dir: Path) -> dict[str, Any]:
    baseline = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    baseline_stage = Path(baseline["stage"]["absolute_path"]).resolve(strict=True)
    finger_limit = Path(baseline["finger_limit_layer"]["absolute_path"]).resolve(strict=True)
    source_hash_before = _sha256(baseline_stage)
    if source_hash_before != baseline["stage"]["sha256"]:
        raise RuntimeError("frozen baseline Stage hash changed")

    plan = build_material_dedup_plan(
        mesh_records=baseline["meshes"],
        duplicate_material_groups=baseline["summary"]["duplicate_materials"],
    )
    if not plan:
        raise RuntimeError("no editable shared-material bindings were found")
    print(f"TASK8_CANDIDATE_PLAN binding_count={len(plan)}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    configuration_dir = output_dir / "configuration"
    configuration_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = configuration_dir / "shared_visual_material_bindings.usda"
    root_path = output_dir / "aloha1_task8_visual_material_dedup_candidate.usda"

    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdShade

    overlay = Sdf.Layer.CreateNew(str(overlay_path))
    if overlay is None:
        raise RuntimeError(f"failed to create {overlay_path}")
    overlay.Save()
    root_layer = Sdf.Layer.CreateNew(str(root_path))
    if root_layer is None:
        raise RuntimeError(f"failed to create {root_path}")
    root_layer.defaultPrim = "World"
    root_layer.subLayerPaths = [
        os.path.relpath(overlay_path, root_path.parent),
        os.path.relpath(baseline_stage, root_path.parent),
    ]
    root_layer.Save()

    stage = Usd.Stage.Open(str(root_path), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"failed to open candidate Stage {root_path}")
    stage.SetEditTarget(overlay)
    for item in plan:
        visual_root = stage.GetPrimAtPath(item["visual_root"])
        material_prim = stage.GetPrimAtPath(item["canonical_material"])
        if not visual_root or not material_prim:
            raise RuntimeError(f"missing binding prims for {item['visual_root']}")
        material = UsdShade.Material(material_prim)
        UsdShade.MaterialBindingAPI.Apply(visual_root).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
    overlay.Save()
    print("TASK8_CANDIDATE_OVERLAY_SAVED", flush=True)
    stage = Usd.Stage.Open(str(root_path), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"failed to reopen candidate Stage {root_path}")
    print("TASK8_CANDIDATE_REOPENED", flush=True)

    readbacks = []
    for item in plan:
        mesh_prim = stage.GetPrimAtPath(item["representative_mesh"])
        material, relation = UsdShade.MaterialBindingAPI(mesh_prim).ComputeBoundMaterial()
        readbacks.append(
            {
                **item,
                "effective_material": str(material.GetPath()) if material else None,
                "binding_relationship": str(relation.GetPath()) if relation else None,
                "pass": bool(material)
                and str(material.GetPath()) == item["canonical_material"],
            }
        )

    candidate_inventory = audit(root_path, finger_limit)
    print("TASK8_CANDIDATE_INVENTORY_COMPLETE", flush=True)
    source_hash_after = _sha256(baseline_stage)
    comparison = {
        "source_stage_unchanged": source_hash_before == source_hash_after,
        "protected_physics_signature_unchanged": (
            baseline["protected_physics_signature"]
            == candidate_inventory["protected_physics_signature"]
        ),
        "instanceable_prim_count_unchanged": (
            baseline["summary"]["instanceable_prim_count"]
            == candidate_inventory["summary"]["instanceable_prim_count"]
        ),
        "baseline_bound_visual_material_count": _bound_visual_material_count(baseline),
        "candidate_bound_visual_material_count": _bound_visual_material_count(
            candidate_inventory
        ),
    }
    print("TASK8_CANDIDATE_COMPARISON_COMPLETE", flush=True)
    pass_gate = (
        all(item["pass"] for item in readbacks)
        and comparison["source_stage_unchanged"]
        and comparison["protected_physics_signature_unchanged"]
        and comparison["instanceable_prim_count_unchanged"]
        and comparison["candidate_bound_visual_material_count"]
        < comparison["baseline_bound_visual_material_count"]
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS_STATIC_CANDIDATE_NOT_PROMOTED" if pass_gate else "FAIL_STATIC_CANDIDATE",
        "classification": "TASK8_VISUAL_MATERIAL_DEDUP_ISOLATED_CANDIDATE",
        "baseline": {
            "stage": str(baseline_stage),
            "sha256": source_hash_before,
            "inventory_sha256": _sha256(BASELINE_REPORT),
        },
        "candidate": {
            "root_stage": str(root_path.resolve()),
            "root_sha256": _sha256(root_path),
            "configuration_layer": str(overlay_path.resolve()),
            "configuration_sha256": _sha256(overlay_path),
            "binding_count": len(plan),
            "bindings": readbacks,
            "inventory": candidate_inventory,
        },
        "comparison": comparison,
        "boundaries": {
            "visual_only": True,
            "collision_or_physics_changed": False,
            "final_or_default_asset_modified": False,
            "candidate_promoted": False,
            "approximate_simulation_policy": "ALLOWED",
        },
    }
    report["deterministic_signature"] = hashlib.sha256(
        _canonical_json(report).encode()
    ).hexdigest()
    print("TASK8_CANDIDATE_SIGNATURE_COMPLETE", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = start_usd_runtime_if_needed()
    result = 1
    try:
        report = build(args.output_dir.resolve())
        print(f"TASK8_CANDIDATE_BUILD_STATUS {report['status']}", flush=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        args.markdown.write_text(_markdown(report), encoding="utf-8")
        print("TASK8_CANDIDATE_REPORTS_WRITTEN", flush=True)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "candidate": report["candidate"]["root_stage"],
                    "binding_count": report["candidate"]["binding_count"],
                    "output": str(args.report.resolve()),
                },
                sort_keys=True,
            )
        )
        result = 0 if report["status"].startswith("PASS") else 1
    except Exception:
        print("TASK8_CANDIDATE_EXCEPTION", flush=True)
        traceback.print_exc()
    finally:
        if app is not None:
            app.close()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
