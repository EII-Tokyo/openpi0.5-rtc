#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
from typing import Any

from tools.aloha1_mapping.task8_optimization import build_inventory_summary
from tools.aloha1_mapping.task8_optimization import build_protected_signature
from tools.aloha1_mapping.task8_optimization import failure_evidence_contract
from tools.aloha1_mapping.task8_optimization import rank_optimization_opportunities

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE = ROOT / (
    "assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/"
    "aloha1_cad_derived_full_body_collider_gripper_decomposition_"
    "tabletop_zero_z_up_meters_diagnostic.usda"
)
DEFAULT_FINGER_LIMIT_LAYER = ROOT / (
    "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "finger_limit_pair_collision_candidate/1.0/configuration/"
    "finger_source_limits.usda"
)
DEFAULT_OUTPUT = (
    ROOT / "reports/aloha1_mapping/aloha1_task8_baseline_inventory.json"
)
DEFAULT_MARKDOWN = (
    ROOT / "reports/aloha1_mapping/aloha1_task8_baseline_inventory.md"
)
EXPECTED_STAGE_SHA256 = (
    "327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9"
)
EXPECTED_FINGER_LIMIT_SHA256 = (
    "2547e6fb374c213b5c6c54f200c7ced37605ab0e1a11735d0a32c0a231fd260f"
)


def start_usd_runtime_if_needed(
    *,
    pxr_available: bool | None = None,
    app_factory: Any | None = None,
) -> Any | None:
    """Start an isolated headless Kit runtime when USD bindings are not loaded."""

    if pxr_available is None:
        pxr_available = importlib.util.find_spec("pxr") is not None
    if pxr_available:
        return None
    if app_factory is None:
        from isaacsim import SimulationApp

        app_factory = SimulationApp
    return app_factory({"headless": True, "create_new_stage": False})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "+Infinity" if value > 0 else "-Infinity"
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    try:
        return [_json_value(item) for item in value]
    except TypeError:
        return str(value)


def _geometry_signature(mesh: Any) -> str:
    payload = {
        "points": [
            [round(float(component), 9) for component in point]
            for point in (mesh.GetPointsAttr().Get() or [])
        ],
        "face_vertex_counts": [
            int(value) for value in (mesh.GetFaceVertexCountsAttr().Get() or [])
        ],
        "face_vertex_indices": [
            int(value) for value in (mesh.GetFaceVertexIndicesAttr().Get() or [])
        ],
        "subdivision_scheme": str(mesh.GetSubdivisionSchemeAttr().Get() or ""),
        "orientation": str(mesh.GetOrientationAttr().Get() or ""),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prim_relevant_properties(prim: Any) -> dict[str, Any]:
    prefixes = ("physics:", "physx", "drive:", "xformOp:")
    attributes = {
        attribute.GetName(): _json_value(attribute.Get())
        for attribute in prim.GetAttributes()
        if attribute.GetName().startswith(prefixes)
    }
    relationships = {
        relation.GetName(): sorted(str(path) for path in relation.GetTargets())
        for relation in prim.GetRelationships()
        if relation.GetName().startswith(("physics:", "physx"))
    }
    return {
        "attributes": attributes,
        "relationships": relationships,
        "applied_schemas": sorted(str(item) for item in prim.GetAppliedSchemas()),
    }


def _material_record(material_prim: Any, usd_shade: Any) -> dict[str, Any]:
    shaders = []
    for child in material_prim.GetChildren():
        if not child.IsA(usd_shade.Shader):
            continue
        shader = usd_shade.Shader(child)
        shaders.append(
            {
                "path": str(child.GetPath()),
                "shader_id": _json_value(shader.GetIdAttr().Get()),
                "inputs": {
                    str(item.GetBaseName()): _json_value(item.Get())
                    for item in sorted(shader.GetInputs(), key=lambda value: value.GetBaseName())
                },
            }
        )
    canonical = [
        {
            "shader_id": shader["shader_id"],
            "inputs": shader["inputs"],
        }
        for shader in shaders
    ]
    signature = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "path": str(material_prim.GetPath()),
        "active": bool(material_prim.IsActive()),
        "shaders": shaders,
        "material_signature": signature,
    }


def canonical_layer_identifier(identifier: str) -> str:
    """Remove the process-local pointer from anonymous session-layer IDs."""

    return re.sub(r"^anon:0x[0-9A-Fa-f]+:", "anon:<session>:", identifier)


def _layer_record(layer: Any) -> dict[str, Any]:
    real_path = Path(layer.realPath) if layer.realPath else None
    return {
        "identifier": canonical_layer_identifier(str(layer.identifier)),
        "real_path": str(real_path.resolve()) if real_path and real_path.exists() else None,
        "sha256": _sha256(real_path) if real_path and real_path.is_file() else None,
        "sub_layer_paths": [str(item) for item in layer.subLayerPaths],
    }


def audit(stage_path: Path, finger_limit_layer: Path) -> dict[str, Any]:
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics
    from pxr import UsdShade
    from pxr import UsdUtils

    stage_path = stage_path.resolve(strict=True)
    finger_limit_layer = finger_limit_layer.resolve(strict=True)
    stage_hash = _sha256(stage_path)
    finger_hash = _sha256(finger_limit_layer)
    stage = Usd.Stage.Open(str(stage_path), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Usd.Stage.Open failed: {stage_path}")

    prims = list(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()))
    prim_type_counts = Counter(str(prim.GetTypeName() or "<typeless>") for prim in prims)
    composition_records = [
        {
            "path": str(prim.GetPath()),
            "has_authored_references": bool(prim.HasAuthoredReferences()),
            "has_authored_payloads": bool(prim.HasAuthoredPayloads()),
            "is_instance": bool(prim.IsInstance()),
            "is_instance_proxy": bool(prim.IsInstanceProxy()),
            "is_instanceable": bool(prim.IsInstanceable()),
        }
        for prim in prims
    ]

    mesh_records = []
    materials = []
    articulations = []
    joints = []
    rigid_bodies = []
    colliders = []
    for prim in prims:
        if prim.IsA(UsdShade.Material):
            materials.append(_material_record(prim, UsdShade))
        relevant = _prim_relevant_properties(prim)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulations.append({"path": str(prim.GetPath()), **relevant})
        if prim.IsA(UsdPhysics.Joint):
            joints.append(
                {
                    "path": str(prim.GetPath()),
                    "type": str(prim.GetTypeName()),
                    **relevant,
                }
            )
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append({"path": str(prim.GetPath()), **relevant})
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            geometry_signature = _geometry_signature(mesh)
            record = {
                "path": str(prim.GetPath()),
                "geometry_signature": geometry_signature,
                "is_collision": bool(prim.HasAPI(UsdPhysics.CollisionAPI)),
                "point_count": len(mesh.GetPointsAttr().Get() or []),
                "face_count": len(mesh.GetFaceVertexCountsAttr().Get() or []),
                "index_count": len(mesh.GetFaceVertexIndicesAttr().Get() or []),
                "material_path": str(material.GetPath()) if material else None,
                "is_instance_proxy": bool(prim.IsInstanceProxy()),
                "is_instanceable": bool(prim.IsInstanceable()),
            }
            mesh_records.append(record)
            if record["is_collision"]:
                colliders.append(
                    {
                        "path": record["path"],
                        "geometry": geometry_signature,
                        **relevant,
                    }
                )
        elif prim.HasAPI(UsdPhysics.CollisionAPI):
            colliders.append(
                {
                    "path": str(prim.GetPath()),
                    "geometry": str(prim.GetTypeName()),
                    **relevant,
                }
            )

    summary = build_inventory_summary(
        mesh_records=mesh_records,
        material_records=materials,
        prim_type_counts=prim_type_counts,
        composition_records=composition_records,
    )
    protected = {
        "articulations": sorted(articulations, key=lambda item: item["path"]),
        "joints": sorted(joints, key=lambda item: item["path"]),
        "rigid_bodies": sorted(rigid_bodies, key=lambda item: item["path"]),
        "colliders": sorted(colliders, key=lambda item: item["path"]),
        "visuals": sorted(
            (record for record in mesh_records if not record["is_collision"]),
            key=lambda item: item["path"],
        ),
    }
    dependency_layers, dependency_assets, unresolved_assets = (
        UsdUtils.ComputeAllDependencies(str(stage_path))
    )
    default_prim = stage.GetDefaultPrim()
    status = (
        "PASS"
        if stage_hash == EXPECTED_STAGE_SHA256
        and finger_hash == EXPECTED_FINGER_LIMIT_SHA256
        and default_prim
        and str(default_prim.GetPath()) == "/World"
        else "FAIL"
    )
    return {
        "schema_version": 1,
        "status": status,
        "classification": "TASK8_BASELINE_FROZEN_INVENTORY",
        "authorization": {
            "task7": "PARTIAL_ACCEPTED_FOR_TASK8",
            "task8": "AUTHORIZED_IN_PROGRESS",
            "final_default_asset_modified": False,
        },
        "usd_runtime": {"usd_version": list(Usd.GetVersion())},
        "stage": {
            "absolute_path": str(stage_path),
            "sha256": stage_hash,
            "expected_sha256": EXPECTED_STAGE_SHA256,
            "default_prim": str(default_prim.GetPath()) if default_prim else None,
            "root_layer": _layer_record(stage.GetRootLayer()),
            "used_layers": sorted(
                (_layer_record(layer) for layer in stage.GetUsedLayers()),
                key=lambda item: item["identifier"],
            ),
        },
        "finger_limit_layer": {
            "absolute_path": str(finger_limit_layer),
            "sha256": finger_hash,
            "expected_sha256": EXPECTED_FINGER_LIMIT_SHA256,
            "promotion_status": "CREATED_NOT_PROMOTED",
        },
        "dependencies": {
            "layer_count": len(dependency_layers),
            "asset_count": len(dependency_assets),
            "unresolved": sorted(str(item) for item in unresolved_assets),
        },
        "summary": summary,
        "meshes": sorted(mesh_records, key=lambda item: item["path"]),
        "materials": sorted(materials, key=lambda item: item["path"]),
        "composition": sorted(composition_records, key=lambda item: item["path"]),
        "protected_inventory": protected,
        "protected_physics_signature": build_protected_signature(protected),
        "opportunities": rank_optimization_opportunities(
            summary,
            known_hydra_instance_regression=True,
        ),
        "failure_evidence_contract": failure_evidence_contract(reproducible=True),
    }


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# ALOHA1 Task 8 baseline inventory",
        "",
        f"Status: `{report['status']}`",
        "",
        "The user-authorized boundary is `Task 7 = PARTIAL_ACCEPTED_FOR_TASK8`; "
        "this inventory does not promote or modify a final/default asset.",
        "",
        "## Frozen inputs",
        "",
        f"- Stage: `{report['stage']['absolute_path']}`",
        f"- Stage SHA-256: `{report['stage']['sha256']}`",
        f"- finger-limit layer: `{report['finger_limit_layer']['absolute_path']}`",
        f"- finger-limit SHA-256: `{report['finger_limit_layer']['sha256']}`",
        "",
        "## Inventory",
        "",
        f"- composed prims (including instance proxies): {summary['prim_count']}",
        f"- meshes: {summary['mesh_count']} "
        f"({summary['visual_mesh_count']} visual, {summary['collision_mesh_count']} collision)",
        f"- points / faces: {summary['point_count']} / {summary['face_count']}",
        f"- materials: {summary['material_count']}",
        f"- instanceable prims: {summary['instanceable_prim_count']}",
        f"- payload prims: {summary['payload_prim_count']}",
        f"- repeated visual geometry groups: {summary['repeated_visual_geometry_groups']}",
        f"- repeated collision geometry groups: {summary['repeated_collision_geometry_groups']}",
        "",
        "## Ranked opportunities",
        "",
    ]
    lines.extend(
        f"- `{item['id']}`: `{item['decision']}`; risk `{item['risk']}`"
        for item in report["opportunities"]
    )
    lines.extend(
        [
            "",
            "The first candidate is limited to repeated visual geometry. Collision "
            "deduplication remains deferred because it changes physics composition. "
            "Existing payload/instanceable authoring and the local Hydra protoPath "
            "failure are explicit constraints.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--finger-limit-layer", type=Path, default=DEFAULT_FINGER_LIMIT_LAYER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = start_usd_runtime_if_needed()
    try:
        report = audit(args.stage, args.finger_limit_layer)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(_markdown(report), encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "stage_sha256": report["stage"]["sha256"],
                    "mesh_count": report["summary"]["mesh_count"],
                    "visual_mesh_count": report["summary"]["visual_mesh_count"],
                    "collision_mesh_count": report["summary"]["collision_mesh_count"],
                    "output": str(args.output.resolve()),
                },
                sort_keys=True,
            )
        )
        return 0 if report["status"] == "PASS" else 1
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
