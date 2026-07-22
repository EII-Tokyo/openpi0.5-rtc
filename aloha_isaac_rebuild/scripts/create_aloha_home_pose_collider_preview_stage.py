#!/usr/bin/env python3
"""Create a home-pose visual/collider candidate preview for clean ALOHA1.

This stage is intentionally still pre-physics.  It exists to check that
candidate collider resources line up with the visual assembly after applying
the verified ALOHA1 home pose.  It does not enable collisions, author rigid
bodies, create joints, run the timeline, or claim contact readiness.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_CONFIG = Path("aloha_isaac_rebuild/configs/physical_reconstruction/stationary_style_rebuild.yaml")
DEFAULT_VALIDATION_JSON = Path(
    "aloha_isaac_rebuild/artifacts/validation/a16_home_pose_collider_preview_audit.json"
)

FORBIDDEN_PHYSICS_APIS = {
    "PhysicsRigidBodyAPI",
    "PhysicsCollisionAPI",
    "PhysicsMassAPI",
    "PhysicsArticulationRootAPI",
}
FORBIDDEN_RUNTIME_TYPES = {
    "PhysicsScene",
    "PhysicsFixedJoint",
    "PhysicsRevoluteJoint",
    "PhysicsPrismaticJoint",
    "Camera",
    "RenderProduct",
}


def _start_isaac_headless():
    from isaacsim import SimulationApp

    return SimulationApp({"headless": True})


def _bbox_dict(stage, path: str) -> dict:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return {"path": path, "valid": False, "reason": "missing_prim"}
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.guide,
        ],
    )
    try:
        aligned = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    except Exception as exc:  # pragma: no cover - defensive against malformed referenced assets
        return {"path": path, "valid": False, "reason": f"bbox_error:{exc}"}
    if aligned.IsEmpty():
        return {"path": path, "valid": False, "reason": "empty_bbox"}
    size = aligned.GetSize()
    center = (aligned.GetMin() + aligned.GetMax()) * 0.5
    valid = any(float(value) > 0.0 for value in size)
    return {
        "path": path,
        "valid": bool(valid),
        "center_m": [float(value) for value in center],
        "size_m": [float(value) for value in size],
    }


def _bbox_residual(visual: dict, collider: dict) -> dict:
    if not visual.get("valid") or not collider.get("valid"):
        return {
            "valid": False,
            "center_residual_m": None,
            "size_axis_abs_residual_m": None,
            "max_size_axis_abs_residual_m": None,
            "max_size_axis_ratio": None,
        }
    center_residual = sum(
        (a - b) ** 2 for a, b in zip(visual["center_m"], collider["center_m"], strict=True)
    ) ** 0.5
    size_abs = [
        abs(a - b) for a, b in zip(visual["size_m"], collider["size_m"], strict=True)
    ]
    ratios = []
    for visual_size, collider_size in zip(visual["size_m"], collider["size_m"], strict=True):
        denom = max(abs(visual_size), 1.0e-9)
        ratios.append(abs(collider_size - visual_size) / denom)
    return {
        "valid": True,
        "center_residual_m": float(center_residual),
        "size_axis_abs_residual_m": [float(value) for value in size_abs],
        "max_size_axis_abs_residual_m": float(max(size_abs)),
        "max_size_axis_ratio": float(max(ratios)),
    }


def _stage_schema_counts(stage) -> tuple[dict[str, int], dict[str, int]]:
    type_counts: dict[str, int] = {}
    api_counts: dict[str, int] = {}
    for prim in stage.Traverse():
        type_name = prim.GetTypeName() or "Typeless"
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
        for schema in prim.GetAppliedSchemas():
            api_counts[schema] = api_counts.get(schema, 0) + 1
    return type_counts, api_counts


def create_home_pose_collider_preview(config_path: Path, validation_json: Path) -> dict:
    from pxr import Sdf, UsdGeom

    from aloha_isaac_rebuild.scripts.create_aloha_home_pose_preview_stage import (
        _extract_home_pose_world_transforms,
    )
    from aloha_isaac_rebuild.scripts.create_aloha_stationary_style_rebuild_stages import (
        _build_visual_assembly,
        _define_stage,
        _reference_path,
        _set_bool_attr,
        _set_string_attr,
        _source_components,
        _write_resource_libraries,
    )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_path = (REPO_ROOT / config["source_aloha1_usd"]).resolve()
    output_path = REPO_ROOT / config["outputs"].get(
        "a16_home_pose_collider_preview",
        "aloha_isaac_rebuild/scenes/a16_aloha_home_pose_collider_preview.usda",
    )
    output_path = output_path.resolve()

    transforms, transform_report = _extract_home_pose_world_transforms(source_path)

    import omni.usd

    source_stage = omni.usd.get_context().get_stage()
    components = _source_components(source_stage)
    source_ref = _reference_path(output_path, source_path)
    stage, root = _define_stage(output_path, "A16_home_pose_collider_candidate_preview", config)
    _write_resource_libraries(
        stage,
        source_stage,
        source_ref,
        components,
        include_visuals=True,
        include_colliders=True,
        include_mesh_catalog=True,
    )
    _build_visual_assembly(
        stage,
        source_stage,
        components,
        include_collisions=True,
        component_world_transforms=transforms,
    )
    root.SetCustomDataByKey(
        "acceptance",
        "home-pose visual/collider candidate spatial preview only; no active collision",
    )
    root.SetCustomDataByKey("home_pose_transform_report_json", json.dumps(transform_report, indent=2, sort_keys=True))
    root.CreateAttribute("aloha:previewOnly", Sdf.ValueTypeNames.Bool).Set(True)
    root.CreateAttribute("aloha:visualColliderCandidatePreview", Sdf.ValueTypeNames.Bool).Set(True)
    root.CreateAttribute("aloha:homePoseBaked", Sdf.ValueTypeNames.Bool).Set(True)
    root.CreateAttribute("aloha:physicsReady", Sdf.ValueTypeNames.Bool).Set(False)
    root.CreateAttribute("aloha:contactValidationReady", Sdf.ValueTypeNames.Bool).Set(False)
    root.CreateAttribute("aloha:cleanArticulationAuthored", Sdf.ValueTypeNames.Bool).Set(False)

    records = []
    missing_collider_sources = []
    unexpected_empty_visual_bboxes = []
    unexpected_empty_collider_bboxes = []
    source_empty_visual_components = []
    source_empty_collider_components = []
    center_residual_warnings = []
    size_residual_warnings = []
    center_residual_threshold_m = 1.0e-3
    size_abs_threshold_m = 5.0e-3
    size_ratio_threshold = 0.05

    for component in components:
        if component.assembly_link_path is None:
            continue
        source_collisions = f"{component.source_component_path}/collisions"
        source_visuals = f"{component.source_component_path}/visuals"
        collider_exists = source_stage.GetPrimAtPath(source_collisions).IsValid()
        source_visual_bbox = _bbox_dict(source_stage, source_visuals)
        source_collider_bbox = _bbox_dict(source_stage, source_collisions)
        visual_path = f"{component.assembly_link_path}/visuals/{component.resource_name}"
        collider_path = f"{component.assembly_link_path}/collisions/{component.resource_name}"
        collider_prim = stage.GetPrimAtPath(collider_path)
        if collider_prim.IsValid():
            _set_bool_attr(collider_prim, "aloha:previewOnly", True)
            _set_bool_attr(collider_prim, "aloha:collisionEnabled", False)
            _set_bool_attr(collider_prim, "aloha:contactValidationReady", False)
            _set_bool_attr(collider_prim, "aloha:collisionApproved", False)
            _set_bool_attr(collider_prim, "aloha:colliderCandidate", collider_exists)
            _set_string_attr(collider_prim, "aloha:validFor", "visual_collider_spatial_candidate_check_only")
        visual_bbox = _bbox_dict(stage, visual_path)
        collider_bbox = _bbox_dict(stage, collider_path)
        residual = _bbox_residual(visual_bbox, collider_bbox)
        if not collider_exists:
            missing_collider_sources.append(
                {
                    "source_component_path": component.source_component_path,
                    "assembly_collider_path": collider_path,
                }
            )
        if not source_visual_bbox.get("valid"):
            source_empty_visual_components.append(
                {
                    "source_component_path": component.source_component_path,
                    "source_visuals_path": source_visuals,
                    "assembly_visual_path": visual_path,
                }
            )
        elif not visual_bbox.get("valid"):
            unexpected_empty_visual_bboxes.append(visual_path)
        if collider_exists and not source_collider_bbox.get("valid"):
            source_empty_collider_components.append(
                {
                    "source_component_path": component.source_component_path,
                    "source_collisions_path": source_collisions,
                    "assembly_collider_path": collider_path,
                }
            )
        elif collider_exists and not collider_bbox.get("valid"):
            unexpected_empty_collider_bboxes.append(collider_path)
        if residual.get("valid") and collider_exists:
            if residual["center_residual_m"] > center_residual_threshold_m:
                center_residual_warnings.append(
                    {
                        "assembly_link_path": component.assembly_link_path,
                        "center_residual_m": residual["center_residual_m"],
                    }
                )
            if (
                residual["max_size_axis_abs_residual_m"] > size_abs_threshold_m
                or residual["max_size_axis_ratio"] > size_ratio_threshold
            ):
                size_residual_warnings.append(
                    {
                        "assembly_link_path": component.assembly_link_path,
                        "max_size_axis_abs_residual_m": residual["max_size_axis_abs_residual_m"],
                        "max_size_axis_ratio": residual["max_size_axis_ratio"],
                    }
                )
        records.append(
            {
                "resource_name": component.resource_name,
                "source_component_path": component.source_component_path,
                "source_visuals_path": source_visuals,
                "source_collisions_path": source_collisions,
                "source_collider_exists": collider_exists,
                "assembly_link_path": component.assembly_link_path,
                "assembly_visual_path": visual_path,
                "assembly_collider_path": collider_path,
                "source_visual_bbox": source_visual_bbox,
                "source_collider_bbox": source_collider_bbox,
                "visual_bbox": visual_bbox,
                "collider_bbox": collider_bbox,
                "bbox_residual": residual,
            }
        )

    stage.GetRootLayer().Save()

    type_counts, api_counts = _stage_schema_counts(stage)
    resource_libraries_hidden = {
        "/visuals": UsdGeom.Imageable(stage.GetPrimAtPath("/visuals")).ComputeVisibility(),
        "/colliders": UsdGeom.Imageable(stage.GetPrimAtPath("/colliders")).ComputeVisibility(),
    }
    forbidden_runtime_type_hits = {
        name: type_counts.get(name, 0) for name in FORBIDDEN_RUNTIME_TYPES if type_counts.get(name, 0)
    }
    forbidden_physics_api_hits = {
        name: api_counts.get(name, 0) for name in FORBIDDEN_PHYSICS_APIS if api_counts.get(name, 0)
    }
    candidate_count = sum(1 for record in records if record["source_collider_exists"])
    ok = (
        resource_libraries_hidden["/visuals"] == "invisible"
        and resource_libraries_hidden["/colliders"] == "invisible"
        and not forbidden_runtime_type_hits
        and not forbidden_physics_api_hits
        and candidate_count > 0
        and not unexpected_empty_visual_bboxes
        and not unexpected_empty_collider_bboxes
    )
    validation = {
        "ok": ok,
        "status": "PASS_HOME_POSE_COLLIDER_CANDIDATE_PREVIEW_NO_ACTIVE_COLLISION" if ok else "FAIL_HOME_POSE_COLLIDER_CANDIDATE_PREVIEW",
        "output_usd": str(output_path),
        "source_usd": str(source_path),
        "pose": "REAL_RUNTIME_RESET_POSE",
        "component_count": len(components),
        "collider_candidate_count": candidate_count,
        "missing_collider_source_count": len(missing_collider_sources),
        "missing_collider_sources": missing_collider_sources,
        "unexpected_empty_visual_bboxes": unexpected_empty_visual_bboxes,
        "unexpected_empty_collider_bboxes": unexpected_empty_collider_bboxes,
        "source_empty_visual_components": source_empty_visual_components,
        "source_empty_collider_components": source_empty_collider_components,
        "center_residual_threshold_m": center_residual_threshold_m,
        "size_abs_threshold_m": size_abs_threshold_m,
        "size_ratio_threshold": size_ratio_threshold,
        "center_residual_warnings": center_residual_warnings,
        "size_residual_warnings": size_residual_warnings,
        "resource_libraries_hidden": resource_libraries_hidden,
        "forbidden_runtime_type_hits": forbidden_runtime_type_hits,
        "forbidden_physics_api_hits": forbidden_physics_api_hits,
        "physics_ready": False,
        "training_eligible": False,
        "contact_validation_ready": False,
        "clean_articulation_authored": False,
        "records": records,
    }
    validation_json.parent.mkdir(parents=True, exist_ok=True)
    validation_json.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_VALIDATION_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    app = _start_isaac_headless()
    try:
        result = create_home_pose_collider_preview(args.config, args.json_output)
        summary = {key: value for key, value in result.items() if key != "records"}
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        app.close()


if __name__ == "__main__":
    main()
