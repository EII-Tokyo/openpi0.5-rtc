#!/usr/bin/env python3
"""Audit A14-A18 Stationary-AI-style ALOHA1 rebuild outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from pxr import Gf, Usd, UsdGeom


DEFAULT_CONFIG = Path("aloha_isaac_rebuild/configs/physical_reconstruction/stationary_style_rebuild.yaml")
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
KEY_BBOX_PATHS = (
    "/aloha",
    "/aloha/tabletop_link",
    "/aloha/frame_link",
    "/aloha/follower_left_base_link",
    "/aloha/follower_right_base_link",
)


def _stage_summary(stage_path: Path) -> dict:
    stage = Usd.Stage.Open(str(stage_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not open stage: {stage_path}")
    default_prim = stage.GetDefaultPrim()
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    key_bboxes = {}
    for path in KEY_BBOX_PATHS:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            key_bboxes[path] = {"valid": False}
            continue
        box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        size = box.GetSize()
        center = (box.GetMin() + box.GetMax()) * 0.5
        key_bboxes[path] = {
            "valid": not box.IsEmpty() and all(value > 0.0 for value in size),
            "center": [float(center[0]), float(center[1]), float(center[2])],
            "size": [float(size[0]), float(size[1]), float(size[2])],
        }
    paths = set()
    type_counts: dict[str, int] = {}
    api_counts: dict[str, int] = {}
    reference_paths = []
    internal_reference_paths = []
    external_reference_paths = []
    collider_candidate_paths = []
    collision_approved_true = []
    resource_attr_missing = []
    assembly_visual_reference_paths = []
    assembly_visual_invisible_paths = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        paths.add(path)
        prim_type = prim.GetTypeName() or "Typeless"
        type_counts[prim_type] = type_counts.get(prim_type, 0) + 1
        for api in prim.GetAppliedSchemas():
            api_counts[api] = api_counts.get(api, 0) + 1
        if prim.HasAuthoredReferences():
            reference_paths.append(path)
            refs = prim.GetMetadata("references")
            refs_text = str(refs)
            if "assetPath = @" in refs_text:
                external_reference_paths.append(path)
            else:
                internal_reference_paths.append(path)
            role_attr = prim.GetAttribute("aloha:stageRole")
            role_value = role_attr.Get() if role_attr.IsValid() else None
            if role_value == "assembly_visual_internal_reference":
                assembly_visual_reference_paths.append(path)
                if UsdGeom.Imageable(prim).ComputeVisibility() == "invisible":
                    assembly_visual_invisible_paths.append(path)
        attr = prim.GetAttribute("aloha:colliderCandidate")
        if attr.IsValid() and attr.Get():
            collider_candidate_paths.append(path)
        approved = prim.GetAttribute("aloha:collisionApproved")
        if approved.IsValid() and approved.Get():
            collision_approved_true.append(path)
        if path.startswith(("/meshes/", "/visuals/", "/colliders/")) and path.count("/") == 2:
            source_attr = prim.GetAttribute("aloha:sourceComponentPrim")
            if not source_attr.IsValid():
                resource_attr_missing.append(path)
    direct_children = {}
    resource_root_visibility = {}
    for root in ("/meshes", "/visuals", "/colliders"):
        prim = stage.GetPrimAtPath(root)
        direct_children[root] = len(list(prim.GetChildren())) if prim and prim.IsValid() else -1
        if prim and prim.IsValid():
            resource_root_visibility[root] = UsdGeom.Imageable(prim).ComputeVisibility()
        else:
            resource_root_visibility[root] = "missing"
    return {
        "stage_path": str(stage_path),
        "default_prim": str(default_prim.GetPath()) if default_prim else None,
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "paths": sorted(paths),
        "type_counts": type_counts,
        "api_counts": api_counts,
        "reference_count": len(reference_paths),
        "internal_reference_count": len(internal_reference_paths),
        "external_reference_count": len(external_reference_paths),
        "collider_candidate_count": len(collider_candidate_paths),
        "collision_approved_true": collision_approved_true,
        "resource_direct_children": direct_children,
        "resource_root_visibility": resource_root_visibility,
        "assembly_visual_reference_count": len(assembly_visual_reference_paths),
        "assembly_visual_invisible_paths": assembly_visual_invisible_paths,
        "resource_attr_missing": resource_attr_missing,
        "key_bboxes": key_bboxes,
        "forbidden_runtime_type_hits": {
            name: type_counts.get(name, 0)
            for name in FORBIDDEN_RUNTIME_TYPES
            if type_counts.get(name, 0)
        },
        "forbidden_physics_api_hits": {
            name: api_counts.get(name, 0)
            for name in FORBIDDEN_PHYSICS_APIS
            if api_counts.get(name, 0)
        },
    }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    outputs = {key: Path(value) for key, value in config["outputs"].items()}
    a18_stage = outputs["a18_camera_status_json"].with_suffix(".usda")
    a16_home_audit_path = Path("aloha_isaac_rebuild/artifacts/validation/a16_home_pose_collider_preview_audit.json")

    summaries = {
        "a14": _stage_summary(outputs["a14_resource_decomposition"]),
        "a15": _stage_summary(outputs["a15_visual_assembly"]),
        "a16": _stage_summary(outputs["a16_collider_structure"]),
        "a18_stage": _stage_summary(a18_stage),
    }
    joint_audit = _load_json(outputs["a17_joint_audit_json"])
    clean_mapping_plan = _load_json(outputs["a17_clean_articulation_mapping_plan_json"])
    clean_tree_preflight = _load_json(outputs["a18_clean_kinematic_tree_preflight_json"])
    camera_status = _load_json(outputs["a18_camera_status_json"])
    a16_home_audit = _load_json(a16_home_audit_path) if a16_home_audit_path.exists() else {}

    expected_top = {"/aloha", "/visuals", "/colliders", "/meshes", "/aloha/source_manifest"}
    a14_paths = set(summaries["a14"]["paths"])
    a15_paths = set(summaries["a15"]["paths"])
    a16_paths = set(summaries["a16"]["paths"])
    a18_paths = set(summaries["a18_stage"]["paths"])
    a15_collision_paths = [path for path in a15_paths if "/collisions" in path]
    a16_collision_paths = [path for path in a16_paths if "/collisions" in path]
    a15_key_bboxes = summaries["a15"]["key_bboxes"]
    a15_key_bboxes_valid = all(a15_key_bboxes[path].get("valid") for path in KEY_BBOX_PATHS)
    left_center = a15_key_bboxes["/aloha/follower_left_base_link"].get("center", [0.0, 0.0, 0.0])
    right_center = a15_key_bboxes["/aloha/follower_right_base_link"].get("center", [0.0, 0.0, 0.0])
    left_right_x_separation = abs(float(left_center[0]) - float(right_center[0]))
    ok = (
        summaries["a14"]["default_prim"] == "/aloha"
        and summaries["a15"]["default_prim"] == "/aloha"
        and summaries["a16"]["default_prim"] == "/aloha"
        and summaries["a18_stage"]["default_prim"] == "/aloha"
        and summaries["a14"]["meters_per_unit"] == 1.0
        and summaries["a14"]["up_axis"] == "Z"
        and expected_top.issubset(a14_paths)
        and expected_top.issubset(a15_paths)
        and expected_top.issubset(a16_paths)
        and not a15_collision_paths
        and len(a16_collision_paths) > 0
        and summaries["a16"]["collider_candidate_count"] > 0
        and not summaries["a16"]["collision_approved_true"]
        and not summaries["a14"]["forbidden_runtime_type_hits"]
        and not summaries["a15"]["forbidden_runtime_type_hits"]
        and not summaries["a16"]["forbidden_runtime_type_hits"]
        and not summaries["a14"]["forbidden_physics_api_hits"]
        and not summaries["a15"]["forbidden_physics_api_hits"]
        and not summaries["a16"]["forbidden_physics_api_hits"]
        and joint_audit.get("author_articulation") is False
        and joint_audit.get("status") == "JOINT_EVIDENCE_RECORDED_CLEAN_ARTICULATION_NOT_AUTHORED"
        and clean_mapping_plan.get("ok") is True
        and clean_mapping_plan.get("author_articulation") is False
        and clean_mapping_plan.get("physics_ready") is False
        and clean_mapping_plan.get("dof_joint_count") == 16
        and clean_mapping_plan.get("joint_count") == 20
        and clean_mapping_plan.get("unmapped_joint_count") == 0
        and clean_mapping_plan.get("no_canonical_dof_joint_count") == 0
        and clean_tree_preflight.get("ok") is True
        and clean_tree_preflight.get("author_articulation") is False
        and clean_tree_preflight.get("physics_ready") is False
        and clean_tree_preflight.get("joint_count") == 20
        and clean_tree_preflight.get("dof_joint_count") == 16
        and clean_tree_preflight.get("root_joint_count") == 2
        and camera_status.get("author_calibrated_cameras") is False
        and camera_status.get("status") == "CAMERA_AUTHORING_DEFERRED"
        and "/aloha/sensors/cam_low" in a18_paths
        and not summaries["a18_stage"]["forbidden_runtime_type_hits"]
        and summaries["a14"]["resource_direct_children"]["/meshes"] == 56
        and summaries["a14"]["resource_direct_children"]["/visuals"] == 56
        and summaries["a14"]["resource_direct_children"]["/colliders"] == 56
        and summaries["a15"]["resource_direct_children"]["/colliders"] == 0
        and summaries["a16"]["resource_direct_children"]["/colliders"] == 56
        and summaries["a14"]["resource_root_visibility"]["/visuals"] == "invisible"
        and summaries["a15"]["resource_root_visibility"]["/visuals"] == "invisible"
        and summaries["a16"]["resource_root_visibility"]["/visuals"] == "invisible"
        and summaries["a14"]["resource_root_visibility"]["/colliders"] == "invisible"
        and summaries["a15"]["resource_root_visibility"]["/colliders"] == "invisible"
        and summaries["a16"]["resource_root_visibility"]["/colliders"] == "invisible"
        and summaries["a15"]["assembly_visual_reference_count"] > 0
        and not summaries["a15"]["assembly_visual_invisible_paths"]
        and not summaries["a14"]["resource_attr_missing"]
        and not summaries["a15"]["resource_attr_missing"]
        and not summaries["a16"]["resource_attr_missing"]
        and a15_key_bboxes_valid
        and left_right_x_separation > 0.2
        and a16_home_audit.get("ok") is True
        and a16_home_audit.get("physics_ready") is False
        and a16_home_audit.get("contact_validation_ready") is False
        and a16_home_audit.get("clean_articulation_authored") is False
    )
    return {
        "ok": ok,
        "config_path": str(config_path),
        "summaries": {
            key: {name: value for name, value in summary.items() if name != "paths"}
            for key, summary in summaries.items()
        },
        "a15_collision_paths": sorted(a15_collision_paths)[:20],
        "a16_collision_path_count": len(a16_collision_paths),
        "a15_key_bboxes_valid": a15_key_bboxes_valid,
        "a15_left_right_x_separation": left_right_x_separation,
        "joint_audit_status": joint_audit.get("status"),
        "joint_source_count": joint_audit.get("source_joint_count"),
        "a17_clean_articulation_mapping_status": clean_mapping_plan.get("status"),
        "a17_clean_articulation_joint_count": clean_mapping_plan.get("joint_count"),
        "a17_clean_articulation_dof_joint_count": clean_mapping_plan.get("dof_joint_count"),
        "a17_clean_articulation_unmapped_joint_count": clean_mapping_plan.get("unmapped_joint_count"),
        "a17_clean_articulation_no_canonical_dof_joint_count": clean_mapping_plan.get("no_canonical_dof_joint_count"),
        "a18_clean_kinematic_tree_preflight_status": clean_tree_preflight.get("status"),
        "a18_clean_kinematic_tree_preflight_root_joint_count": clean_tree_preflight.get("root_joint_count"),
        "camera_status": camera_status.get("status"),
        "camera_count": len(camera_status.get("camera_status", [])),
        "a16_home_pose_collider_preview_status": a16_home_audit.get("status", "MISSING"),
        "a16_home_pose_collider_preview_candidate_count": a16_home_audit.get("collider_candidate_count"),
        "a16_home_pose_collider_preview_size_warning_count": len(a16_home_audit.get("size_residual_warnings", [])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(args.config)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
