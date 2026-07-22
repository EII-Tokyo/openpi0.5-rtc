#!/usr/bin/env python3
"""Create A14-A18 Stationary-AI-style ALOHA1 rebuild stages.

These stages reorganize the already-confirmed ALOHA1 source USD into a cleaner
USD asset layout. They intentionally do not make runtime claims about physics,
articulation, cameras, or training readiness.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


DEFAULT_CONFIG = Path("aloha_isaac_rebuild/configs/physical_reconstruction/stationary_style_rebuild.yaml")

LEFT_LINK_ALIASES = {
    "left_base_link": "follower_left_base_link",
    "left_shoulder_link": "follower_left_link_1",
    "left_upper_arm_link": "follower_left_link_2",
    "left_upper_forearm_link": "follower_left_link_3",
    "left_lower_forearm_link": "follower_left_link_4",
    "left_wrist_link": "follower_left_link_5",
    "left_gripper_link": "follower_left_link_6",
    "left_gripper_base": "follower_left_gripper_base",
    "left_left_finger_link": "follower_left_gripper_left",
    "left_right_finger_link": "follower_left_gripper_right",
}

RIGHT_LINK_ALIASES = {
    "right_base_link": "follower_right_base_link",
    "right_shoulder_link": "follower_right_link_1",
    "right_upper_arm_link": "follower_right_link_2",
    "right_upper_forearm_link": "follower_right_link_3",
    "right_lower_forearm_link": "follower_right_link_4",
    "right_wrist_link": "follower_right_link_5",
    "right_gripper_link": "follower_right_link_6",
    "right_gripper_base": "follower_right_gripper_base",
    "right_left_finger_link": "follower_right_gripper_left",
    "right_right_finger_link": "follower_right_gripper_right",
}

PHYSICS_API_SCHEMAS = (
    UsdPhysics.CollisionAPI,
    UsdPhysics.RigidBodyAPI,
    UsdPhysics.MassAPI,
    UsdPhysics.ArticulationRootAPI,
    UsdPhysics.FilteredPairsAPI,
)


@dataclass(frozen=True)
class Component:
    name: str
    role: str
    source_component_path: str
    resource_name: str
    assembly_link_path: str | None


def _set_string_attr(prim: Usd.Prim, name: str, value: str) -> None:
    prim.CreateAttribute(name, Sdf.ValueTypeNames.String).Set(value)


def _set_bool_attr(prim: Usd.Prim, name: str, value: bool) -> None:
    prim.CreateAttribute(name, Sdf.ValueTypeNames.Bool).Set(value)


def _set_token_attr(prim: Usd.Prim, name: str, value: str) -> None:
    prim.CreateAttribute(name, Sdf.ValueTypeNames.Token).Set(value)


def _remove_physics_apis(prim: Usd.Prim) -> None:
    for api_schema in PHYSICS_API_SCHEMAS:
        if prim.HasAPI(api_schema):
            prim.RemoveAPI(api_schema)


def _remove_physics_apis_matching_source_subtree(
    dst_stage: Usd.Stage,
    src_stage: Usd.Stage,
    source_root: str,
    destination_root: str,
) -> None:
    src_root = src_stage.GetPrimAtPath(source_root)
    if not src_root.IsValid():
        return
    for src_prim in Usd.PrimRange(src_root):
        if not src_prim.GetAppliedSchemas():
            continue
        src_path = str(src_prim.GetPath())
        relative = src_path.removeprefix(source_root).strip("/")
        dst_path = destination_root if not relative else f"{destination_root}/{relative}"
        _remove_physics_apis(dst_stage.OverridePrim(dst_path))


def _set_transform_matrix(prim: Usd.Prim, matrix: Gf.Matrix4d) -> None:
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    transform_op = xformable.AddTransformOp()
    transform_op.Set(matrix)


def _source_world_transform(src_stage: Usd.Stage, source_path: str) -> Gf.Matrix4d:
    prim = src_stage.GetPrimAtPath(source_path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing source prim for transform: {source_path}")
    return UsdGeom.XformCache().GetLocalToWorldTransform(prim)


def _reference_path(output_path: Path, source_path: Path) -> str:
    return os.path.relpath(source_path.resolve(), output_path.resolve().parent)


def _safe_worldbody_alias(index: int, source_name: str) -> str:
    if source_name == "_":
        suffix = "underscore"
    elif source_name.startswith("__"):
        suffix = source_name[2:] or "anonymous"
    else:
        suffix = source_name
    return f"frame_part_{index:02d}_{suffix}"


def _child_has_source_subtree(src_stage: Usd.Stage, path: str) -> bool:
    prim = src_stage.GetPrimAtPath(path)
    return prim.IsValid() and bool(list(prim.GetChildren()))


def _source_bbox_valid(src_stage: Usd.Stage, path: str) -> bool:
    prim = src_stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return False
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]).ComputeWorldBound(prim).ComputeAlignedBox()
    if box.IsEmpty():
        return False
    size = box.GetSize()
    return any(float(value) > 0.0 for value in size)


def _define_stage(output_path: Path, root_role: str, config: dict) -> tuple[Usd.Stage, Usd.Prim]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, float(config["policy"]["meters_per_unit"]))
    root = UsdGeom.Xform.Define(stage, config["root_prim"]).GetPrim()
    stage.SetDefaultPrim(root)
    _set_string_attr(root, "aloha:stageRole", root_role)
    _set_string_attr(root, "aloha:sourceKind", "original_aloha1_reorganized_to_stationary_ai_style")
    _set_bool_attr(root, "aloha:sourceAloha1ReadOnly", True)
    _set_bool_attr(root, "aloha:stationaryAiRuntimeCompatible", False)
    _set_bool_attr(root, "aloha:physicsReady", False)
    _set_bool_attr(root, "aloha:trainingEligible", False)
    root.SetCustomDataByKey("source_aloha1_usd", config["source_aloha1_usd"])
    root.SetCustomDataByKey("trossen_reference_usd", config["trossen_reference_usd"])
    return stage, root


def _define_top_libraries(stage: Usd.Stage) -> None:
    for path, role, hide_from_viewport in (
        ("/meshes", "stationary_ai_style_mesh_catalog", False),
        ("/visuals", "stationary_ai_style_visual_resource_library", True),
        ("/colliders", "stationary_ai_style_collider_candidate_library", True),
    ):
        prim = UsdGeom.Scope.Define(stage, path).GetPrim()
        _set_string_attr(prim, "aloha:stageRole", role)
        if hide_from_viewport:
            # Match the Stationary AI asset-library pattern: resources exist for
            # composition/reference, but they are not independent scene objects.
            UsdGeom.Imageable(prim).MakeInvisible()


def _define_resource_manifest(stage: Usd.Stage, path: str, role: str, records: list[dict]) -> None:
    prim = UsdGeom.Scope.Define(stage, path).GetPrim()
    _set_string_attr(prim, "aloha:stageRole", role)
    prim.SetCustomDataByKey("records_json", json.dumps(records, indent=2, sort_keys=True))


def _define_reference_resource(
    stage: Usd.Stage,
    src_stage: Usd.Stage,
    source_ref: str,
    resource_path: str,
    source_subtree_path: str,
    *,
    role: str,
    component: Component,
    strip_physics: bool,
    mark_collider_candidate: bool,
    semantic_only_if_missing: bool = False,
) -> bool:
    exists = src_stage.GetPrimAtPath(source_subtree_path).IsValid()
    if exists and not semantic_only_if_missing:
        prim = UsdGeom.Xform.Define(stage, resource_path).GetPrim()
        prim.GetReferences().AddReference(source_ref, source_subtree_path)
        if strip_physics:
            _remove_physics_apis_matching_source_subtree(stage, src_stage, source_subtree_path, resource_path)
    else:
        prim = UsdGeom.Scope.Define(stage, resource_path).GetPrim()
    _set_string_attr(prim, "aloha:stageRole", role)
    _set_string_attr(prim, "aloha:sourceComponentPrim", component.source_component_path)
    _set_string_attr(prim, "aloha:sourceSubtreePrim", source_subtree_path)
    _set_string_attr(prim, "aloha:resourceName", component.resource_name)
    _set_string_attr(prim, "aloha:componentRole", component.role)
    _set_bool_attr(prim, "aloha:sourceSubtreeExists", exists)
    _set_bool_attr(prim, "aloha:physicsReady", False)
    _set_bool_attr(prim, "aloha:trainingEligible", False)
    if strip_physics:
        _set_bool_attr(prim, "aloha:physicsApisStripped", True)
    if mark_collider_candidate:
        _set_bool_attr(prim, "aloha:colliderCandidate", exists)
        _set_bool_attr(prim, "aloha:collisionApproved", False)
    return exists


def _add_internal_reference(stage: Usd.Stage, dst_path: str, source_path: str, role: str) -> Usd.Prim:
    prim = UsdGeom.Xform.Define(stage, dst_path).GetPrim()
    prim.GetReferences().AddInternalReference(Sdf.Path(source_path))
    _set_string_attr(prim, "aloha:stageRole", role)
    _set_string_attr(prim, "aloha:internalResourcePrim", source_path)
    _set_bool_attr(prim, "aloha:physicsReady", False)
    _set_bool_attr(prim, "aloha:trainingEligible", False)
    return prim


def _source_components(src_stage: Usd.Stage) -> list[Component]:
    components: list[Component] = []
    worldbody = src_stage.GetPrimAtPath("/scene/worldBody")
    if not worldbody.IsValid():
        raise RuntimeError("Missing /scene/worldBody in ALOHA1 source")
    for index, child in enumerate(worldbody.GetChildren()):
        name = child.GetName()
        if name == "table":
            resource = "tabletop_link"
            assembly = "/aloha/tabletop_link"
            role = "table"
        elif name == "floor":
            resource = "floor_reference_link"
            assembly = "/aloha/floor_reference_link"
            role = "floor_reference"
        else:
            resource = _safe_worldbody_alias(index, name)
            assembly = f"/aloha/frame_link/{resource}"
            role = "support_frame"
        components.append(Component(name, role, str(child.GetPath()), resource, assembly))

    for source_root, aliases, side in (
        ("/scene/left_base_link", LEFT_LINK_ALIASES, "left_robot"),
        ("/scene/right_base_link", RIGHT_LINK_ALIASES, "right_robot"),
    ):
        root = src_stage.GetPrimAtPath(source_root)
        if not root.IsValid():
            raise RuntimeError(f"Missing {source_root} in ALOHA1 source")
        for child in root.GetChildren():
            if child.GetName() not in aliases:
                continue
            resource = aliases[child.GetName()]
            components.append(
                Component(
                    child.GetName(),
                    side,
                    str(child.GetPath()),
                    resource,
                    f"/aloha/{resource}",
                )
            )
    return components


def _write_resource_libraries(
    stage: Usd.Stage,
    src_stage: Usd.Stage,
    source_ref: str,
    components: list[Component],
    *,
    include_visuals: bool,
    include_colliders: bool,
    include_mesh_catalog: bool,
) -> list[dict]:
    records: list[dict] = []
    _define_top_libraries(stage)
    for component in components:
        collider_source = f"{component.source_component_path}/collisions"
        declared_visual_source = f"{component.source_component_path}/visuals"
        visual_source = declared_visual_source
        visual_source_kind = "declared_visuals"
        if not _source_bbox_valid(src_stage, visual_source) and _source_bbox_valid(src_stage, collider_source):
            visual_source = collider_source
            visual_source_kind = "collision_subtree_used_as_visual_proxy"
        visual_path = f"/visuals/{component.resource_name}"
        collider_path = f"/colliders/{component.resource_name}"
        mesh_catalog_path = f"/meshes/{component.resource_name}"
        visual_exists = src_stage.GetPrimAtPath(visual_source).IsValid()
        collider_exists = src_stage.GetPrimAtPath(collider_source).IsValid()
        if include_visuals:
            visual_exists = _define_reference_resource(
                stage,
                src_stage,
                source_ref,
                visual_path,
                visual_source,
                role="visual_resource_from_original_aloha1_visuals",
                component=component,
                strip_physics=True,
                mark_collider_candidate=False,
            )
        if include_colliders:
            collider_exists = _define_reference_resource(
                stage,
                src_stage,
                source_ref,
                collider_path,
                collider_source,
                role="collider_candidate_from_original_aloha1_collisions",
                component=component,
                strip_physics=True,
                mark_collider_candidate=True,
            )
        if include_mesh_catalog:
            catalog = UsdGeom.Scope.Define(stage, mesh_catalog_path).GetPrim()
            _set_string_attr(catalog, "aloha:stageRole", "mesh_catalog_entry_not_flattened")
            _set_string_attr(catalog, "aloha:sourceComponentPrim", component.source_component_path)
            _set_string_attr(catalog, "aloha:sourceVisualsPrim", visual_source)
            _set_string_attr(catalog, "aloha:sourceCollidersPrim", collider_source)
            _set_bool_attr(catalog, "aloha:copiedMeshData", False)
            _set_bool_attr(catalog, "aloha:requiresFutureMeshDecomposition", True)
        records.append(
            {
                "resource_name": component.resource_name,
                "role": component.role,
                "source_component_path": component.source_component_path,
                "assembly_link_path": component.assembly_link_path,
                "visual_resource": visual_path,
                "visual_source_exists": visual_exists,
                "visual_source_kind": visual_source_kind,
                "declared_visual_source": declared_visual_source,
                "effective_visual_source": visual_source,
                "collider_resource": collider_path,
                "collider_source_exists": collider_exists,
                "mesh_catalog": mesh_catalog_path,
            }
        )
    _define_resource_manifest(stage, "/aloha/source_manifest", "stationary_style_source_manifest", records)
    return records


def _build_visual_assembly(
    stage: Usd.Stage,
    src_stage: Usd.Stage,
    components: list[Component],
    *,
    include_collisions: bool = False,
    component_world_transforms: dict[str, Gf.Matrix4d] | None = None,
) -> None:
    frame = UsdGeom.Xform.Define(stage, "/aloha/frame_link").GetPrim()
    _set_string_attr(frame, "aloha:stageRole", "support_frame_assembly_link")
    _set_bool_attr(frame, "aloha:physicsReady", False)
    _set_bool_attr(frame, "aloha:trainingEligible", False)

    for component in components:
        if component.assembly_link_path is None:
            continue
        link = UsdGeom.Xform.Define(stage, component.assembly_link_path).GetPrim()
        matrix = (
            component_world_transforms.get(component.source_component_path)
            if component_world_transforms is not None
            else None
        )
        if matrix is None:
            matrix = _source_world_transform(src_stage, component.source_component_path)
        _set_transform_matrix(link, matrix)
        _set_string_attr(link, "aloha:stageRole", f"{component.role}_assembly_link")
        _set_string_attr(link, "aloha:sourceComponentPrim", component.source_component_path)
        _set_bool_attr(link, "aloha:physicsReady", False)
        _set_bool_attr(link, "aloha:trainingEligible", False)
        _add_internal_reference(
            stage,
            f"{component.assembly_link_path}/visuals/{component.resource_name}",
            f"/visuals/{component.resource_name}",
            "assembly_visual_internal_reference",
        )
        if include_collisions:
            collider_source = f"{component.source_component_path}/collisions"
            collider_exists = src_stage.GetPrimAtPath(collider_source).IsValid()
            prim = _add_internal_reference(
                stage,
                f"{component.assembly_link_path}/collisions/{component.resource_name}",
                f"/colliders/{component.resource_name}",
                "assembly_collider_candidate_internal_reference",
            )
            _set_bool_attr(prim, "aloha:colliderCandidate", collider_exists)
            _set_bool_attr(prim, "aloha:collisionApproved", False)
            _set_bool_attr(prim, "aloha:collisionEnabled", False)
            _set_bool_attr(prim, "aloha:contactValidationReady", False)
            _set_string_attr(prim, "aloha:sourceCollisionsPrim", collider_source)


def _audit_source_joints(src_stage: Usd.Stage) -> dict:
    joint_records: list[dict] = []
    for prim in src_stage.Traverse():
        type_name = prim.GetTypeName() or ""
        if "Joint" not in type_name and not str(prim.GetPath()).endswith("/joints"):
            continue
        record = {
            "path": str(prim.GetPath()),
            "type": type_name,
            "applied_schemas": list(prim.GetAppliedSchemas()),
            "relationships": {},
            "attributes": {},
        }
        for rel in prim.GetRelationships():
            targets = [str(target) for target in rel.GetTargets()]
            if targets:
                record["relationships"][rel.GetName()] = targets
        for attr in prim.GetAuthoredAttributes():
            name = attr.GetName()
            if name.startswith("physics:") or name.startswith("drive:") or "limit" in name.lower() or "axis" in name.lower():
                value = attr.Get()
                record["attributes"][name] = str(value)
        joint_records.append(record)
    return {
        "status": "JOINT_EVIDENCE_RECORDED_CLEAN_ARTICULATION_NOT_AUTHORED",
        "author_articulation": False,
        "verified_aloha1_control_facts": {
            "status": "EVIDENCE_AVAILABLE_BUT_NOT_YET_AUTHORED_IN_CLEAN_ALOHA",
            "dof_order_per_arm": [
                "waist",
                "shoulder",
                "elbow",
                "forearm_roll",
                "wrist_angle",
                "wrist_rotate",
                "gripper",
                "left_finger",
                "right_finger",
            ],
            "arm_limits_rad": {
                "waist": [-3.141582489, 3.141582489],
                "shoulder": [-1.8500489, 1.256637096],
                "elbow": [-1.762782454, 1.605702758],
                "forearm_roll": [-3.141582489, 3.141582489],
                "wrist_angle": [-1.867502093, 2.234021187],
                "wrist_rotate": [-3.141582489, 3.141582489],
            },
            "replay_cadence_hz": 50,
            "validated_reports": [
                "reports/aloha1_isaac_adaptation/phase18_runtime_articulation_20260718/physics_wrapper_runtime_articulation.json",
                "reports/aloha1_isaac_adaptation/phase19_native_wrapper_candidate_20260718/physics_wrapper_runtime_articulation.json",
                "reports/aloha1_isaac_adaptation/phase20_dof_drive_limits_20260718/dof_drive_limits.md",
                "reports/aloha1_isaac_adaptation/phase20_dof_drive_limits_20260718/dof_drive_limits.json",
                "reports/aloha1_isaac_adaptation/phase21_arm_qpos_replay_native_20260718/replay_metrics.json",
                "reports/aloha1_isaac_adaptation/phase97_scene_proxy_hdf5_replay_drive_target_arm1600_kd100_finger200_native_workcell_20260718/gripper_passive_contact_metrics.json",
                "configs/aloha/original_stationary_aloha_mapping.yaml",
                "configs/aloha/trossen_scene_base_link_aloha1_left_controller_mapping.yaml",
            ],
            "strongest_runtime_gate": {
                "report": "reports/aloha1_isaac_adaptation/phase97_scene_proxy_hdf5_replay_drive_target_arm1600_kd100_finger200_native_workcell_20260718/gripper_passive_contact_metrics.json",
                "target_limit_gate_ok": True,
                "controller_tracking_gate_pass": True,
                "max_controlled_error": 0.012857437133789062,
                "contact_trace_status": "PASS_BILATERAL_CONTACT_CANDIDATE",
                "failure_reasons": [],
            },
            "not_authoring_reason": (
                "These reports validate ALOHA1 control facts in prior source/wrapper stages. "
                "The clean /aloha Stationary-AI-style hierarchy still needs an explicit joint "
                "authoring pass that maps those facts onto the new prim paths and then reruns "
                "set-target/readback, hold, and replay gates."
            ),
        },
        "source_joint_count": len(joint_records),
        "source_joints": joint_records,
        "reason": "A17 records source joints plus prior verified ALOHA1 DOF/order/limit/replay evidence. Clean /aloha articulation is still not authored; the next pass must map evidence onto the new prim paths and rerun controller gates.",
    }


def _write_camera_status(output_path: Path) -> dict:
    cameras = [
        {
            "runtime_name": "cam_high",
            "recommended_prim": "/aloha/sensors/cam_high",
            "status": "DEFERRED",
            "reason": "Camera calibration is intentionally deferred until support frame and robot assembly are stable.",
        },
        {
            "runtime_name": "cam_low",
            "recommended_prim": "/aloha/sensors/cam_low",
            "status": "DEFERRED",
            "reason": "Needs measured mount pose, intrinsics, and frame convention before authoring UsdGeom.Camera.",
        },
        {
            "runtime_name": "cam_left_wrist",
            "recommended_prim": "/aloha/follower_left_camera_link/camera",
            "status": "DEFERRED",
            "reason": "Original converted D405 visual mesh is not a calibrated Isaac Camera sensor.",
        },
        {
            "runtime_name": "cam_right_wrist",
            "recommended_prim": "/aloha/follower_right_camera_link/camera",
            "status": "DEFERRED",
            "reason": "Original converted D405 visual mesh is not a calibrated Isaac Camera sensor.",
        },
    ]
    data = {
        "status": "CAMERA_AUTHORING_DEFERRED",
        "author_calibrated_cameras": False,
        "camera_status": cameras,
        "required_evidence": [
            "measured or calibrated camera extrinsics",
            "camera intrinsics and distortion model",
            "runtime camera name mapping",
            "image resolution and optical frame convention",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def _write_camera_gate_stage(output_path: Path, config: dict, camera_status: dict) -> None:
    stage, root = _define_stage(output_path, "A18_camera_semantic_gate", config)
    sensors = UsdGeom.Scope.Define(stage, "/aloha/sensors").GetPrim()
    _set_string_attr(sensors, "aloha:stageRole", "camera_slots_not_calibrated_sensors")
    for camera in camera_status["camera_status"]:
        slot = UsdGeom.Xform.Define(stage, camera["recommended_prim"]).GetPrim()
        _set_string_attr(slot, "aloha:runtimeCameraName", camera["runtime_name"])
        _set_string_attr(slot, "aloha:status", camera["status"])
        _set_string_attr(slot, "aloha:reason", camera["reason"])
        _set_bool_attr(slot, "aloha:isIsaacCameraSensor", False)
        _set_bool_attr(slot, "aloha:cameraCalibrationReady", False)
    root.SetCustomDataByKey("camera_status_json", json.dumps(camera_status, indent=2, sort_keys=True))
    stage.GetRootLayer().Save()


def create_all(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_path = Path(config["source_aloha1_usd"])
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    source_stage = Usd.Stage.Open(str(source_path), load=Usd.Stage.LoadAll)
    if source_stage is None:
        raise RuntimeError(f"Could not open source stage: {source_path}")
    components = _source_components(source_stage)
    outputs = {key: Path(value) for key, value in config["outputs"].items()}

    source_ref_a14 = _reference_path(outputs["a14_resource_decomposition"], source_path)
    stage, root = _define_stage(outputs["a14_resource_decomposition"], "A14_resource_decomposition", config)
    records = _write_resource_libraries(
        stage,
        source_stage,
        source_ref_a14,
        components,
        include_visuals=True,
        include_colliders=True,
        include_mesh_catalog=True,
    )
    root.SetCustomDataByKey("component_count", len(records))
    root.SetCustomDataByKey("acceptance", "resource libraries exist; physics remains blocked")
    stage.GetRootLayer().Save()

    source_ref_a15 = _reference_path(outputs["a15_visual_assembly"], source_path)
    stage, root = _define_stage(outputs["a15_visual_assembly"], "A15_visual_assembly", config)
    _write_resource_libraries(
        stage,
        source_stage,
        source_ref_a15,
        components,
        include_visuals=True,
        include_colliders=False,
        include_mesh_catalog=True,
    )
    _build_visual_assembly(stage, source_stage, components, include_collisions=False)
    root.SetCustomDataByKey("acceptance", "assembly uses /visuals resources only; no collisions under assembly")
    stage.GetRootLayer().Save()

    source_ref_a16 = _reference_path(outputs["a16_collider_structure"], source_path)
    stage, root = _define_stage(outputs["a16_collider_structure"], "A16_collider_structure_candidate", config)
    _write_resource_libraries(
        stage,
        source_stage,
        source_ref_a16,
        components,
        include_visuals=True,
        include_colliders=True,
        include_mesh_catalog=True,
    )
    _build_visual_assembly(stage, source_stage, components, include_collisions=True)
    root.SetCustomDataByKey("acceptance", "collider candidate resources are separated but not physics-approved")
    stage.GetRootLayer().Save()

    joint_audit = _audit_source_joints(source_stage)
    outputs["a17_joint_audit_json"].parent.mkdir(parents=True, exist_ok=True)
    outputs["a17_joint_audit_json"].write_text(json.dumps(joint_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    camera_status = _write_camera_status(outputs["a18_camera_status_json"])
    camera_gate_stage = outputs["a18_camera_status_json"].with_suffix(".usda")
    _write_camera_gate_stage(camera_gate_stage, config, camera_status)

    return {
        "component_count": len(components),
        "outputs": {key: str(value) for key, value in outputs.items()},
        "a18_camera_gate_stage": str(camera_gate_stage),
        "joint_audit_status": joint_audit["status"],
        "camera_status": camera_status["status"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = create_all(args.config)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
