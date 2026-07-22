#!/usr/bin/env python3
"""Create A13 clean visual baseline from original ALOHA1 USD visuals."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


DEFAULT_CONFIG = Path("aloha_isaac_rebuild/configs/physical_reconstruction/original_aloha1_visual_baseline.yaml")
DEFAULT_OUTPUT = Path("aloha_isaac_rebuild/scenes/aloha_original_visual_baseline.usda")


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


def _set_string_attr(prim: Usd.Prim, name: str, value: str) -> None:
    prim.CreateAttribute(name, Sdf.ValueTypeNames.String).Set(value)


def _set_bool_attr(prim: Usd.Prim, name: str, value: bool) -> None:
    prim.CreateAttribute(name, Sdf.ValueTypeNames.Bool).Set(value)


def _copy_xform_ops(src_prim: Usd.Prim, dst_prim: Usd.Prim) -> None:
    src_xform = UsdGeom.Xformable(src_prim)
    dst_xform = UsdGeom.Xformable(dst_prim)
    dst_xform.ClearXformOpOrder()
    for op in src_xform.GetOrderedXformOps():
        op_type = op.GetOpType()
        precision = op.GetPrecision()
        suffix = op.GetOpName().split(":")[-1] if ":" in op.GetOpName() else ""
        if suffix in {"translate", "rotateXYZ", "rotateX", "rotateY", "rotateZ", "orient", "scale", "transform"}:
            suffix = ""
        inverse = op.IsInverseOp()
        dst_op = dst_xform.AddXformOp(op_type, precision, suffix, inverse)
        value = op.Get()
        if value is not None:
            dst_op.Set(value)
    if src_xform.GetResetXformStack():
        dst_xform.SetResetXformStack(True)


def _remove_physics_apis(prim: Usd.Prim) -> None:
    for api_schema in (
        UsdPhysics.CollisionAPI,
        UsdPhysics.RigidBodyAPI,
        UsdPhysics.MassAPI,
        UsdPhysics.ArticulationRootAPI,
        UsdPhysics.FilteredPairsAPI,
    ):
        if prim.HasAPI(api_schema):
            prim.RemoveAPI(api_schema)


def _remove_physics_apis_matching_source_subtree(
    dst_stage: Usd.Stage,
    src_component: Usd.Prim,
    source_component_path: str,
    destination_component_path: str,
) -> None:
    for src_prim in Usd.PrimRange(src_component):
        if not src_prim.GetAppliedSchemas():
            continue
        src_path = str(src_prim.GetPath())
        relative = src_path.removeprefix(source_component_path).strip("/")
        dst_path = destination_component_path if not relative else f"{destination_component_path}/{relative}"
        dst_prim = dst_stage.OverridePrim(dst_path)
        _remove_physics_apis(dst_prim)


def _safe_worldbody_alias(index: int, source_name: str) -> str:
    if source_name == "_":
        suffix = "underscore"
    elif source_name.startswith("__"):
        suffix = source_name[2:] or "anonymous"
    else:
        suffix = source_name
    return f"source_worldbody_part_{index:02d}_{suffix}"


def _reference_path(output_path: Path, source_path: Path) -> str:
    return os.path.relpath(source_path.resolve(), output_path.resolve().parent)


def _define_clean_root(stage: Usd.Stage, config: dict) -> Usd.Prim:
    root = UsdGeom.Xform.Define(stage, config["root_prim"]).GetPrim()
    stage.SetDefaultPrim(root)
    _set_string_attr(root, "aloha:stageRole", config["stage"])
    _set_string_attr(root, "aloha:sourceKind", "original_aloha1_usd_visual_baseline")
    _set_string_attr(root, "aloha:sourceType", "USER_CONFIRMED")
    _set_bool_attr(root, "aloha:visualOnly", True)
    _set_bool_attr(root, "aloha:physicsEligible", False)
    _set_bool_attr(root, "aloha:collisionEligible", False)
    _set_bool_attr(root, "aloha:controllerEligible", False)
    _set_bool_attr(root, "aloha:articulationCompatible", False)
    _set_bool_attr(root, "aloha:stationaryAiRuntimeCompatible", False)
    _set_bool_attr(root, "aloha:cameraCalibrationReady", False)
    _set_bool_attr(root, "aloha:trainingEligible", False)
    root.SetCustomDataByKey("source_aloha1_usd", config["source_aloha1_usd"])
    root.SetCustomDataByKey("visual_baseline_policy", "reference original ALOHA1 visuals only; no mesh copying")
    root.SetCustomDataByKey("known_missing_measurement", "inter_base_anchor_x_distance_cm")
    return root


def _define_semantic_scope(stage: Usd.Stage, path: str, role: str, source_root: str) -> Usd.Prim:
    prim = UsdGeom.Scope.Define(stage, path).GetPrim()
    _set_string_attr(prim, "aloha:stageRole", role)
    _set_string_attr(prim, "aloha:sourceRoot", source_root)
    _set_bool_attr(prim, "aloha:visualOnly", True)
    _set_bool_attr(prim, "aloha:physicsEligible", False)
    _set_bool_attr(prim, "aloha:collisionEligible", False)
    _set_bool_attr(prim, "aloha:trainingEligible", False)
    return prim


def _add_visual_component(
    dst_stage: Usd.Stage,
    src_stage: Usd.Stage,
    source_usd_ref: str,
    source_component_path: str,
    destination_component_path: str,
    role: str,
    *,
    alias_name: str | None = None,
) -> None:
    src_component = src_stage.GetPrimAtPath(source_component_path)
    if not src_component.IsValid():
        raise RuntimeError(f"Missing source component: {source_component_path}")
    src_visuals = src_component.GetChild("visuals")
    if not src_visuals.IsValid():
        raise RuntimeError(f"Missing source visuals: {source_component_path}/visuals")

    dst_component = UsdGeom.Xform.Define(dst_stage, destination_component_path).GetPrim()
    dst_component.GetReferences().AddReference(source_usd_ref, source_component_path)
    _remove_physics_apis(dst_component)
    _set_string_attr(dst_component, "aloha:stageRole", role)
    _set_string_attr(dst_component, "aloha:sourcePrim", source_component_path)
    if alias_name:
        _set_string_attr(dst_component, "aloha:stationaryAiStyleName", alias_name)
    _set_bool_attr(dst_component, "aloha:visualOnly", True)
    _set_bool_attr(dst_component, "aloha:referenceOriginalComponentRoot", True)
    _set_bool_attr(dst_component, "aloha:physicsEligible", False)
    _set_bool_attr(dst_component, "aloha:collisionEligible", False)
    _set_bool_attr(dst_component, "aloha:trainingEligible", False)
    _remove_physics_apis_matching_source_subtree(dst_stage, src_component, source_component_path, destination_component_path)


def _add_visual_children(
    dst_stage: Usd.Stage,
    src_stage: Usd.Stage,
    source_usd_ref: str,
    source_root: str,
    destination_root: str,
    role: str,
    *,
    alias_map: dict[str, str] | None = None,
    skip_child_names: set[str] | None = None,
    worldbody_aliases: bool = False,
) -> list[str]:
    src_root = src_stage.GetPrimAtPath(source_root)
    if not src_root.IsValid():
        raise RuntimeError(f"Missing source root: {source_root}")
    dst_root = UsdGeom.Xform.Define(dst_stage, destination_root).GetPrim()
    _copy_xform_ops(src_root, dst_root)
    _set_string_attr(dst_root, "aloha:sourceRoot", source_root)
    _set_bool_attr(dst_root, "aloha:visualOnly", True)
    _set_bool_attr(dst_root, "aloha:physicsEligible", False)
    _set_bool_attr(dst_root, "aloha:collisionEligible", False)
    added = []
    skip_child_names = skip_child_names or set()
    for index, child in enumerate(src_root.GetChildren()):
        if not child.GetChild("visuals").IsValid():
            continue
        if child.GetName() in skip_child_names:
            continue
        if alias_map is not None:
            alias = alias_map.get(child.GetName(), child.GetName())
        elif worldbody_aliases:
            alias = _safe_worldbody_alias(index, child.GetName())
        else:
            alias = child.GetName()
        dst_path = f"{destination_root}/{alias}"
        _add_visual_component(dst_stage, src_stage, source_usd_ref, str(child.GetPath()), dst_path, role, alias_name=alias)
        added.append(dst_path)
    return added


def create_stage(output_path: Path, config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_path = Path(config["source_aloha1_usd"])
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    _define_clean_root(stage, config)

    source_stage = Usd.Stage.Open(str(source_path), load=Usd.Stage.LoadAll)
    if source_stage is None:
        raise RuntimeError(f"Could not open source stage: {source_path}")
    source_ref = _reference_path(output_path, source_path)

    roots = config["semantic_roots"]
    source_roots = config["source_roots"]
    support_scope = _define_semantic_scope(stage, roots["support_frame"], "A13_support_frame_semantic_alias", source_roots["support_frame"])
    support_scope.SetCustomDataByKey("primary_frame_link", roots["frame"])
    _define_semantic_scope(stage, roots["frame"], "A13_frame_link_visual_baseline", source_roots["support_frame"])
    manifest = _define_semantic_scope(stage, roots["source_manifest"], "A13_source_manifest", str(source_path))

    looks = UsdGeom.Xform.Define(stage, roots["materials_and_lights"]).GetPrim()
    looks.GetReferences().AddReference(source_ref, source_roots["looks"])
    _set_string_attr(looks, "aloha:stageRole", "A13_original_materials_and_lights_reference")
    _set_bool_attr(looks, "aloha:visualOnly", True)

    component_policy = config["component_policy"]
    _add_visual_component(
        stage,
        source_stage,
        source_ref,
        f'{source_roots["support_frame"]}/table',
        component_policy["support_frame"]["table_destination"],
        "A13_tabletop_link_visual_baseline",
        alias_name="tabletop_link",
    )
    _add_visual_component(
        stage,
        source_stage,
        source_ref,
        f'{source_roots["support_frame"]}/floor',
        component_policy["support_frame"]["floor_destination"],
        "A13_floor_reference_visual_baseline",
        alias_name="floor_reference_link",
    )
    support_paths = _add_visual_children(
        stage,
        source_stage,
        source_ref,
        source_roots["support_frame"],
        component_policy["support_frame"]["destination_root"],
        "A13_support_frame_component",
        skip_child_names={"table", "floor"},
        worldbody_aliases=True,
    )
    left_paths = _add_visual_children(
        stage,
        source_stage,
        source_ref,
        source_roots["left_robot"],
        component_policy["robot"]["destination_left_root"],
        "A13_left_robot_link",
        alias_map=LEFT_LINK_ALIASES,
    )
    right_paths = _add_visual_children(
        stage,
        source_stage,
        source_ref,
        source_roots["right_robot"],
        component_policy["robot"]["destination_right_root"],
        "A13_right_robot_link",
        alias_map=RIGHT_LINK_ALIASES,
    )

    manifest.SetCustomDataByKey("support_frame_visual_component_count", len(support_paths) + 2)
    manifest.SetCustomDataByKey("support_frame_auxiliary_visual_component_count", len(support_paths))
    manifest.SetCustomDataByKey("left_robot_visual_link_count", len(left_paths))
    manifest.SetCustomDataByKey("right_robot_visual_link_count", len(right_paths))
    manifest.SetCustomDataByKey("source_aloha1_usd", str(source_path))
    manifest.SetCustomDataByKey("source_reference_path", source_ref)
    manifest.SetCustomDataByKey("policy", config["global_policy"])
    manifest.SetCustomDataByKey("trusted_baseline", config["trusted_baseline"])
    manifest.SetCustomDataByKey("naming_policy", "Align top-level link names with Trossen stationary_ai style while preserving original ALOHA1 sourcePrim metadata.")
    manifest.SetCustomDataByKey("support_frame_paths_text", "\n".join(support_paths))
    manifest.SetCustomDataByKey("left_robot_paths_text", "\n".join(left_paths))
    manifest.SetCustomDataByKey("right_robot_paths_text", "\n".join(right_paths))

    stage.GetRootLayer().Save()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_stage(args.output, args.config)
    print(args.output)


if __name__ == "__main__":
    main()
