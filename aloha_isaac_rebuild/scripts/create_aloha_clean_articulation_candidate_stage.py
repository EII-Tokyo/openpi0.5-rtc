#!/usr/bin/env python3
"""Author an A19 clean articulation candidate stage.

This stage is intentionally a candidate layer.  It composes the A16 home-pose
visual/collider preview as a sublayer, then authors clean rigid-body, mass,
articulation-root, joint, and drive evidence.  It does not enable collision
schemas, create a PhysicsScene, run simulation, replay HDF5, or claim readiness.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from pxr import Sdf, Usd, UsdPhysics


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_CONFIG = Path("aloha_isaac_rebuild/configs/physical_reconstruction/stationary_style_rebuild.yaml")


def _copy_attr(src_prim: Usd.Prim, dst_prim: Usd.Prim, name: str) -> None:
    attr = src_prim.GetAttribute(name)
    if not attr.IsValid():
        return
    value = attr.Get()
    if value is None:
        return
    dst_prim.CreateAttribute(name, attr.GetTypeName(), custom=attr.IsCustom()).Set(value)


def _copy_attrs_with_prefixes(src_prim: Usd.Prim, dst_prim: Usd.Prim, prefixes: tuple[str, ...]) -> None:
    for attr in src_prim.GetAuthoredAttributes():
        name = attr.GetName()
        if name.startswith(prefixes):
            _copy_attr(src_prim, dst_prim, name)


def _set_bool(prim: Usd.Prim, name: str, value: bool) -> None:
    prim.CreateAttribute(name, Sdf.ValueTypeNames.Bool).Set(value)


def _set_string(prim: Usd.Prim, name: str, value: str) -> None:
    prim.CreateAttribute(name, Sdf.ValueTypeNames.String).Set(value)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _define_joint(stage: Usd.Stage, source_stage: Usd.Stage, record: dict) -> Usd.Prim:
    joint_path = record["proposed_clean_joint_path"]
    joint_type = record["joint_type"]
    joint_prim = stage.DefinePrim(joint_path, joint_type)
    source_joint = source_stage.GetPrimAtPath(record["source_joint_path"])
    if not source_joint.IsValid():
        raise RuntimeError(f"Missing source joint: {record['source_joint_path']}")

    body0 = [path for path in record.get("clean_body0", []) if path]
    body1 = [path for path in record.get("clean_body1", []) if path]
    if body0:
        joint_prim.CreateRelationship("physics:body0").SetTargets([Sdf.Path(body0[0])])
    if body1:
        joint_prim.CreateRelationship("physics:body1").SetTargets([Sdf.Path(body1[0])])

    _copy_attrs_with_prefixes(
        source_joint,
        joint_prim,
        (
            "physics:",
            "drive:",
            "physxLimit:",
            "physxJoint:",
            "state:",
        ),
    )
    if joint_type == "PhysicsRevoluteJoint":
        UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
    elif joint_type == "PhysicsPrismaticJoint":
        UsdPhysics.DriveAPI.Apply(joint_prim, "linear")

    _set_bool(joint_prim, "aloha:candidateOnly", True)
    _set_bool(joint_prim, "aloha:controlReady", False)
    _set_string(joint_prim, "aloha:sourceJointPrim", record["source_joint_path"])
    _set_string(joint_prim, "aloha:sourceJointName", record["source_joint_name"])
    if record.get("canonical_dof_name"):
        _set_string(joint_prim, "aloha:canonicalDofName", record["canonical_dof_name"])
    return joint_prim


def create_candidate(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    outputs = {key: Path(value) for key, value in config["outputs"].items()}
    source_usd = (REPO_ROOT / config["source_aloha1_usd"]).resolve()
    base_stage_path = (REPO_ROOT / outputs["a16_home_pose_collider_preview"]).resolve()
    mapping_path = REPO_ROOT / outputs["a17_clean_articulation_mapping_plan_json"]
    tree_path = REPO_ROOT / outputs["a18_clean_kinematic_tree_preflight_json"]
    output_path = (REPO_ROOT / outputs["a19_clean_articulation_candidate"]).resolve()
    audit_path = REPO_ROOT / outputs["a19_clean_articulation_candidate_audit_json"]
    report_path = REPO_ROOT / outputs["a19_clean_articulation_candidate_md"]

    mapping = _load_json(mapping_path)
    tree = _load_json(tree_path)
    if not mapping.get("ok"):
        raise RuntimeError(f"A17 mapping plan not OK: {mapping_path}")
    if not tree.get("ok"):
        raise RuntimeError(f"A18 kinematic-tree preflight not OK: {tree_path}")

    source_stage = Usd.Stage.Open(str(source_usd), load=Usd.Stage.LoadAll)
    if source_stage is None:
        raise RuntimeError(f"Could not open source stage: {source_usd}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path))
    stage.GetRootLayer().subLayerPaths.append(os.path.relpath(base_stage_path, output_path.parent))
    root = stage.OverridePrim("/aloha")
    stage.SetDefaultPrim(root)
    _set_bool(root, "aloha:candidateOnly", True)
    _set_bool(root, "aloha:physicsReady", False)
    _set_bool(root, "aloha:collisionReady", False)
    _set_bool(root, "aloha:controlReady", False)
    _set_bool(root, "aloha:replayReady", False)
    _set_bool(root, "aloha:trainingEligible", False)
    _set_string(root, "aloha:articulationPolicy", "two_independent_follower_articulation_candidates")
    root.SetCustomDataByKey("source_mapping_plan", str(mapping_path))
    root.SetCustomDataByKey("source_kinematic_tree_preflight", str(tree_path))
    root.SetCustomDataByKey("base_visual_collider_preview", str(base_stage_path))

    source_to_clean = mapping["source_to_clean_link_map"]
    clean_to_source = {clean: source for source, clean in source_to_clean.items() if clean}
    joint_body_paths = sorted(
        {
            path
            for record in mapping["joint_records"]
            for key in ("clean_body0", "clean_body1")
            for path in record.get(key, [])
            if path
        }
    )
    body_records = []
    for clean_path in joint_body_paths:
        source_path = clean_to_source.get(clean_path)
        if not source_path:
            raise RuntimeError(f"No source body for clean link: {clean_path}")
        source_prim = source_stage.GetPrimAtPath(source_path)
        clean_prim = stage.OverridePrim(clean_path)
        if not clean_prim.IsValid():
            raise RuntimeError(f"Could not override clean body: {clean_path}")
        UsdPhysics.RigidBodyAPI.Apply(clean_prim)
        UsdPhysics.MassAPI.Apply(clean_prim)
        if clean_path in {"/aloha/follower_left_base_link", "/aloha/follower_right_base_link"}:
            UsdPhysics.ArticulationRootAPI.Apply(clean_prim)
            clean_prim.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
        _copy_attrs_with_prefixes(source_prim, clean_prim, ("physics:", "physxRigidBody:", "physxMass:"))
        _set_bool(clean_prim, "aloha:candidateOnly", True)
        _set_bool(clean_prim, "aloha:controlReady", False)
        _set_string(clean_prim, "aloha:sourceRigidBodyPrim", source_path)
        body_records.append({"clean_body": clean_path, "source_body": source_path})

    joints_root = stage.OverridePrim("/aloha/joints")
    _set_bool(joints_root, "aloha:candidateOnly", True)
    joint_records = [_define_joint(stage, source_stage, record) for record in mapping["joint_records"]]

    stage.GetRootLayer().Save()

    audit = {
        "ok": True,
        "status": "PASS_CLEAN_ARTICULATION_CANDIDATE_AUTHORED_STATIC_ONLY",
        "output_usd": str(output_path),
        "base_sublayer": str(base_stage_path),
        "source_usd": str(source_usd),
        "candidate_only": True,
        "physics_ready": False,
        "collision_ready": False,
        "control_ready": False,
        "replay_ready": False,
        "training_eligible": False,
        "articulation_policy": "two_independent_follower_articulation_candidates",
        "body_count": len(body_records),
        "joint_count": len(joint_records),
        "dof_joint_count": sum(1 for record in mapping["joint_records"] if record["is_dof_joint"]),
        "root_articulation_prim_paths": [
            "/aloha/follower_left_base_link",
            "/aloha/follower_right_base_link",
        ],
        "body_records": body_records,
        "next_required_gates": [
            "static audit of relationship targets and schema counts",
            "Isaac Asset Validator RobotRules and PhysicsRules",
            "readback DOF names and order without object contact",
            "small set-target/readback sign gate",
            "gravity-off and gravity-on hold gates",
            "50Hz qpos replay gate",
        ],
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# A19 Clean Articulation Candidate",
                "",
                "This stage authors a candidate articulation layer only. It is not physics-ready, control-ready, replay-ready, or RL-ready.",
                "",
                "```text",
                f"status = {audit['status']}",
                f"articulation_policy = {audit['articulation_policy']}",
                f"body_count = {audit['body_count']}",
                f"joint_count = {audit['joint_count']}",
                f"dof_joint_count = {audit['dof_joint_count']}",
                "collision_ready = false",
                "control_ready = false",
                "```",
                "",
                "Root articulation candidates:",
                "",
                "- `/aloha/follower_left_base_link`",
                "- `/aloha/follower_right_base_link`",
                "",
                "Next gates must be Asset Validator, DOF readback, small set-target/readback, hold, and 50 Hz replay. Do not use this stage for contact or RL yet.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = create_candidate(args.config)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"body_records", "next_required_gates"}
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
