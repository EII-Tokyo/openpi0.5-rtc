#!/usr/bin/env python3
"""Audit the A19 clean articulation candidate stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from pxr import Usd


DEFAULT_CONFIG = Path("aloha_isaac_rebuild/configs/physical_reconstruction/stationary_style_rebuild.yaml")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    outputs = {key: Path(value) for key, value in config["outputs"].items()}
    stage_path = outputs["a19_clean_articulation_candidate"]
    mapping_path = outputs["a17_clean_articulation_mapping_plan_json"]
    output_path = outputs["a19_clean_articulation_candidate_audit_json"]
    mapping = _load_json(mapping_path)

    stage = Usd.Stage.Open(str(stage_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not open stage: {stage_path}")

    type_counts: dict[str, int] = {}
    api_counts: dict[str, int] = {}
    invalid_relationship_targets = []
    joint_paths = []
    dof_joint_paths = []
    collision_api_paths = []
    physics_scene_paths = []
    articulation_root_paths = []
    rigid_body_paths = []
    mass_api_paths = []
    candidate_false_paths = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        prim_type = prim.GetTypeName() or "Typeless"
        type_counts[prim_type] = type_counts.get(prim_type, 0) + 1
        schemas = list(prim.GetAppliedSchemas())
        for schema in schemas:
            api_counts[schema] = api_counts.get(schema, 0) + 1
        if "PhysicsCollisionAPI" in schemas:
            collision_api_paths.append(path)
        if "PhysicsArticulationRootAPI" in schemas:
            articulation_root_paths.append(path)
        if "PhysicsRigidBodyAPI" in schemas:
            rigid_body_paths.append(path)
        if "PhysicsMassAPI" in schemas:
            mass_api_paths.append(path)
        if prim_type == "PhysicsScene":
            physics_scene_paths.append(path)
        if prim_type in {"PhysicsFixedJoint", "PhysicsRevoluteJoint", "PhysicsPrismaticJoint"}:
            joint_paths.append(path)
            if prim_type in {"PhysicsRevoluteJoint", "PhysicsPrismaticJoint"}:
                dof_joint_paths.append(path)
            for rel_name in ("physics:body0", "physics:body1"):
                rel = prim.GetRelationship(rel_name)
                if not rel.IsValid():
                    continue
                for target in rel.GetTargets():
                    if not stage.GetPrimAtPath(target).IsValid():
                        invalid_relationship_targets.append(
                            {
                                "joint": path,
                                "relationship": rel_name,
                                "target": str(target),
                            }
                        )
        candidate_attr = prim.GetAttribute("aloha:candidateOnly")
        if path.startswith("/aloha") and candidate_attr.IsValid() and candidate_attr.Get() is False:
            candidate_false_paths.append(path)

    expected_joints = sorted(record["proposed_clean_joint_path"] for record in mapping["joint_records"])
    missing_joints = [path for path in expected_joints if not stage.GetPrimAtPath(path).IsValid()]
    extra_joints = sorted(set(joint_paths) - set(expected_joints))
    expected_roots = ["/aloha/follower_left_base_link", "/aloha/follower_right_base_link"]
    ok = (
        stage.GetDefaultPrim()
        and str(stage.GetDefaultPrim().GetPath()) == "/aloha"
        and len(joint_paths) == 20
        and len(dof_joint_paths) == 16
        and not missing_joints
        and not extra_joints
        and len(rigid_body_paths) == 20
        and len(mass_api_paths) == 20
        and sorted(articulation_root_paths) == expected_roots
        and not collision_api_paths
        and not physics_scene_paths
        and not invalid_relationship_targets
        and not candidate_false_paths
    )
    result = {
        "ok": bool(ok),
        "status": "PASS_A19_CLEAN_ARTICULATION_CANDIDATE_STATIC_AUDIT"
        if ok
        else "FAIL_A19_CLEAN_ARTICULATION_CANDIDATE_STATIC_AUDIT",
        "stage_path": str(stage_path),
        "default_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None,
        "candidate_only": True,
        "physics_ready": False,
        "collision_ready": False,
        "control_ready": False,
        "replay_ready": False,
        "training_eligible": False,
        "type_counts": type_counts,
        "api_counts": api_counts,
        "joint_count": len(joint_paths),
        "dof_joint_count": len(dof_joint_paths),
        "rigid_body_count": len(rigid_body_paths),
        "mass_api_count": len(mass_api_paths),
        "articulation_root_paths": sorted(articulation_root_paths),
        "collision_api_paths": collision_api_paths,
        "physics_scene_paths": physics_scene_paths,
        "missing_joints": missing_joints,
        "extra_joints": extra_joints,
        "invalid_relationship_targets": invalid_relationship_targets,
        "candidate_false_paths": candidate_false_paths,
        "next_required_gates": [
            "Isaac runtime open-stage smoke",
            "Asset Validator RobotRules/PhysicsRules",
            "articulation count and DOF readback",
            "small set-target/readback sign gate",
            "hold gates before any replay/contact/RL",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(args.config)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"type_counts", "api_counts"}
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
