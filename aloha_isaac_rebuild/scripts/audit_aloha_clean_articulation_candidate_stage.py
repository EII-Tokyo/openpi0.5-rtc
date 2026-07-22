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
    visual_or_collider_relationship_targets = []
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
                    if "/visuals" in str(target) or "/colliders" in str(target):
                        visual_or_collider_relationship_targets.append(
                            {
                                "joint": path,
                                "relationship": rel_name,
                                "target": str(target),
                            }
                        )
        candidate_attr = prim.GetAttribute("aloha:candidateOnly")
        if path.startswith("/aloha") and candidate_attr.IsValid() and candidate_attr.Get() is False:
            candidate_false_paths.append(path)

    source_joint_paths = sorted(record["proposed_clean_joint_path"] for record in mapping["joint_records"])
    expected_joints = sorted(["/aloha/root_joint", *source_joint_paths])
    missing_joints = [path for path in expected_joints if not stage.GetPrimAtPath(path).IsValid()]
    extra_joints = sorted(set(joint_paths) - set(expected_joints))
    expected_root = "/aloha/root_joint"
    root_joint_prim = stage.GetPrimAtPath(expected_root)
    root_joint_body0 = (
        [str(path) for path in root_joint_prim.GetRelationship("physics:body0").GetTargets()]
        if root_joint_prim.IsValid() and root_joint_prim.GetRelationship("physics:body0").IsValid()
        else []
    )
    root_joint_body1 = (
        [str(path) for path in root_joint_prim.GetRelationship("physics:body1").GetTargets()]
        if root_joint_prim.IsValid() and root_joint_prim.GetRelationship("physics:body1").IsValid()
        else []
    )
    base_articulation_root_leaks = [
        path
        for path in ("/aloha/follower_left_base_link", "/aloha/follower_right_base_link")
        if "PhysicsArticulationRootAPI" in list(stage.GetPrimAtPath(path).GetAppliedSchemas())
    ]
    root = stage.GetPrimAtPath("/aloha")
    robot_link_targets = (
        [str(path) for path in root.GetRelationship("isaac:physics:robotLinks").GetTargets()]
        if root.IsValid() and root.GetRelationship("isaac:physics:robotLinks").IsValid()
        else []
    )
    robot_joint_targets = (
        [str(path) for path in root.GetRelationship("isaac:physics:robotJoints").GetTargets()]
        if root.IsValid() and root.GetRelationship("isaac:physics:robotJoints").IsValid()
        else []
    )
    bad_robot_link_targets = [path for path in robot_link_targets if "/visuals" in path or "/colliders" in path]
    bad_robot_joint_targets = [path for path in robot_joint_targets if "/visuals" in path or "/colliders" in path]
    expected_link_targets = sorted(
        {
            path
            for record in mapping["joint_records"]
            for key in ("clean_body0", "clean_body1")
            for path in record.get(key, [])
            if path
        }
        | {"/aloha/tabletop_link"}
    )

    graph_edges = []
    duplicate_child_parents: dict[str, list[str]] = {}
    child_to_parent: dict[str, str] = {}
    reparented_joints = []
    for joint_path in source_joint_paths:
        joint_prim = stage.GetPrimAtPath(joint_path)
        body0_targets = [str(path) for path in joint_prim.GetRelationship("physics:body0").GetTargets()]
        body1_targets = [str(path) for path in joint_prim.GetRelationship("physics:body1").GetTargets()]
        if not body0_targets or not body1_targets:
            continue
        body0 = body0_targets[0]
        body1 = body1_targets[0]
        graph_edges.append((body0, body1, joint_path))
        if body1 in child_to_parent:
            duplicate_child_parents.setdefault(body1, [child_to_parent[body1]]).append(body0)
        else:
            child_to_parent[body1] = body0
        attr = joint_prim.GetAttribute("aloha:intentionalStationaryAiStyleReparenting")
        if attr.IsValid() and attr.Get():
            reparented_joints.append(joint_path)

    adjacency: dict[str, list[str]] = {path: [] for path in expected_link_targets}
    for parent, child, _joint in graph_edges:
        adjacency.setdefault(parent, []).append(child)
    reachable = []
    queue = ["/aloha/tabletop_link"]
    seen = set()
    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        reachable.append(current)
        queue.extend(adjacency.get(current, []))
    unreachable_links = sorted(set(expected_link_targets) - seen)

    ok = (
        stage.GetDefaultPrim()
        and str(stage.GetDefaultPrim().GetPath()) == "/aloha"
        and len(joint_paths) == 21
        and len(dof_joint_paths) == 16
        and not missing_joints
        and not extra_joints
        and len(rigid_body_paths) == 21
        and len(mass_api_paths) == 21
        and sorted(articulation_root_paths) == [expected_root]
        and root_joint_prim.GetTypeName() == "PhysicsFixedJoint"
        and root_joint_body0 == []
        and root_joint_body1 == ["/aloha/tabletop_link"]
        and not base_articulation_root_leaks
        and sorted(robot_link_targets) == expected_link_targets
        and sorted(robot_joint_targets) == source_joint_paths
        and not bad_robot_link_targets
        and not bad_robot_joint_targets
        and not collision_api_paths
        and not physics_scene_paths
        and not invalid_relationship_targets
        and not visual_or_collider_relationship_targets
        and not candidate_false_paths
        and sorted(reparented_joints)
        == ["/aloha/joints/rootJoint_left_base_link", "/aloha/joints/rootJoint_right_base_link"]
        and not duplicate_child_parents
        and not unreachable_links
        and len(graph_edges) == len(expected_link_targets) - 1
    )
    result = {
        "ok": bool(ok),
        "status": "PASS_A19_SINGLE_ROOT_ARTICULATION_CANDIDATE_AUTHORED_NO_COLLISION_NO_RUNTIME_READY"
        if ok
        else "FAIL_A19_SINGLE_ROOT_ARTICULATION_CANDIDATE_STATIC_AUDIT",
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
        "root_joint_path": expected_root,
        "root_joint_type": root_joint_prim.GetTypeName() if root_joint_prim.IsValid() else None,
        "root_joint_body0": root_joint_body0,
        "root_joint_body1": root_joint_body1,
        "base_articulation_root_leaks": base_articulation_root_leaks,
        "robot_link_target_count": len(robot_link_targets),
        "robot_joint_target_count": len(robot_joint_targets),
        "bad_robot_link_targets": bad_robot_link_targets,
        "bad_robot_joint_targets": bad_robot_joint_targets,
        "expected_link_target_count": len(expected_link_targets),
        "visual_or_collider_relationship_targets": visual_or_collider_relationship_targets,
        "reparented_joints": sorted(reparented_joints),
        "duplicate_child_parents": duplicate_child_parents,
        "unreachable_links_from_tabletop": unreachable_links,
        "graph_edge_count": len(graph_edges),
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
