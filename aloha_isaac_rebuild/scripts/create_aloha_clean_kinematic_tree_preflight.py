#!/usr/bin/env python3
"""Create the A18 clean kinematic-tree preflight report.

This is a graph audit only.  It does not author clean joints, articulation
roots, rigid bodies, drives, controllers, cameras, or a physics scene.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml


DEFAULT_CONFIG = Path("aloha_isaac_rebuild/configs/physical_reconstruction/stationary_style_rebuild.yaml")

EXPECTED_LEFT_ARM = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
]
EXPECTED_RIGHT_ARM = [
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
]


def _clean_one(values: list[str | None]) -> str | None:
    present = [value for value in values if value]
    if not present:
        return None
    if len(present) > 1:
        return "|".join(present)
    return present[0]


def _chain_status(records: list[dict], expected: list[str]) -> dict:
    by_name = {record["source_joint_name"]: record for record in records}
    missing = [name for name in expected if name not in by_name]
    edges = []
    continuity_errors = []
    previous_child = None
    for name in expected:
        record = by_name.get(name)
        if record is None:
            continue
        parent = _clean_one(record.get("clean_body0", []))
        child = _clean_one(record.get("clean_body1", []))
        edges.append({"joint": name, "parent": parent, "child": child})
        if previous_child is not None and parent != previous_child:
            continuity_errors.append(
                {
                    "joint": name,
                    "expected_parent": previous_child,
                    "actual_parent": parent,
                }
            )
        previous_child = child
    return {
        "expected": expected,
        "missing": missing,
        "edges": edges,
        "continuity_errors": continuity_errors,
        "ok": not missing and not continuity_errors,
    }


def _markdown_report(result: dict) -> str:
    lines = [
        "# A18 Clean Kinematic Tree Preflight",
        "",
        "This is a preflight report only. It does not author joints, physics, drives, controllers, replay, cameras, or RL semantics.",
        "",
        "## Status",
        "",
        "```text",
        f"status = {result['status']}",
        f"ok = {str(result['ok']).lower()}",
        f"joint_count = {result['joint_count']}",
        f"dof_joint_count = {result['dof_joint_count']}",
        f"root_joint_count = {result['root_joint_count']}",
        f"duplicate_child_parent_count = {len(result['duplicate_child_parent_links'])}",
        f"left_chain_ok = {str(result['left_arm_chain']['ok']).lower()}",
        f"right_chain_ok = {str(result['right_arm_chain']['ok']).lower()}",
        "author_articulation = false",
        "physics_ready = false",
        "```",
        "",
        "## Clean Root Joints",
        "",
    ]
    for root in result["root_joints"]:
        lines.append(f"- `{root['source_joint_name']}` anchors `{root['clean_child']}` under `/aloha`.")
    lines.extend(
        [
            "",
            "## Left Arm Chain",
            "",
            "| Joint | Parent link | Child link |",
            "| --- | --- | --- |",
        ]
    )
    for edge in result["left_arm_chain"]["edges"]:
        lines.append(f"| `{edge['joint']}` | `{edge['parent']}` | `{edge['child']}` |")
    lines.extend(
        [
            "",
            "## Right Arm Chain",
            "",
            "| Joint | Parent link | Child link |",
            "| --- | --- | --- |",
        ]
    )
    for edge in result["right_arm_chain"]["edges"]:
        lines.append(f"| `{edge['joint']}` | `{edge['parent']}` | `{edge['child']}` |")
    lines.extend(
        [
            "",
            "## Gripper Branches",
            "",
            "| Side | Base link | Finger joints |",
            "| --- | --- | --- |",
        ]
    )
    for branch in result["gripper_branches"]:
        joints = ", ".join(f"`{name}`" for name in branch["finger_joints"])
        lines.append(f"| {branch['side']} | `{branch['base_link']}` | {joints} |")
    lines.extend(
        [
            "",
            "## Important Limitation",
            "",
            "The current A15/A16 visual stages are still visual/collider-preview stages. Their links are positioned for inspection, not yet authored as a verified controllable articulation. The next implementation step must author clean joints in a new candidate layer and then run Isaac Asset Validator, set-target/readback, hold, and 50 Hz replay gates.",
        ]
    )
    return "\n".join(lines) + "\n"


def create_preflight(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    outputs = {key: Path(value) for key, value in config["outputs"].items()}
    mapping_path = outputs["a17_clean_articulation_mapping_plan_json"]
    result_json = outputs["a18_clean_kinematic_tree_preflight_json"]
    result_md = outputs["a18_clean_kinematic_tree_preflight_md"]

    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    records = mapping["joint_records"]
    dof_records = [record for record in records if record["is_dof_joint"]]

    child_parents: dict[str, list[dict]] = defaultdict(list)
    root_joints = []
    gripper_branches_by_side: dict[str, dict] = {
        "left": {"side": "left", "base_link": "/aloha/follower_left_gripper_base", "finger_joints": []},
        "right": {"side": "right", "base_link": "/aloha/follower_right_gripper_base", "finger_joints": []},
    }
    for record in records:
        parent = _clean_one(record.get("clean_body0", []))
        child = _clean_one(record.get("clean_body1", []))
        if child:
            child_parents[child].append(
                {
                    "source_joint_name": record["source_joint_name"],
                    "parent": parent,
                    "joint_type": record["joint_type"],
                }
            )
        if record["source_joint_name"].startswith("rootJoint_"):
            root_joints.append(
                {
                    "source_joint_name": record["source_joint_name"],
                    "clean_child": child,
                    "joint_type": record["joint_type"],
                }
            )
        if record["source_joint_name"] in {"left_left_finger", "left_right_finger"}:
            gripper_branches_by_side["left"]["finger_joints"].append(record["source_joint_name"])
        if record["source_joint_name"] in {"right_left_finger", "right_right_finger"}:
            gripper_branches_by_side["right"]["finger_joints"].append(record["source_joint_name"])

    duplicate_child_parent_links = [
        {"child": child, "parents": parents}
        for child, parents in sorted(child_parents.items())
        if len(parents) > 1
    ]
    left_arm_chain = _chain_status(dof_records, EXPECTED_LEFT_ARM)
    right_arm_chain = _chain_status(dof_records, EXPECTED_RIGHT_ARM)
    gripper_branches = list(gripper_branches_by_side.values())
    gripper_ok = all(len(branch["finger_joints"]) == 2 for branch in gripper_branches)
    ok = (
        mapping.get("ok") is True
        and len(records) == 20
        and len(dof_records) == 16
        and len(root_joints) == 2
        and not duplicate_child_parent_links
        and left_arm_chain["ok"]
        and right_arm_chain["ok"]
        and gripper_ok
    )
    result = {
        "ok": ok,
        "status": "PASS_CLEAN_KINEMATIC_TREE_PREFLIGHT_NOT_AUTHORED"
        if ok
        else "FAIL_CLEAN_KINEMATIC_TREE_PREFLIGHT",
        "source_mapping_plan": str(mapping_path),
        "author_articulation": False,
        "physics_ready": False,
        "training_eligible": False,
        "joint_count": len(records),
        "dof_joint_count": len(dof_records),
        "root_joint_count": len(root_joints),
        "root_joints": root_joints,
        "duplicate_child_parent_links": duplicate_child_parent_links,
        "left_arm_chain": left_arm_chain,
        "right_arm_chain": right_arm_chain,
        "gripper_branches": gripper_branches,
        "next_required_gate": "author a clean articulation candidate layer, then run Asset Validator plus set-target/readback before calling it control-ready",
    }
    result_json.parent.mkdir(parents=True, exist_ok=True)
    result_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result_md.parent.mkdir(parents=True, exist_ok=True)
    result_md.write_text(_markdown_report(result), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = create_preflight(args.config)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key
                not in {
                    "left_arm_chain",
                    "right_arm_chain",
                    "duplicate_child_parent_links",
                }
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
