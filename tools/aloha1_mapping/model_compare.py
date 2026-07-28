"""Semantic comparison between ALOHA-specific and standard Interbotix URDFs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

JOINT_FIELDS = (
    "type",
    "parent",
    "child",
    "axis",
    "origin_xyz",
    "origin_rpy",
    "lower",
    "upper",
    "effort",
    "velocity",
    "mimic_parent",
    "mimic_multiplier",
    "mimic_offset",
)


def _normalized_links(audit: Mapping[str, Any]) -> list[str]:
    robot_name = str(audit.get("robot_name", ""))
    prefix = f"{robot_name}_"
    return [
        name.removeprefix(prefix)
        for name in audit.get("link_order", [])
    ]


def _mesh_hashes(audit: Mapping[str, Any]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for mesh in audit.get("meshes", []):
        path = mesh.get("resolved_path")
        digest = mesh.get("sha256")
        if path and digest:
            result.setdefault(Path(path).name, set()).add(str(digest))
    return result


def _joint_field_value(
    audit: Mapping[str, Any],
    joint: Mapping[str, Any],
    field: str,
) -> Any:
    value = joint.get(field)
    if field in {"parent", "child"} and isinstance(value, str):
        return value.removeprefix(f"{audit.get('robot_name', '')}_")
    return value


def compare_urdf_audits(
    aloha: Mapping[str, Any],
    standard: Mapping[str, Any],
) -> dict[str, Any]:
    aloha_joints = {
        item["name"]: item for item in aloha.get("joints", [])
    }
    standard_joints = {
        item["name"]: item for item in standard.get("joints", [])
    }
    joint_differences: list[dict[str, Any]] = []
    for name in sorted(set(aloha_joints) | set(standard_joints)):
        aloha_joint = aloha_joints.get(name)
        standard_joint = standard_joints.get(name)
        if aloha_joint is None or standard_joint is None:
            joint_differences.append(
                {
                    "joint": name,
                    "fields": {
                        "presence": {
                            "aloha": aloha_joint is not None,
                            "standard": standard_joint is not None,
                        }
                    },
                }
            )
            continue
        fields = {
            field: {
                "aloha": _joint_field_value(aloha, aloha_joint, field),
                "standard": _joint_field_value(
                    standard,
                    standard_joint,
                    field,
                ),
            }
            for field in JOINT_FIELDS
            if _joint_field_value(aloha, aloha_joint, field)
            != _joint_field_value(standard, standard_joint, field)
        }
        if fields:
            joint_differences.append({"joint": name, "fields": fields})

    aloha_meshes = _mesh_hashes(aloha)
    standard_meshes = _mesh_hashes(standard)
    mesh_differences: list[dict[str, Any]] = []
    for name in sorted(set(aloha_meshes) | set(standard_meshes)):
        aloha_hashes = sorted(aloha_meshes.get(name, set()))
        standard_hashes = sorted(standard_meshes.get(name, set()))
        if aloha_hashes != standard_hashes:
            mesh_differences.append(
                {
                    "mesh": name,
                    "aloha_sha256": (
                        aloha_hashes[0]
                        if len(aloha_hashes) == 1
                        else aloha_hashes
                    ),
                    "standard_sha256": (
                        standard_hashes[0]
                        if len(standard_hashes) == 1
                        else standard_hashes
                    ),
                }
            )
    gripper_differences = [
        item
        for item in joint_differences
        if "finger" in item["joint"].lower()
        or "gripper" in item["joint"].lower()
    ]
    return {
        "aloha_robot": aloha.get("robot_name"),
        "standard_robot": standard.get("robot_name"),
        "link_order_equal": _normalized_links(aloha)
        == _normalized_links(standard),
        "joint_order_equal": aloha.get("joint_order")
        == standard.get("joint_order"),
        "joint_differences": joint_differences,
        "gripper_joint_differences": gripper_differences,
        "mesh_hash_differences": mesh_differences,
        "all_mesh_hashes_equal": not mesh_differences,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aloha-audit", type=Path, required=True)
    parser.add_argument("--standard-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    aloha_report = json.loads(arguments.aloha_audit.read_text())
    standard_report = json.loads(arguments.standard_audit.read_text())
    aloha_generation = json.loads(
        arguments.aloha_audit.with_name("urdf_generation_manifest.json").read_text()
    )
    standard_generation = json.loads(
        arguments.standard_audit.with_name(
            "urdf_generation_manifest.json"
        ).read_text()
    )
    aloha_by_name = {
        item["robot_name"]: item for item in aloha_report["robots"]
    }
    standard_by_name = {
        item["robot_name"]: item for item in standard_report["robots"]
    }
    aloha_sources = {
        item["robot"]: item for item in aloha_generation["records"]
    }
    standard_sources = {
        item["robot"]: item for item in standard_generation["records"]
    }
    pairs = [
        ("follower_left", "standard_vx300s"),
        ("leader_left", "standard_wx250s"),
    ]
    output = {
        "schema_version": 1,
        "status": "PASS",
        "comparisons": [],
    }
    for aloha_name, standard_name in pairs:
        comparison = compare_urdf_audits(
            aloha_by_name[aloha_name],
            standard_by_name[standard_name],
        )
        aloha_source = aloha_sources[aloha_name]
        standard_source = standard_sources[standard_name]
        comparison["xacro_source"] = {
            "aloha_path": aloha_source["source_xacro"],
            "aloha_sha256": aloha_source["source_xacro_sha256"],
            "standard_path": standard_source["source_xacro"],
            "standard_sha256": standard_source["source_xacro_sha256"],
            "content_equal": (
                aloha_source["source_xacro_sha256"]
                == standard_source["source_xacro_sha256"]
            ),
        }
        output["comparisons"].append(comparison)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
