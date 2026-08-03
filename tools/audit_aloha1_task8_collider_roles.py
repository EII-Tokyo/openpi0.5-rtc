#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.aloha1_mapping.task8_collider_lod import classify_link_role
from tools.aloha1_mapping.task8_collider_lod import select_throughput_links

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_REPORT = ROOT / "reports/aloha1_mapping/aloha1_cad_derived_collider_geometry.json"
STATIC_REPORT = ROOT / "reports/aloha1_mapping/aloha1_cad_derived_collision_replan_static.json"
SWEPT_REPORT = ROOT / "reports/aloha1_mapping/aloha1_cad_derived_five_pose_swept_collision.json"
BASELINE_REPORT = ROOT / "reports/aloha1_mapping/aloha1_task8_baseline_inventory.json"
OUTPUT_JSON = ROOT / "reports/aloha1_mapping/aloha1_task8_collider_roles.json"
OUTPUT_MD = ROOT / "reports/aloha1_mapping/aloha1_task8_collider_roles.md"

LINK_SUFFIXES = (
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "upper_forearm_link",
    "lower_forearm_link",
    "wrist_link",
    "gripper_link",
    "gripper_bar_link",
    "gripper_prop_link",
    "left_finger_link",
    "right_finger_link",
    "ee_arm_link",
    "ee_gripper_link",
    "fingers_link",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report() -> dict[str, Any]:
    geometry = _load(GEOMETRY_REPORT)
    static = _load(STATIC_REPORT)
    swept = _load(SWEPT_REPORT)
    baseline = _load(BASELINE_REPORT)
    geometry_by_link = {
        (str(record["robot"]), str(record["link_suffix"])): record
        for record in geometry["physical_link_records"]
    }
    collider_paths = [
        str(record["path"])
        for record in baseline["meshes"]
        if record["is_collision"]
    ]
    records = []
    selection_inputs = []
    for robot in ("follower_left", "follower_right"):
        for suffix in LINK_SUFFIXES:
            link_token = f"/{robot}_{suffix}/"
            paths = sorted(path for path in collider_paths if link_token in path)
            source = geometry_by_link.get((robot, suffix))
            has_collider = bool(paths)
            role = classify_link_role(suffix, has_collider=has_collider)
            source_status = str(source["status"]) if source else "NOT_MAPPED_AS_PHYSICAL_CAD_LINK"
            source_piece_count = int(source.get("convex_piece_count", 0)) if source else 0
            record = {
                "robot": robot,
                "link_suffix": suffix,
                "urdf_link_name": f"{robot}_{suffix}",
                "role": role,
                "role_evidence": (
                    "Bottle/finger grasp contact and gripper internal linkage"
                    if role == "task_contact_critical"
                    else (
                        "Legal-range and accepted-trajectory environment clearance"
                        if role == "environment_clearance_critical"
                        else (
                            "Geometry-free helper frame"
                            if role == "non_contact_visual_only"
                            else "No complete collision-role evidence"
                        )
                    )
                ),
                "active_collider_paths": paths,
                "active_authored_collider_mesh_count": len(paths),
                "source_cad_status": source_status,
                "source_object": source.get("source_object") if source else None,
                "source_convex_piece_count": source_piece_count,
                "source_vertex_count": source.get("vertex_count") if source else None,
                "source_triangle_count": source.get("triangle_count") if source else None,
                "source_aabb_link_local_m": source.get("aabb_link_local_m") if source else None,
                "source_brep_volume_mm3": source.get("brep_volume_mm3") if source else None,
                "source_transform_determinant": source.get("transform_determinant") if source else None,
                "unit_conversion_mm_to_m": source.get("unit_conversion_mm_to_m") if source else None,
                "baseline_static_audit": static["status"],
                "baseline_static_pose_count": static["summary"]["pose_count"],
                "baseline_swept_audit": swept["status"],
                "baseline_swept_waypoint_count": swept["summary"]["total_waypoint_count"],
                "allowed_simplification": (
                    "SINGLE_HULL_CANDIDATE_PENDING_CANDIDATE_STATIC_AND_SWEPT_REGRESSION"
                    if suffix == "upper_arm_link" and source_piece_count > 1
                    else "NONE"
                ),
                "known_limitation": (
                    "Single hull fills gaps between four supplier-CAD components; deviation and early-contact risk must be measured."
                    if suffix == "upper_arm_link"
                    else (
                        "Invalid supplier wrist B-Rep; baseline fallback must remain unchanged."
                        if suffix == "wrist_link"
                        else None
                    )
                ),
            }
            records.append(record)
            selection_inputs.append(
                {
                    "link_suffix": suffix,
                    "role": role,
                    "source_convex_piece_count": source_piece_count,
                    "source_brep_valid": source_status == "PASS",
                    "baseline_static_audit": static["status"],
                    "baseline_swept_audit": swept["status"],
                }
            )

    selected = select_throughput_links(selection_inputs)
    selected = sorted(set(selected))
    status = "PASS" if selected == ["upper_arm_link"] else "PARTIAL"
    return {
        "schema_version": 1,
        "status": status,
        "classification": "CONTACT_AWARE_COLLIDER_ROLE_AUDIT",
        "inputs": [
            {"absolute_path": str(path.resolve()), "sha256": _sha256(path)}
            for path in (GEOMETRY_REPORT, STATIC_REPORT, SWEPT_REPORT, BASELINE_REPORT)
        ],
        "evidence_boundary": {
            "cad_geometry": "supplier CAD plus deterministic project-local tessellation",
            "accepted_static_pose_count": static["summary"]["pose_count"],
            "accepted_swept_waypoint_count": swept["summary"]["total_waypoint_count"],
            "grasp_success_used_to_fit_geometry": False,
        },
        "role_counts": dict(sorted(Counter(record["role"] for record in records).items())),
        "records": records,
        "throughput_candidate_link_suffixes": selected,
        "protected_task_contact_link_suffixes": sorted(
            {record["link_suffix"] for record in records if record["role"] == "task_contact_critical"}
        ),
        "final_or_default_asset_modified": False,
        "candidate_promoted": False,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ALOHA1 Task 8 collider role audit",
        "",
        f"- Status: `{report['status']}`",
        f"- Candidate link suffixes: `{', '.join(report['throughput_candidate_link_suffixes'])}`",
        f"- Baseline static poses: `{report['evidence_boundary']['accepted_static_pose_count']}`",
        f"- Baseline swept waypoints: `{report['evidence_boundary']['accepted_swept_waypoint_count']}`",
        "- Candidate promoted: `false`",
        "",
        "| Robot/link | Role | Source pieces | Active meshes | Simplification |",
        "|---|---|---:|---:|---|",
    ]
    lines.extend(
        (
            f"| {record['urdf_link_name']} | {record['role']} | "
            f"{record['source_convex_piece_count']} | "
            f"{record['active_authored_collider_mesh_count']} | "
            f"{record['allowed_simplification']} |"
        )
        for record in report["records"]
    )
    lines.extend(
        [
            "",
            "Only the two `upper_arm_link` instances enter the diagnostic candidate. "
            "Their four supplier-CAD components may be represented by one outer convex hull "
            "only if candidate static and swept collision regressions pass. Gripper, fingers, "
            "Bottle500 and the tabletop support region are unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    report = build_report()
    OUTPUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUTPUT_MD.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "candidate_links": report["throughput_candidate_link_suffixes"],
                "records": len(report["records"]),
                "output": str(OUTPUT_JSON.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
