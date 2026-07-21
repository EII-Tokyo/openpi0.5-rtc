from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE_USD = (
    REPO_ROOT
    / "reports/aloha1_isaac_adaptation/episode19_dynamic_bottle_grasp_gate_20260721_fast_tracking_v2/"
    / "debug_stage_after_object_placement.usda"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "reports/aloha1_isaac_adaptation/episode19_dynamic_bottle_grasp_gate_20260721_fast_tracking_v2_collider_baseline"
)
DEFAULT_COMMAND_SPIKE_REPORT = (
    REPO_ROOT
    / "reports/aloha1_isaac_adaptation/episode19_dynamic_bottle_grasp_gate_20260721_fast_tracking_v2_command_spike_feasibility/"
    / "command_spike_feasibility.json"
)

COMPLEX_MESH_POINT_THRESHOLD = 128
COMPLEX_MESH_FACE_THRESHOLD = 128


def _rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        return list(value)
    except Exception:
        return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n")


def _applied(prim: Any) -> list[str]:
    return [str(item) for item in prim.GetAppliedSchemas()]


def _has_schema(prim: Any, schema_name: str) -> bool:
    return schema_name in _applied(prim)


def _nearest_schema_ancestor(prim: Any, schema_name: str) -> str | None:
    current = prim.GetParent()
    while current and current.IsValid():
        if _has_schema(current, schema_name):
            return str(current.GetPath())
        current = current.GetParent()
    return None


def _attr_value(prim: Any, name: str) -> Any:
    attr = prim.GetAttribute(name)
    if not attr:
        return None
    return attr.Get()


def _attrs_matching(prim: Any, needle: str) -> dict[str, Any]:
    needle = needle.lower()
    result: dict[str, Any] = {}
    for attr in prim.GetAttributes():
        name = attr.GetName()
        if needle in name.lower():
            result[name] = attr.Get()
    return result


def _mesh_stats(prim: Any) -> dict[str, Any]:
    from pxr import UsdGeom

    if prim.GetTypeName() != "Mesh":
        return {"mesh_points": None, "mesh_faces": None}
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get() or []
    faces = mesh.GetFaceVertexCountsAttr().Get() or []
    return {"mesh_points": len(points), "mesh_faces": len(faces)}


def classify_path(path: str) -> str:
    lower = path.lower()
    if "officeenvironment" in lower or "official_props" in lower or "warehouse" in lower:
        return "workcell_or_environment"
    if "support_base" in lower or "base_placeholder" in lower:
        return "workcell_or_environment"
    if "phase43_passive_contact_cube" in lower or "/world/bottle500" in lower:
        return "bottle"
    if "/world/bottle" in lower or "bottle" in lower:
        return "bottle"
    if "pipe" in lower or "water" in lower or "tube" in lower:
        return "pipe"
    if "finger" in lower or "gripper" in lower or "_g0" in lower or "_g1" in lower:
        return "finger"
    if "/scene/" in lower and ("left_" in lower or "right_" in lower):
        return "robot_link"
    if "/world/" in lower or "table" in lower or "workcell" in lower or "colliders" in lower:
        return "workcell_or_environment"
    return "other"


def collider_shape_family(type_name: str, approximation: str | None, mesh_points: int | None, mesh_faces: int | None) -> str:
    primitive = {"Cube", "Sphere", "Capsule", "Cylinder", "Cone"}
    if type_name in primitive:
        return type_name.lower()
    if type_name == "Plane":
        return "plane"
    if type_name != "Mesh":
        return "xform_or_container"
    approx = (approximation or "").lower()
    if approx in {"convexhull", "convex hull"}:
        return "mesh_convex_hull"
    if "convex" in approx and "decomposition" in approx:
        return "mesh_convex_decomposition"
    if "sdf" in approx:
        return "mesh_sdf"
    if mesh_points is not None and mesh_faces is not None:
        if mesh_points > COMPLEX_MESH_POINT_THRESHOLD or mesh_faces > COMPLEX_MESH_FACE_THRESHOLD:
            return "complex_mesh_unspecified"
        return "small_mesh_unspecified"
    return "mesh_unspecified"


def baseline_findings(row: dict[str, Any]) -> list[str]:
    if row.get("collision_enabled") is False:
        return []
    category = row["category"]
    family = row["shape_family"]
    findings: list[str] = []
    is_dynamic = bool(row["has_rigid_body_api"] or row["rigid_body_ancestor"])
    is_complex_mesh = family == "complex_mesh_unspecified"
    is_visual_mesh_like = row["type_name"] == "Mesh" and is_complex_mesh and not row["approximation"]

    if category in {"robot_link", "finger"} and is_dynamic and is_visual_mesh_like:
        findings.append("dynamic_robot_collision_uses_complex_mesh_without_explicit_approximation")
    if category == "bottle" and family not in {
        "capsule",
        "cylinder",
        "sphere",
        "cube",
        "mesh_convex_hull",
        "mesh_convex_decomposition",
        "xform_or_container",
    }:
        findings.append("bottle_collider_not_simple_or_convex_baseline")
    if category == "pipe" and family not in {
        "capsule",
        "cylinder",
        "mesh_convex_hull",
        "mesh_convex_decomposition",
        "mesh_sdf",
        "xform_or_container",
    }:
        findings.append("pipe_precision_collider_not_convex_or_sdf_baseline")
    if is_dynamic and row["type_name"] == "Mesh" and family in {"complex_mesh_unspecified", "mesh_unspecified"}:
        findings.append("dynamic_mesh_collision_requires_explicit_supported_approximation_review")
    if category in {"finger", "bottle"} and row["bbox_max_dim"] is not None and row["bbox_max_dim"] < 0.01:
        findings.append("small_fast_contact_part_ccd_candidate_after_geometry_review")
    return findings


def baseline_status(rows: list[dict[str, Any]]) -> str:
    hard_findings = {
        "dynamic_robot_collision_uses_complex_mesh_without_explicit_approximation",
        "bottle_collider_not_simple_or_convex_baseline",
        "pipe_precision_collider_not_convex_or_sdf_baseline",
        "dynamic_mesh_collision_requires_explicit_supported_approximation_review",
    }
    active_rows = [row for row in rows if row.get("collision_enabled") is not False]
    if any(hard_findings.intersection(row["findings"]) for row in active_rows):
        return "NEEDS_COLLIDER_BASELINE_REPAIR"
    return "PASS_BASELINE_GEOMETRY_REVIEW"


def _bool_attr_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes", "on"}
    return bool(value)


def ccd_policy_from_evidence(
    *,
    baseline_status_value: str,
    active_rows: list[dict[str, Any]],
    command_spike_report: dict[str, Any] | None,
) -> dict[str, Any]:
    ccd_enabled_paths: list[str] = []
    ccd_candidates: list[dict[str, Any]] = []
    advisory_reasons: list[str] = []
    blockers: list[str] = []

    for row in active_rows:
        ccd_attrs = row.get("ccd_attrs") or {}
        if any(_bool_attr_enabled(value) for value in ccd_attrs.values()):
            ccd_enabled_paths.append(row["path"])
        if "small_fast_contact_part_ccd_candidate_after_geometry_review" in row.get("findings", []):
            ccd_candidates.append(
                {
                    "path": row["path"],
                    "category": row["category"],
                    "reason": "thin_or_small_contact_part_after_geometry_review",
                }
            )

    command_classification = None
    if command_spike_report is not None:
        command_classification = command_spike_report.get("failure_classification")
        if command_classification in {
            "SINGLE_SPIKE_RESIDUAL",
            "REPEATED_SPIKE_CLUSTER",
            "GLOBAL_HIGH_RATE_COMMAND_MISMATCH",
            "CONTACT_LOADED_TRACKING_RESIDUAL",
        }:
            blockers.append("COMMAND_SMOOTHNESS_NOT_VERIFIED")
            advisory_reasons.append("command_spike_without_tunneling")

    if baseline_status_value != "PASS_BASELINE_GEOMETRY_REVIEW":
        blockers.append("COLLIDER_SEMANTICS_NOT_VERIFIED")
        recommendation = "CCD_NOT_ALLOWED_BAD_COLLIDER_BASELINE"
    elif "COMMAND_SMOOTHNESS_NOT_VERIFIED" in blockers:
        recommendation = "FIX_COMMAND_TARGET_CONTINUITY_BEFORE_CCD"
    elif ccd_candidates:
        recommendation = "CCD_LOCAL_DIAGNOSTIC_ONLY"
        advisory_reasons.append("small_or_thin_contact_candidate_without_tunneling_proof")
    else:
        recommendation = "CCD_NOT_NEEDED"
        advisory_reasons.append("not_needed_low_speed_contact")

    return {
        "default_ccd_policy": "off_by_default",
        "do_not_tune_ccd_before_collider_and_command_smoothness": True,
        "ccd_enabled_any": bool(ccd_enabled_paths),
        "active_ccd_body_count": len(ccd_enabled_paths),
        "active_ccd_bodies": ccd_enabled_paths,
        "ccd_candidate_count": len(ccd_candidates),
        "ccd_candidates": ccd_candidates,
        "ccd_advisory_reasons": advisory_reasons,
        "ccd_blockers_before_recommendation": blockers,
        "command_spike_classification": command_classification,
        "ccd_recommendation": recommendation,
        "ccd_scope": "specific_bodies_only",
        "ccd_required_for_pass": False,
        "notes": [
            "This is a recommendation from current evidence, not a CCD-enabled validation result.",
            "CCD is not enabled by this audit script.",
        ],
    }


def _category_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    category_summary: dict[str, Any] = {}
    for category in sorted({row["category"] for row in rows}):
        category_rows = [row for row in rows if row["category"] == category]
        category_summary[category] = {
            "count": len(category_rows),
            "shape_families": dict(Counter(row["shape_family"] for row in category_rows)),
            "findings": dict(Counter(finding for row in category_rows for finding in row["findings"])),
        }
    return category_summary


def _bbox_row(cache: Any, prim: Any) -> dict[str, Any]:
    box = cache.ComputeWorldBound(prim).GetBox()
    if box.IsEmpty():
        return {"bbox_valid": False, "bbox_min": None, "bbox_max": None, "bbox_size": None, "bbox_max_dim": None}
    min_pt = box.GetMin()
    max_pt = box.GetMax()
    size = [float(max_pt[i] - min_pt[i]) for i in range(3)]
    return {
        "bbox_valid": True,
        "bbox_min": [float(min_pt[i]) for i in range(3)],
        "bbox_max": [float(max_pt[i]) for i in range(3)],
        "bbox_size": size,
        "bbox_max_dim": max(size) if size else 0.0,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    active_summary = summary["active_category_summary"]
    ccd_policy = summary["ccd_policy"]
    lines = [
        "# ALOHA1 Replay Collider Baseline Audit",
        "",
        f"- status: `{payload['status']}`",
        f"- baseline status: `{summary['baseline_status']}`",
        f"- stage: `{payload['inputs']['stage_usd']}`",
        f"- collider rows: `{summary['collision_prim_count']}`",
        f"- active collider rows: `{summary['active_collision_prim_count']}`",
        f"- complex mesh candidates: `{summary['complex_mesh_candidate_count']}`",
        f"- dynamic mesh review candidates: `{summary['dynamic_mesh_review_count']}`",
        f"- CCD advisory candidates: `{summary['ccd_advisory_count']}`",
        f"- CCD recommendation: `{ccd_policy['ccd_recommendation']}`",
        "",
        "## Why This Gate Exists",
        "",
        "Collider geometry is a prerequisite for meaningful physics tuning. Increasing physics frequency or enabling CCD can reduce tunneling, but it cannot make an incorrect collision shape semantically correct.",
        "",
        "Baseline rule used here:",
        "",
        "- robot links: primitive, bbox/capsule-like, or explicit convex approximation;",
        "- bottle body: cylinder/capsule or a small number of convex pieces;",
        "- bottle mouth and pipe: convex decomposition or SDF only where precision is needed;",
        "- CCD: only an advisory after collider shape is correct, especially for small fast gripper/bottle parts.",
        "",
        "## CCD Policy",
        "",
        f"- default CCD policy: `{ccd_policy['default_ccd_policy']}`",
        f"- active CCD body count: `{ccd_policy['active_ccd_body_count']}`",
        f"- active CCD bodies: `{ccd_policy['active_ccd_bodies']}`",
        f"- CCD candidate count: `{ccd_policy['ccd_candidate_count']}`",
        f"- CCD blockers before recommendation: `{ccd_policy['ccd_blockers_before_recommendation']}`",
        f"- command spike classification: `{ccd_policy['command_spike_classification']}`",
        f"- recommendation: `{ccd_policy['ccd_recommendation']}`",
        "",
        "This section is a recommendation from current evidence, not a CCD-enabled validation result. This audit does not enable CCD or modify the stage.",
        "",
        "## Active Category Summary",
        "",
        "| category | count | shape families | findings |",
        "| --- | ---: | --- | --- |",
    ]
    for category, row in sorted(active_summary.items()):
        families = ", ".join(f"{key}:{value}" for key, value in sorted(row["shape_families"].items()))
        findings = ", ".join(f"{key}:{value}" for key, value in sorted(row["findings"].items())) or "-"
        lines.append(f"| {category} | {row['count']} | {families} | {findings} |")

    lines.extend(
        [
            "",
            "## Highest Priority Findings",
            "",
            "| category | path | type | approximation | mesh points/faces | finding |",
            "| --- | --- | --- | --- | ---: | --- |",
        ]
    )
    flagged = [row for row in payload["collision_rows"] if row["collision_enabled"] is not False and row["findings"]]
    for row in flagged[:40]:
        mesh = "-"
        if row["mesh_points"] is not None or row["mesh_faces"] is not None:
            mesh = f"{row['mesh_points']}/{row['mesh_faces']}"
        lines.append(
            f"| {row['category']} | `{row['path']}` | {row['type_name']} | "
            f"{row['approximation'] or '-'} | {mesh} | {', '.join(row['findings'])} |"
        )
    if len(flagged) > 40:
        lines.append(f"| ... | ... | ... | ... | ... | {len(flagged) - 40} more rows in JSON |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit collider shape baseline for an ALOHA1 replay debug stage.")
    parser.add_argument("--stage-usd", default=str(DEFAULT_STAGE_USD))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--command-spike-report",
        default=str(DEFAULT_COMMAND_SPIKE_REPORT),
        help="Optional read-only command-spike report JSON. If present, it can block CCD recommendation.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    json_path = output_dir / "collider_baseline_audit.json"
    md_path = output_dir / "collider_baseline_audit.md"
    payload: dict[str, Any] = {
        "status": "STARTED",
        "real_robot_touched": False,
        "stage_saved": False,
        "inputs": {"stage_usd": _rel(args.stage_usd), "command_spike_report": _rel(args.command_spike_report)},
    }
    _write_json(json_path, payload)

    try:
        from isaacsim import SimulationApp

        app_config = dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG)
        app_config["fast_shutdown"] = False
        _app = SimulationApp(app_config)
        import isaacsim.core.utils.stage as stage_utils
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

        stage_utils.open_stage(str(Path(args.stage_usd).resolve()))
        stage = stage_utils.get_current_stage()
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=False,
        )

        rows: list[dict[str, Any]] = []
        for prim in stage.Traverse():
            if not _has_schema(prim, "PhysicsCollisionAPI"):
                continue
            path = str(prim.GetPath())
            mesh_stats = _mesh_stats(prim)
            mesh_collision = UsdPhysics.MeshCollisionAPI(prim)
            approximation = None
            if mesh_collision:
                approximation = mesh_collision.GetApproximationAttr().Get()
            rigid_body_ancestor = _nearest_schema_ancestor(prim, "PhysicsRigidBodyAPI")
            ccd_attrs = _attrs_matching(prim, "ccd")
            if not ccd_attrs and rigid_body_ancestor:
                ancestor = stage.GetPrimAtPath(rigid_body_ancestor)
                if ancestor:
                    ccd_attrs = _attrs_matching(ancestor, "ccd")
                    physx_body = PhysxSchema.PhysxRigidBodyAPI(ancestor)
                    if physx_body:
                        ccd_attr = physx_body.GetEnableCCDAttr()
                        if ccd_attr:
                            ccd_attrs.setdefault(str(ccd_attr.GetName()), ccd_attr.Get())
            row = {
                "path": path,
                "category": classify_path(path),
                "type_name": prim.GetTypeName(),
                "applied_schemas": _applied(prim),
                "has_mesh_collision_api": bool(mesh_collision),
                "approximation": str(approximation) if approximation is not None else None,
                "has_rigid_body_api": _has_schema(prim, "PhysicsRigidBodyAPI"),
                "rigid_body_ancestor": rigid_body_ancestor,
                "ccd_attrs": ccd_attrs,
                "collision_enabled": UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
                else None,
                "physx_collision_attrs": {
                    "contact_offset": _attr_value(prim, "physxCollision:contactOffset"),
                    "rest_offset": _attr_value(prim, "physxCollision:restOffset"),
                },
            }
            row.update(mesh_stats)
            row["shape_family"] = collider_shape_family(
                row["type_name"], row["approximation"], row["mesh_points"], row["mesh_faces"]
            )
            row.update(_bbox_row(bbox_cache, prim))
            row["findings"] = baseline_findings(row)
            rows.append(row)

        active_rows = [row for row in rows if row["collision_enabled"] is not False]
        command_spike_report = None
        command_spike_report_path = Path(args.command_spike_report)
        if command_spike_report_path.exists():
            command_spike_report = json.loads(command_spike_report_path.read_text())
        baseline_status_value = baseline_status(rows)

        summary = {
            "collision_prim_count": len(rows),
            "active_collision_prim_count": len(active_rows),
            "baseline_status": baseline_status_value,
            "category_summary": _category_summary(rows),
            "active_category_summary": _category_summary(active_rows),
            "complex_mesh_candidate_count": sum(row["shape_family"] == "complex_mesh_unspecified" for row in rows),
            "dynamic_mesh_review_count": sum(
                "dynamic_mesh_collision_requires_explicit_supported_approximation_review" in row["findings"]
                for row in rows
            ),
            "ccd_advisory_count": sum(
                "small_fast_contact_part_ccd_candidate_after_geometry_review" in row["findings"] for row in rows
            ),
            "flagged_count": sum(bool(row["findings"]) for row in rows),
        }
        summary["ccd_policy"] = ccd_policy_from_evidence(
            baseline_status_value=baseline_status_value,
            active_rows=active_rows,
            command_spike_report=command_spike_report,
        )

        payload.update(
            {
                "status": "PASS",
                "stage": {
                    "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
                    "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
                },
                "summary": summary,
                "collision_rows": rows,
                "outputs": {"json": _rel(json_path), "markdown": _rel(md_path)},
            }
        )
        _write_json(json_path, payload)
        md_path.write_text(_render_markdown(_json_safe(payload)))
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "json": _rel(json_path),
                    "markdown": _rel(md_path),
                    "summary": summary,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except BaseException as exc:
        payload.update(
            {
                "status": "EXCEPTION",
                "exception": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc().splitlines()[-25:],
            }
        )
        _write_json(json_path, payload)
        print(json.dumps({"status": "EXCEPTION", "json": _rel(json_path), "exception": payload["exception"]}, ensure_ascii=False), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    raise SystemExit(main())
