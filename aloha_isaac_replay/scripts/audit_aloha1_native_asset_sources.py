from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase15_aloha1_native_source_audit_20260718"
DEFAULT_ASSET_ROOTS = (
    REPO_ROOT / "assets/isaac/original_stationary_aloha",
    REPO_ROOT / "assets/isaac/original_stationary_aloha_dynamic",
    REPO_ROOT / "assets/isaac/original_stationary_aloha_arm_only",
)


def _rel(path: str | Path | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _size(path: Path) -> int | None:
    return path.stat().st_size if path.exists() else None


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(number):
        return number
    return None


def _resolve_mesh_path(filename: str, urdf_path: Path, package_dir: Path | None) -> Path | None:
    if filename.startswith("package://"):
        if package_dir is None:
            return None
        package_rel = filename.split("package://", 1)[1]
        parts = package_rel.split("/", 1)
        if len(parts) == 2 and parts[0] == package_dir.name:
            return package_dir / parts[1]
        if len(parts) == 2 and parts[0] == "interbotix_xsarm_descriptions":
            return package_dir / parts[1]
        return None
    expanded = Path(filename)
    if expanded.is_absolute():
        return expanded
    return (urdf_path.parent / expanded).resolve()


def _mesh_rows_from_urdf(urdf_path: Path, package_dir: Path | None) -> list[dict[str, Any]]:
    if not urdf_path.exists():
        return []
    root = ET.parse(urdf_path).getroot()
    rows = []
    for link in root.findall("link"):
        link_name = link.get("name")
        for section in ("visual", "collision"):
            for item in link.findall(section):
                mesh = item.find("./geometry/mesh")
                if mesh is None:
                    continue
                filename = mesh.get("filename")
                resolved = _resolve_mesh_path(filename or "", urdf_path, package_dir)
                rows.append(
                    {
                        "link": link_name,
                        "section": section,
                        "filename": filename,
                        "resolved_path": _rel(resolved) if resolved else None,
                        "exists": bool(resolved and resolved.exists()),
                        "extension": Path(filename or "").suffix.lower(),
                    }
                )
    return rows


def _joint_rows_from_urdf(urdf_path: Path) -> list[dict[str, Any]]:
    if not urdf_path.exists():
        return []
    root = ET.parse(urdf_path).getroot()
    rows = []
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        axis = joint.find("axis")
        parent = joint.find("parent")
        child = joint.find("child")
        rows.append(
            {
                "name": joint.get("name"),
                "type": joint.get("type"),
                "parent": parent.get("link") if parent is not None else None,
                "child": child.get("link") if child is not None else None,
                "axis": axis.get("xyz") if axis is not None else None,
                "lower": _float_or_none(limit.get("lower")) if limit is not None else None,
                "upper": _float_or_none(limit.get("upper")) if limit is not None else None,
            }
        )
    return rows


def _summarize_urdf(urdf_path: Path, package_dir: Path | None) -> dict[str, Any]:
    root = ET.parse(urdf_path).getroot() if urdf_path.exists() else None
    links = root.findall("link") if root is not None else []
    joints = _joint_rows_from_urdf(urdf_path)
    mesh_rows = _mesh_rows_from_urdf(urdf_path, package_dir)
    by_section = Counter(row["section"] for row in mesh_rows)
    missing = [row for row in mesh_rows if not row["exists"]]
    return {
        "path": _rel(urdf_path),
        "exists": urdf_path.exists(),
        "size_bytes": _size(urdf_path),
        "robot_name": root.get("name") if root is not None else None,
        "link_count": len(links),
        "joint_count": len(joints),
        "joint_type_counts": dict(Counter(row["type"] for row in joints)),
        "mesh_reference_count": len(mesh_rows),
        "mesh_reference_counts_by_section": dict(by_section),
        "mesh_extension_counts": dict(Counter(row["extension"] for row in mesh_rows)),
        "missing_mesh_reference_count": len(missing),
        "missing_mesh_references_sample": missing[:20],
        "joints_sample": joints[:20],
    }


def _report_metrics(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {"exists": False}
    side_reports = report.get("side_reports", {})
    combined = report.get("combined_report", {})
    side_summary = {}
    for side, side_report in side_reports.items():
        side_summary[side] = {
            "mesh_count": side_report.get("mesh_count"),
            "collision_count": side_report.get("collision_count"),
            "rigid_body_count": side_report.get("rigid_body_count"),
            "joint_count": side_report.get("joint_count"),
            "articulation_roots": side_report.get("articulation_roots", []),
            "default_prim": side_report.get("default_prim"),
        }
    return {
        "exists": True,
        "status": report.get("status"),
        "combined_usd": _rel(report.get("combined_usd")),
        "package_dir": _rel(report.get("package_dir")),
        "source_urdfs": {key: _rel(value) for key, value in report.get("source_urdfs", {}).items()},
        "resolved_urdfs": {key: _rel(value) for key, value in report.get("resolved_urdfs", {}).items()},
        "import_urdfs": {key: _rel(value) for key, value in report.get("import_urdfs", {}).items()},
        "arm_only": report.get("arm_only", False),
        "merge_fixed_joints": report.get("merge_fixed_joints"),
        "side_reports": side_summary,
        "combined_report": {
            "mesh_count": combined.get("mesh_count"),
            "collision_count": combined.get("collision_count"),
            "rigid_body_count": combined.get("rigid_body_count"),
            "joint_count": combined.get("joint_count"),
            "articulation_roots": combined.get("articulation_roots", []),
        },
    }


def _usd_file_rows(root: Path) -> list[dict[str, Any]]:
    generated = root / "generated"
    rows = []
    for path in sorted(generated.rglob("*.usd")) + sorted(generated.rglob("*.usda")):
        rows.append({"path": _rel(path), "exists": path.exists(), "size_bytes": _size(path)})
    return rows


def _gate_from_variant(metrics: dict[str, Any]) -> dict[str, Any]:
    combined = metrics.get("combined_report", {})
    side_reports = metrics.get("side_reports", {})
    mesh_ok = (combined.get("mesh_count") or 0) > 0
    collision_ok = (combined.get("collision_count") or 0) > 0
    rigid_ok = (combined.get("rigid_body_count") or 0) > 0
    joint_ok = (combined.get("joint_count") or 0) >= 12
    articulation_ok = bool(combined.get("articulation_roots"))
    side_defaults_ok = all(bool(row.get("default_prim")) for row in side_reports.values())
    controller_ready = all([mesh_ok, collision_ok, rigid_ok, joint_ok, articulation_ok, side_defaults_ok])
    blockers = []
    if not mesh_ok:
        blockers.append("generated USD has no Mesh prims")
    if not collision_ok:
        blockers.append("generated USD has no CollisionAPI prims")
    if not rigid_ok:
        blockers.append("generated USD has no RigidBodyAPI prims")
    if not joint_ok:
        blockers.append("generated USD joint count is too low")
    if not articulation_ok:
        blockers.append("generated USD has no articulation root")
    if not side_defaults_ok:
        blockers.append("side USD defaultPrim missing")
    return {
        "controller_ready": controller_ready,
        "mesh_ok": mesh_ok,
        "collision_ok": collision_ok,
        "rigid_body_ok": rigid_ok,
        "joint_ok": joint_ok,
        "articulation_ok": articulation_ok,
        "side_default_prims_ok": side_defaults_ok,
        "blockers": blockers,
    }


def _audit_variant(root: Path) -> dict[str, Any]:
    report_path = root / "reports/import_report.json"
    report = _read_json(report_path)
    metrics = _report_metrics(report)
    package_dir = Path(report.get("package_dir")).resolve() if report and report.get("package_dir") else None
    urdf_inputs = {}
    for category in ("source_urdfs", "resolved_urdfs", "import_urdfs", "arm_only_urdfs"):
        for key, value in (report or {}).get(category, {}).items():
            urdf_inputs[f"{category}.{key}"] = Path(value)
    urdf_summaries = {key: _summarize_urdf(path, package_dir) for key, path in sorted(urdf_inputs.items()) if path.exists()}
    return {
        "root": _rel(root),
        "report_path": _rel(report_path),
        "report_metrics": metrics,
        "usd_files": _usd_file_rows(root),
        "urdf_summaries": urdf_summaries,
        "gate": _gate_from_variant(metrics),
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    variants = payload["variants"]
    lines = [
        "# Phase 15 ALOHA1 Native Asset Source Audit",
        "",
        "This report audits the project-local ALOHA1 URDF/USD sources after Phase 14 showed that forcing ALOHA1 qpos into the Trossen joint chain is not a valid controller path.",
        "",
        "## Summary",
        "",
        f"- Variants audited: {len(variants)}",
        f"- Controller-ready variants: {sum(1 for item in variants if item['gate']['controller_ready'])}",
        "- Decision: do not use the current generated ALOHA1 USDs as controller targets until mesh and collision import are fixed.",
        "",
        "## Variant Gates",
        "",
        "| Variant | Mesh | Collision | Rigid bodies | Joints | Articulation | Controller ready | Blockers |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for item in variants:
        combined = item["report_metrics"].get("combined_report", {})
        gate = item["gate"]
        lines.append(
            "| {variant} | {mesh} | {collision} | {rigid} | {joints} | {articulation} | {ready} | {blockers} |".format(
                variant=item["root"],
                mesh=combined.get("mesh_count"),
                collision=combined.get("collision_count"),
                rigid=combined.get("rigid_body_count"),
                joints=combined.get("joint_count"),
                articulation=len(combined.get("articulation_roots", [])),
                ready="PASS" if gate["controller_ready"] else "BLOCKED",
                blockers="<br>".join(gate["blockers"]) if gate["blockers"] else "-",
            )
        )
    lines.extend(
        [
            "",
            "## URDF Source Findings",
            "",
            "| Variant | URDF key | Links | Joints | Mesh refs | Visual refs | Collision refs | Missing mesh refs |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in variants:
        for key, summary in item["urdf_summaries"].items():
            by_section = summary["mesh_reference_counts_by_section"]
            lines.append(
                "| {variant} | {key} | {links} | {joints} | {meshes} | {visuals} | {collisions} | {missing} |".format(
                    variant=item["root"],
                    key=key,
                    links=summary["link_count"],
                    joints=summary["joint_count"],
                    meshes=summary["mesh_reference_count"],
                    visuals=by_section.get("visual", 0),
                    collisions=by_section.get("collision", 0),
                    missing=summary["missing_mesh_reference_count"],
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The resolved URDFs contain link, joint, visual mesh, and collision mesh references, and the referenced mesh files resolve locally. However, all generated USD variants currently report zero Mesh prims and zero CollisionAPI prims. That means the import process preserved enough joint and rigid-body structure to create an articulation-like skeleton, but it did not produce a usable visual/collision robot asset.",
            "",
            "For ALOHA1-native rebuilding, the source of truth should therefore be the ALOHA1 URDF joint semantics and local mesh package, not the current generated USD output and not the Trossen joint chain. Trossen should remain a framework reference for USD organization, drive tuning, validation, and Isaac Lab task structure.",
            "",
            "## Next Implementation Gates",
            "",
            "1. Re-run or replace the URDF import so the output has nonzero Mesh prims and CollisionAPI prims.",
            "2. Validate the imported USD with Isaac asset validation rules for Robot and Physics categories.",
            "3. Verify DOF names/order/limits against real ALOHA1 qpos before any controller replay.",
            "4. Only after those gates pass, add controller and grasp/replay tests.",
            "",
            "## Artifacts",
            "",
            f"- JSON: `{_rel(payload['json_path'])}`",
            f"- Markdown: `{_rel(path)}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit project-local ALOHA1 native URDF/USD asset sources.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--asset-root", action="append", default=[])
    args = parser.parse_args()

    asset_roots = [Path(item).resolve() for item in args.asset_root] if args.asset_root else list(DEFAULT_ASSET_ROOTS)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "repo_root": str(REPO_ROOT),
        "variants": [_audit_variant(root) for root in asset_roots],
    }
    payload["overall"] = {
        "variant_count": len(payload["variants"]),
        "controller_ready_count": sum(1 for item in payload["variants"] if item["gate"]["controller_ready"]),
        "recommended_action": "rebuild ALOHA1-native USD import path before controller work",
    }
    json_path = output_dir / "aloha1_native_asset_source_audit.json"
    md_path = output_dir / "aloha1_native_asset_source_audit.md"
    payload["json_path"] = str(json_path)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    _write_markdown(payload, md_path)
    print(json.dumps({"json": _rel(json_path), "markdown": _rel(md_path), "overall": payload["overall"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
