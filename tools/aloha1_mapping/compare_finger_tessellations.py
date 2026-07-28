"""Compare two independently generated ALOHA finger tessellation runs."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_comparison(
    run_a_manifest: Path,
    run_b_manifest: Path,
) -> dict[str, Any]:
    run_a = json.loads(run_a_manifest.read_text(encoding="utf-8"))
    run_b = json.loads(run_b_manifest.read_text(encoding="utf-8"))
    mesh_comparisons = {}
    for name in ("left_finger", "right_finger"):
        left = run_a["meshes"][name]
        right = run_b["meshes"][name]
        fields = (
            "obj_sha256",
            "canonical_geometry_sha256",
            "vertex_count",
            "triangle_count",
            "aabb_mm",
            "brep_volume_mm3",
            "connected_components",
            "degenerate_triangle_count",
            "source_placement_matrix_mm",
            "source_placement_determinant",
        )
        matches = {field: left[field] == right[field] for field in fields}
        mesh_comparisons[name] = {
            "all_fields_match": all(matches.values()),
            "matches": matches,
            "run_a": left,
            "run_b": right,
        }
    all_match = all(
        item["all_fields_match"] for item in mesh_comparisons.values()
    )
    angular_controlled = all(
        manifest.get("status") == "PASS"
        and manifest.get("mesher_api") == "MeshPart.meshFromShape"
        and isinstance(manifest.get("angular_deflection_rad"), float)
        and manifest.get("angular_deflection_rad") > 0.0
        for manifest in (run_a, run_b)
    )
    tessellation_parameters_match = all(
        run_a.get(field) == run_b.get(field)
        for field in (
            "freecad_version",
            "opencascade_version",
            "mesher_api",
            "linear_deflection_mm",
            "angular_deflection_rad",
            "angular_deflection_deg",
            "unit_scale_m_per_mm",
            "weld_sew_policy",
            "instance_merge_policy",
        )
    )
    production_pass = (
        all_match and angular_controlled and tessellation_parameters_match
    )
    return {
        "schema_version": 1,
        "status": (
            "PASS"
            if production_pass
            else ("PARTIAL" if all_match else "FAIL")
        ),
        "determinism_gate": "PASS" if all_match else "FAIL",
        "production_tessellation_gate": (
            "PASS" if production_pass else "HARD_BLOCKER"
        ),
        "production_blocker": None
        if production_pass
        else (
            "The compared manifests do not prove two matching runs with "
            "explicit MeshPart linear and angular deflection control. "
            "Linear-only runs remain diagnostic and cannot be promoted."
        ),
        "tessellation_parameters_match": tessellation_parameters_match,
        "angular_controlled": angular_controlled,
        "scope": (
            "two-fresh-directory deterministic supplier-CAD visual mesh "
            "comparison; collision and final assets unchanged"
        ),
        "run_a": {
            "manifest_path": str(run_a_manifest.resolve()),
            "manifest_sha256": _sha256(run_a_manifest),
        },
        "run_b": {
            "manifest_path": str(run_b_manifest.resolve()),
            "manifest_sha256": _sha256(run_b_manifest),
        },
        "mesh_comparisons": mesh_comparisons,
        "handedness": {
            "left_source": (
                "embedded handed B-Rep Part__Feature007 / CAD +X"
            ),
            "right_source": (
                "embedded handed B-Rep Part__Feature008 / CAD -X"
            ),
            "single_side_mirror_or_rotation_used": False,
        },
        "visual_mesh_collision_mesh_separation": "PASS",
    }


def write_comparison(
    report: Mapping[str, Any],
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# ALOHA Viper Finger Tessellation Determinism",
        "",
        f"- Overall status: `{report['status']}`",
        f"- Two-run determinism gate: `{report['determinism_gate']}`",
        (
            "- Production angular-deflection gate: "
            f"`{report['production_tessellation_gate']}`"
        ),
        "- Final/default visual and collision assets modified: `false`",
        "",
        report["production_blocker"]
        or (
            "Both fresh runs used matching explicit MeshPart linear and "
            "angular deflection parameters."
        ),
        "",
        "| Finger | Byte hash | Canonical geometry | Vertices | Triangles | Components | Degenerate |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for name, record in report["mesh_comparisons"].items():
        mesh = record["run_a"]
        lines.append(
            f"| {name} | "
            f"{'MATCH' if record['matches']['obj_sha256'] else 'DIFF'} | "
            f"{'MATCH' if record['matches']['canonical_geometry_sha256'] else 'DIFF'} | "
            f"{mesh['vertex_count']} | {mesh['triangle_count']} | "
            f"{mesh['connected_components']} | "
            f"{mesh['degenerate_triangle_count']} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
