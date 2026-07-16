"""Create Iteration 005 by rebuilding mesh payloads from facets.

Iteration 004 applies conservative mesh cleanup but still leaves several
FreeCAD load warnings. This script goes one step further: it creates fresh
`Mesh.Mesh()` objects from each object's facet coordinates, then saves a new
FCStd. That removes stale or malformed mesh-kernel serialization while keeping
the visible geometry in the same coordinate frame.

The original imported assets and earlier iterations are left untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import FreeCAD as App
import Mesh


ROOT = Path("/home/eii/project/openpi0.5-rtc-reward-learning")
WORKDIR = ROOT / "scene_reconstruction" / "cad" / "aloha_incremental"
ITER4_DIR = WORKDIR / "iterations" / "iter_004_mesh_repaired_lower_camera_top_position"
ITER5_DIR = WORKDIR / "iterations" / "iter_005_mesh_rebuilt_lower_camera_top_position"
INPUT_FCSTD = ITER4_DIR / "iter_004_mesh_repaired_lower_camera_top_position.FCStd"
OUTPUT_FCSTD = ITER5_DIR / "iter_005_mesh_rebuilt_lower_camera_top_position.FCStd"


def _metric(mesh, attr: str):
    value = getattr(mesh, attr, None)
    if value is None:
        return None
    if callable(value):
        try:
            return value()
        except Exception:
            return None
    return value


def _mesh_metrics(mesh) -> dict[str, object]:
    return {
        "count_points": _metric(mesh, "CountPoints"),
        "count_facets": _metric(mesh, "CountFacets"),
        "count_edges": _metric(mesh, "CountEdges"),
        "is_solid": _metric(mesh, "isSolid"),
        "has_non_manifolds": _metric(mesh, "hasNonManifolds"),
        "has_self_intersections": _metric(mesh, "hasSelfIntersections"),
        "has_non_uniform_oriented_facets": _metric(mesh, "hasNonUniformOrientedFacets"),
        "count_non_uniform_oriented_facets": _metric(mesh, "countNonUniformOrientedFacets"),
        "count_components": _metric(mesh, "countComponents"),
        "count_segments": _metric(mesh, "countSegments"),
    }


def _rebuild_mesh_from_facets(source_mesh) -> Mesh.Mesh:
    rebuilt = Mesh.Mesh()
    for facet in source_mesh.Facets:
        points = facet.Points
        if len(points) < 3:
            continue
        # Facets from the ALOHA STL assets are triangles. If FreeCAD ever gives
        # a polygon here, triangulate it as a fan so the geometry remains valid.
        first = points[0]
        for i in range(1, len(points) - 1):
            rebuilt.addFacet(first, points[i], points[i + 1])
    for method_name in (
        "removeDuplicatedPoints",
        "removeDuplicatedFacets",
        "fixDegenerations",
        "harmonizeNormals",
    ):
        method = getattr(rebuilt, method_name, None)
        if method is not None:
            method()
    return rebuilt


def main() -> None:
    ITER5_DIR.mkdir(parents=True, exist_ok=True)
    doc = App.openDocument(str(INPUT_FCSTD))

    report: dict[str, object] = {
        "source_fcstd": str(INPUT_FCSTD),
        "output_fcstd": str(OUTPUT_FCSTD),
        "strategy": "rebuild each Mesh::Feature payload from facet coordinates",
        "mesh_objects": [],
    }

    for obj in doc.Objects:
        if not hasattr(obj, "Mesh"):
            continue
        before = _mesh_metrics(obj.Mesh)
        rebuilt = _rebuild_mesh_from_facets(obj.Mesh)
        obj.Mesh = rebuilt
        after = _mesh_metrics(obj.Mesh)
        report["mesh_objects"].append(
            {
                "name": obj.Name,
                "label": obj.Label,
                "before": before,
                "after": after,
                "changed_counts": {
                    "points": before.get("count_points") != after.get("count_points"),
                    "facets": before.get("count_facets") != after.get("count_facets"),
                    "edges": before.get("count_edges") != after.get("count_edges"),
                },
            }
        )

    doc.recompute()
    doc.saveAs(str(OUTPUT_FCSTD))
    App.closeDocument(doc.Name)

    (ITER5_DIR / "mesh_rebuild_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (ITER5_DIR / "changes.md").write_text(
        """# Iteration 005: mesh-rebuilt lower camera top-position model

This iteration is a sanitized copy of iter_004. It rebuilds every mesh payload
from facet coordinates into new FreeCAD `Mesh.Mesh()` objects.

Purpose:

- remove FreeCAD load warnings from stale or malformed mesh-kernel data;
- keep the lower-camera placement and scene layout unchanged;
- keep original imported mesh assets and earlier iterations read-only.

The detailed before/after metrics are in `mesh_rebuild_report.json`.
""",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_FCSTD}")
    print(f"Wrote {ITER5_DIR / 'mesh_rebuild_report.json'}")


if __name__ == "__main__":
    main()
