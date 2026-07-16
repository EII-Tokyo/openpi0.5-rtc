"""Create Iteration 004 by sanitizing mesh data from Iteration 003.

FreeCAD 1.1.1 reports "The mesh data structure has some defects" eight times
when loading iter_003. The usual mesh quality predicates do not flag visible
geometry problems after the document is loaded, so this script keeps the visual
scene unchanged and rewrites mesh payloads in a fresh FCStd copy.

The original imported assets and iter_003 are left untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import FreeCAD as App


ROOT = Path("/home/eii/project/openpi0.5-rtc-reward-learning")
WORKDIR = ROOT / "scene_reconstruction" / "cad" / "aloha_incremental"
ITER3_DIR = WORKDIR / "iterations" / "iter_003_lower_camera_top_position"
ITER4_DIR = WORKDIR / "iterations" / "iter_004_mesh_repaired_lower_camera_top_position"
INPUT_FCSTD = ITER3_DIR / "iter_003_lower_camera_top_position.FCStd"
OUTPUT_FCSTD = ITER4_DIR / "iter_004_mesh_repaired_lower_camera_top_position.FCStd"


def _count_or_none(mesh, attr: str):
    try:
        value = getattr(mesh, attr)
    except Exception:
        return None
    if callable(value):
        try:
            return value()
        except Exception:
            return None
    return value


def _mesh_metrics(obj) -> dict[str, object]:
    mesh = obj.Mesh
    return {
        "name": obj.Name,
        "label": obj.Label,
        "count_points": _count_or_none(mesh, "CountPoints"),
        "count_facets": _count_or_none(mesh, "CountFacets"),
        "count_edges": _count_or_none(mesh, "CountEdges"),
        "is_solid": _count_or_none(mesh, "isSolid"),
        "has_non_manifolds": _count_or_none(mesh, "hasNonManifolds"),
        "has_self_intersections": _count_or_none(mesh, "hasSelfIntersections"),
        "has_non_uniform_oriented_facets": _count_or_none(mesh, "hasNonUniformOrientedFacets"),
        "count_non_uniform_oriented_facets": _count_or_none(mesh, "countNonUniformOrientedFacets"),
        "count_components": _count_or_none(mesh, "countComponents"),
        "count_segments": _count_or_none(mesh, "countSegments"),
    }


def _sanitize_mesh(obj) -> tuple[dict[str, object], dict[str, object], list[str]]:
    before = _mesh_metrics(obj)
    repaired = obj.Mesh.copy()
    applied: list[str] = []

    # Keep this conservative: do not call destructive topology repair such as
    # removeNonManifolds() unless a later diagnostic proves it is necessary.
    for method_name in (
        "fixIndices",
        "removeDuplicatedPoints",
        "removeDuplicatedFacets",
        "fixDegenerations",
        "harmonizeNormals",
    ):
        method = getattr(repaired, method_name, None)
        if method is None:
            continue
        method()
        applied.append(method_name)

    # Reassigning the copied mesh forces FreeCAD to serialize a fresh mesh data
    # structure in the new FCStd file.
    obj.Mesh = repaired
    after = _mesh_metrics(obj)
    return before, after, applied


def main() -> None:
    ITER4_DIR.mkdir(parents=True, exist_ok=True)
    doc = App.openDocument(str(INPUT_FCSTD))

    report: dict[str, object] = {
        "source_fcstd": str(INPUT_FCSTD),
        "output_fcstd": str(OUTPUT_FCSTD),
        "strategy": "copy mesh, conservative mesh repair, reassign, save as new FCStd",
        "mesh_objects": [],
    }

    for obj in doc.Objects:
        if not hasattr(obj, "Mesh"):
            continue
        before, after, applied = _sanitize_mesh(obj)
        report["mesh_objects"].append(
            {
                "name": obj.Name,
                "label": obj.Label,
                "applied_methods": applied,
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

    (ITER4_DIR / "mesh_repair_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (ITER4_DIR / "changes.md").write_text(
        """# Iteration 004: mesh-repaired lower camera top-position model

This iteration is a sanitized copy of `iter_003_lower_camera_top_position`.

Purpose:

- remove the repeated FreeCAD startup warning `The mesh data structure has some defects`;
- keep the lower-camera placement and scene geometry from iter_003;
- avoid modifying original imported mesh assets or older iterations.

Repair policy:

- copy each `Mesh::Feature` mesh payload;
- apply conservative FreeCAD mesh cleanup methods;
- reassign the mesh so FreeCAD serializes a fresh mesh data structure;
- save as a new FCStd file.

The detailed before/after mesh metrics are in `mesh_repair_report.json`.
""",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_FCSTD}")
    print(f"Wrote {ITER4_DIR / 'mesh_repair_report.json'}")


if __name__ == "__main__":
    main()
