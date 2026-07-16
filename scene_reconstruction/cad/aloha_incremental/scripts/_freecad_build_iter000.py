from __future__ import annotations

import json
import sys
from pathlib import Path

import FreeCAD
import Mesh
import Part
from FreeCAD import Base

sys.path.insert(0, "/home/eii/project/openpi0.5-rtc-reward-learning/scene_reconstruction/cad/aloha_incremental/scripts")
import mjcf_reference


ROOT = mjcf_reference.REPO_ROOT
WORKDIR = ROOT / "scene_reconstruction" / "cad" / "aloha_incremental"
ITER_DIR = WORKDIR / "iterations" / "iter_000_reference"


def _base_matrix(matrix):
    m = Base.Matrix()
    m.A11, m.A12, m.A13, m.A14 = matrix[0]
    m.A21, m.A22, m.A23, m.A24 = matrix[1]
    m.A31, m.A32, m.A33, m.A34 = matrix[2]
    m.A41, m.A42, m.A43, m.A44 = matrix[3]
    return m


def _set_color(obj, color, transparency=0):
    try:
        obj.ViewObject.ShapeColor = color
        obj.ViewObject.Transparency = transparency
    except Exception:
        pass


def _document_bbox(doc):
    boxes = []
    for obj in doc.Objects:
        box = None
        if hasattr(obj, "Mesh"):
            box = obj.Mesh.BoundBox
        elif hasattr(obj, "Shape"):
            box = obj.Shape.BoundBox
        if box is not None and box.isValid():
            boxes.append(box)
    if not boxes:
        raise RuntimeError("No valid object bounding boxes in document")
    out = boxes[0]
    for box in boxes[1:]:
        out.add(box)
    return out


def main():
    ITER_DIR.mkdir(parents=True, exist_ok=True)
    doc = FreeCAD.newDocument("iter_000_reference")
    group = doc.addObject("App::DocumentObjectGroup", "READ_ONLY_ALOHA_REFERENCE")
    meshes = mjcf_reference.load_current_isaac_reference_meshes()
    object_count = 0
    for ref in meshes:
        mesh = Mesh.Mesh(str(ref.mesh_path))
        mesh.transform(_base_matrix(mjcf_reference.matrix_m_to_mm(ref.matrix_m)))
        obj = doc.addObject("Mesh::Feature", ref.name)
        obj.Mesh = mesh
        obj.addProperty("App::PropertyBool", "ReferenceLocked", "Reference")
        obj.ReferenceLocked = True
        obj.addProperty("App::PropertyString", "SourceAsset", "Reference")
        obj.SourceAsset = str(ref.mesh_path.relative_to(ROOT))
        if ref.category == "table":
            _set_color(obj, (0.55, 0.43, 0.32), 25)
        elif ref.category == "frame":
            _set_color(obj, (0.12, 0.12, 0.12), 8)
        elif ref.category == "camera":
            _set_color(obj, (0.05, 0.05, 0.05), 0)
        else:
            _set_color(obj, (0.13, 0.15, 0.17), 15)
        group.addObject(obj)
        object_count += 1

    # Transparent desktop reference plane for measurement orientation only. The
    # real table geometry from the current Isaac reference is imported above.
    table_min, table_max = mjcf_reference.table_reference_box_mm()
    table = doc.addObject("Part::Box", "REF_TABLE_DESKTOP_PLANE")
    table.Length = table_max[0] - table_min[0]
    table.Width = table_max[1] - table_min[1]
    table.Height = table_max[2] - table_min[2]
    table.Placement.Base = Base.Vector(table_min[0], table_min[1], table_min[2])
    table.addProperty("App::PropertyBool", "ReferenceLocked", "Reference")
    table.ReferenceLocked = True
    table.addProperty("App::PropertyString", "SourceAsset", "Reference")
    table.SourceAsset = "external/mujoco_menagerie/aloha/scene.xml"
    _set_color(table, (0.45, 0.62, 0.78), 72)

    # Simple world axes in the FCStd file; render labels are added by render_standard_views.py.
    axis_len = 250.0
    x_axis = doc.addObject("Part::Box", "REF_AXIS_X_RED")
    x_axis.Length, x_axis.Width, x_axis.Height = axis_len, 6.0, 6.0
    x_axis.Placement.Base = Base.Vector(0, -3, 4)
    _set_color(x_axis, (0.85, 0.12, 0.12), 0)
    y_axis = doc.addObject("Part::Box", "REF_AXIS_Y_GREEN")
    y_axis.Length, y_axis.Width, y_axis.Height = 6.0, axis_len, 6.0
    y_axis.Placement.Base = Base.Vector(-3, 0, 4)
    _set_color(y_axis, (0.12, 0.60, 0.20), 0)
    z_axis = doc.addObject("Part::Box", "REF_AXIS_Z_BLUE")
    z_axis.Length, z_axis.Width, z_axis.Height = 6.0, 6.0, axis_len
    z_axis.Placement.Base = Base.Vector(-3, -3, 0)
    _set_color(z_axis, (0.12, 0.25, 0.85), 0)

    doc.recompute()
    fcstd = ITER_DIR / "iter_000_reference.FCStd"
    doc.saveAs(str(fcstd))

    bbox = _document_bbox(doc)
    metadata = {
        "iteration": "iter_000_reference",
        "units": "mm",
        "source_assets_read_only": [
            str(mjcf_reference.ALOHA_XML.relative_to(ROOT)),
            str(mjcf_reference.SCENE_XML.relative_to(ROOT)),
            "local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose/aloha2_menagerie_scene_deep_black_real_start_pose.usd",
        ],
        "mesh_object_count": object_count,
        "known_dimensions": mjcf_reference.known_dimensions(),
        "freecad_file": str(fcstd.relative_to(ROOT)),
        "document_bbox_mm": {
            "xmin": bbox.XMin,
            "xmax": bbox.XMax,
            "ymin": bbox.YMin,
            "ymax": bbox.YMax,
            "zmin": bbox.ZMin,
            "zmax": bbox.ZMax,
            "x_length": bbox.XLength,
            "y_length": bbox.YLength,
            "z_length": bbox.ZLength,
        },
        "notes": [
            "Current Isaac ALOHA source mesh objects are imported as read-only reference meshes, not converted into guessed parametric CAD.",
            "No separate ALOHA2 workcell_v2 STL is used as the primary reference.",
            "The desktop plane is an orientation aid; imported scene mesh contains the table/frame visual reference.",
        ],
    }
    (ITER_DIR / "bbox_and_dimensions.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


main()
