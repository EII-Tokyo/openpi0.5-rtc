"""Build the local-only zero-thickness ALOHA sandpaper review geometry.

Required environment variables:
  ALOHA_SANDPAPER_STEP
  ALOHA_SANDPAPER_OUTPUT_DIR

Run only with the pinned project-local FreeCAD 1.1.1 wrapper.  The supplier
STEP remains immutable; generated geometry is local-only because the source
redistribution license has not been confirmed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any

import FreeCAD as App
import Import
import MeshPart
import Part

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.aloha1_mapping.sandpaper_template import EXPECTED_FREECAD_VERSION  # noqa: E402
from tools.aloha1_mapping.sandpaper_template import EXPECTED_MAIN_FACE_AREA_MM2  # noqa: E402
from tools.aloha1_mapping.sandpaper_template import EXPECTED_OPENCASCADE_VERSION  # noqa: E402
from tools.aloha1_mapping.sandpaper_template import EXPECTED_SOURCE_SHA256  # noqa: E402
from tools.aloha1_mapping.sandpaper_template import FINGER_CONTRACTS  # noqa: E402
from tools.aloha1_mapping.sandpaper_template import validate_review_report  # noqa: E402

POINT_TOLERANCE_MM = 1.0e-8
AREA_TOLERANCE_MM2 = 1.0e-7
DISPLAY_OFFSET_MM = 0.15
REVIEW_LINEAR_DEFLECTION_MM = 0.20
REVIEW_ANGULAR_DEFLECTION_DEG = 20.0
REVIEW_ANGULAR_DEFLECTION_RAD = math.radians(REVIEW_ANGULAR_DEFLECTION_DEG)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_import_document(path: Path) -> Any:
    imported = Import.open(str(path))
    document = getattr(imported[0], "Document", None) if isinstance(imported, list | tuple) and imported else imported
    if document is not None:
        return document
    documents = list(App.listDocuments().values())
    if not documents:
        raise RuntimeError(f"FreeCAD did not open {path}")
    return documents[-1]


def _face_normal(face: Any) -> Any:
    center = face.CenterOfMass
    try:
        u_value, v_value = face.Surface.parameter(center)
    except Exception:
        u_min, u_max, v_min, v_max = face.ParameterRange
        u_value = (float(u_min) + float(u_max)) / 2.0
        v_value = (float(v_min) + float(v_max)) / 2.0
    normal = face.normalAt(float(u_value), float(v_value))
    normal.normalize()
    return normal


def _geometric_plane_normal(face: Any) -> Any:
    points = [vertex.Point for vertex in face.Vertexes]
    if len(points) < 3:
        raise RuntimeError("planar face has fewer than three vertices")
    first = points[0]
    for middle in points[1:]:
        first_axis = middle.sub(first)
        if first_axis.Length <= POINT_TOLERANCE_MM:
            continue
        for last in points[2:]:
            second_axis = last.sub(first)
            normal = first_axis.cross(second_axis)
            if normal.Length > POINT_TOLERANCE_MM:
                normal.normalize()
                return normal
    raise RuntimeError("could not derive a geometric plane normal")


def _adjacent_faces(shape: Any, edge: Any, excluded_face_index: int) -> list[int]:
    return [
        face_index
        for face_index, face in enumerate(shape.Faces, start=1)
        if face_index != excluded_face_index and any(edge.isSame(candidate) for candidate in face.Edges)
    ]


def _project_basis(main_face: Any) -> tuple[Any, Any, Any, Any]:
    origin = main_face.CenterOfMass
    normal = _face_normal(main_face)
    nominal_length_axis = App.Vector(0.0, -1.0, 0.0)
    normal_component = App.Vector(normal)
    normal_component.multiply(nominal_length_axis.dot(normal))
    u_axis = nominal_length_axis.sub(normal_component)
    u_axis.normalize()
    v_axis = App.Vector(0.0, 0.0, 1.0)
    if abs(v_axis.dot(normal)) > 1.0e-10:
        raise RuntimeError("supplier main finger face is no longer parallel to global Z")
    return origin, u_axis, v_axis, normal


def _project_point(point: Any, origin: Any, u_axis: Any, v_axis: Any) -> list[float]:
    relative = point.sub(origin)
    return [float(relative.dot(u_axis)), float(relative.dot(v_axis))]


def _wire_points(wire: Any) -> list[Any]:
    edges = list(wire.Edges)
    if not edges:
        raise RuntimeError("face contains an empty wire")
    if any(type(edge.Curve).__name__ != "Line" for edge in edges):
        raise RuntimeError("sandpaper review currently requires exact straight CAD edges")
    remaining = [[edge.Vertexes[0].Point, edge.Vertexes[-1].Point] for edge in edges]
    ordered = remaining.pop(0)
    while remaining:
        tail = ordered[-1]
        match_index = None
        reverse = False
        for index, pair in enumerate(remaining):
            if tail.distanceToPoint(pair[0]) <= POINT_TOLERANCE_MM:
                match_index = index
                break
            if tail.distanceToPoint(pair[1]) <= POINT_TOLERANCE_MM:
                match_index = index
                reverse = True
                break
        if match_index is None:
            raise RuntimeError("could not order exact polygon wire")
        pair = remaining.pop(match_index)
        ordered.append(pair[0] if reverse else pair[1])
    if ordered[-1].distanceToPoint(ordered[0]) > POINT_TOLERANCE_MM:
        raise RuntimeError("polygon wire is not closed")
    return ordered[:-1]


def _face_wires_2d(face: Any, basis: tuple[Any, Any, Any, Any]) -> list[list[list[float]]]:
    origin, u_axis, v_axis, _ = basis
    return [[_project_point(point, origin, u_axis, v_axis) for point in _wire_points(wire)] for wire in face.Wires]


def _point_key(point: list[float]) -> tuple[float, float]:
    return (round(float(point[0]), 7), round(float(point[1]), 7))


def _boundary_wires(panel_wires: list[list[list[float]]]) -> list[list[list[float]]]:
    segments: dict[
        tuple[tuple[float, float], tuple[float, float]],
        list[tuple[tuple[float, float], tuple[float, float]]],
    ] = {}
    coordinates: dict[tuple[float, float], list[float]] = {}
    for wire in panel_wires:
        for start, end in zip(wire, [*wire[1:], wire[0]], strict=True):
            start_key = _point_key(start)
            end_key = _point_key(end)
            coordinates.setdefault(start_key, start)
            coordinates.setdefault(end_key, end)
            undirected = tuple(sorted((start_key, end_key)))
            segments.setdefault(undirected, []).append((start_key, end_key))
    boundary = [records[0] for records in segments.values() if len(records) == 1]
    unexpected = [key for key, records in segments.items() if len(records) > 2]
    if unexpected:
        raise RuntimeError(f"non-manifold flat pattern boundary: {unexpected[:4]}")
    adjacency: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for start, end in boundary:
        adjacency.setdefault(start, []).append(end)
        adjacency.setdefault(end, []).append(start)
    if any(len(neighbours) != 2 for neighbours in adjacency.values()):
        raise RuntimeError("flat cut boundary is not a collection of closed loops")

    unused = {tuple(sorted((start, end))) for start, end in boundary}
    loops: list[list[list[float]]] = []
    while unused:
        first_edge = next(iter(unused))
        start, current = first_edge
        loop_keys = [start, current]
        unused.remove(first_edge)
        previous = start
        while current != start:
            candidates = [candidate for candidate in adjacency[current] if candidate != previous]
            if not candidates:
                raise RuntimeError("flat cut boundary traversal stopped before closing")
            next_key = candidates[0]
            edge_key = tuple(sorted((current, next_key)))
            if edge_key not in unused:
                if next_key == start:
                    break
                raise RuntimeError("flat cut boundary reused an edge")
            unused.remove(edge_key)
            loop_keys.append(next_key)
            previous, current = current, next_key
        if loop_keys[-1] == loop_keys[0]:
            loop_keys.pop()
        loops.append([coordinates[key] for key in loop_keys])
    return loops


def _polygon_face_xy(wires: list[list[list[float]]], *, x_offset: float = 0.0) -> Any:
    wire_faces = []
    for points in wires:
        vectors = [App.Vector(x + x_offset, y, 0.0) for x, y in points]
        wire = Part.makePolygon([*vectors, vectors[0]])
        wire_faces.append(Part.Face(wire))
    outer_index = max(range(len(wire_faces)), key=lambda index: float(wire_faces[index].Area))
    shape = wire_faces[outer_index]
    for index, candidate in enumerate(wire_faces):
        if index != outer_index:
            shape = shape.cut(candidate)
    return shape


def _translated(shape: Any, vector: Any) -> Any:
    result = shape.copy()
    result.translate(vector)
    return result


def _add_part(document: Any, name: str, label: str, shape: Any, color: tuple[float, float, float]) -> Any:
    obj = document.addObject("Part::Feature", name)
    obj.Label = label
    obj.Shape = shape
    if obj.ViewObject is not None:
        obj.ViewObject.ShapeColor = color
        obj.ViewObject.LineColor = color
    obj.addProperty("App::PropertyColor", "ReviewColor", "Review")
    obj.ReviewColor = color
    return obj


def _build_side(
    source_document: Any,
    side: str,
    contract: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    source_object = source_document.getObject(contract["object_name"])
    if source_object is None:
        raise RuntimeError(f"missing supplier object {contract['object_name']}")
    if str(source_object.Label) != contract["expected_label"]:
        raise RuntimeError(f"unexpected supplier label: {source_object.Label}")
    if source_object.Shape.isNull() or not source_object.Shape.isValid():
        raise RuntimeError(f"invalid supplier B-Rep: {contract['object_name']}")

    main_index = int(contract["main_face_index_1_based"])
    main_face = source_object.Shape.Faces[main_index - 1]
    if type(main_face.Surface).__name__ != "Plane":
        raise RuntimeError(f"{side}: main coverage face is not planar")
    if abs(float(main_face.Area) - EXPECTED_MAIN_FACE_AREA_MM2) > 1.0e-6:
        raise RuntimeError(f"{side}: main coverage face area changed")
    basis = _project_basis(main_face)
    main_normal = basis[3]

    flat_faces: dict[str, Any] = {"main": main_face.copy()}
    folded_faces: dict[str, Any] = {"main": main_face.copy()}
    fold_records: list[dict[str, Any]] = []
    for fold in contract["folds"]:
        edge_index = int(fold["main_edge_index_1_based"])
        adjacent_index = int(fold["adjacent_face_index_1_based"])
        shared_edge = main_face.Edges[edge_index - 1]
        adjacent = _adjacent_faces(source_object.Shape, shared_edge, main_index)
        if adjacent != [adjacent_index]:
            raise RuntimeError(f"{side}/{fold['name']}: adjacency changed: {adjacent} != {[adjacent_index]}")
        adjacent_face = source_object.Shape.Faces[adjacent_index - 1]
        if type(adjacent_face.Surface).__name__ != "Plane":
            raise RuntimeError(f"{side}/{fold['name']}: adjacent panel is not planar")
        folded_faces[fold["name"]] = adjacent_face.copy()

        endpoints = [vertex.Point for vertex in shared_edge.Vertexes]
        if len(endpoints) != 2:
            raise RuntimeError(f"{side}/{fold['name']}: fold edge is not a straight segment")
        axis = endpoints[1].sub(endpoints[0])
        axis.normalize()
        adjacent_normal = _face_normal(adjacent_face)
        sine = float(axis.dot(adjacent_normal.cross(main_normal)))
        cosine = float(adjacent_normal.dot(main_normal))
        aligned_angle_deg = math.degrees(math.atan2(sine, cosine))
        opposite_angle_deg = aligned_angle_deg + (-180.0 if aligned_angle_deg > 0.0 else 180.0)
        candidates = []
        for candidate_angle_deg in (aligned_angle_deg, opposite_angle_deg):
            candidate = adjacent_face.copy()
            candidate.rotate(endpoints[0], axis, candidate_angle_deg)
            candidate_plane_residual = max(
                abs(float(vertex.Point.sub(basis[0]).dot(main_normal))) for vertex in candidate.Vertexes
            )
            candidate_overlap_area = float(abs(candidate.common(main_face).Area))
            candidates.append(
                (
                    candidate_overlap_area,
                    candidate_plane_residual,
                    abs(candidate_angle_deg),
                    candidate_angle_deg,
                    candidate,
                )
            )
        overlap_area, plane_residual, _, angle_deg, unfolded = min(candidates, key=lambda item: item[:3])
        unfolded_normal = _geometric_plane_normal(unfolded)
        normal_residual = abs(1.0 - abs(float(unfolded_normal.dot(main_normal))))
        rotated_edge = shared_edge.copy()
        rotated_edge.rotate(endpoints[0], axis, angle_deg)
        rotated_points = [vertex.Point for vertex in rotated_edge.Vertexes]
        shared_residual = max(
            min(point.distanceToPoint(candidate) for candidate in endpoints) for point in rotated_points
        )
        if normal_residual > 1.0e-9 or plane_residual > POINT_TOLERANCE_MM:
            raise RuntimeError(
                f"{side}/{fold['name']}: exact planar unfold failed "
                f"(angle_deg={angle_deg:.12f}, normal_residual={normal_residual:.12e}, "
                f"plane_residual_mm={plane_residual:.12e}, "
                f"main_normal=({main_normal.x:.6f},{main_normal.y:.6f},{main_normal.z:.6f}), "
                f"unfolded_normal=({unfolded_normal.x:.6f},{unfolded_normal.y:.6f},"
                f"{unfolded_normal.z:.6f}))"
            )
        if overlap_area > AREA_TOLERANCE_MM2:
            diagnostics = [
                {
                    "angle_deg": candidate[3],
                    "overlap_area_mm2": candidate[0],
                    "plane_residual_mm": candidate[1],
                }
                for candidate in candidates
            ]
            raise RuntimeError(f"{side}/{fold['name']}: unfolded panel overlaps the main face: {diagnostics}")
        flat_faces[fold["name"]] = unfolded
        fold_records.append(
            {
                "name": fold["name"],
                "main_edge_index_1_based": edge_index,
                "adjacent_face_index_1_based": adjacent_index,
                "fold_edge_length_mm": float(shared_edge.Length),
                "adjacent_panel_area_mm2": float(adjacent_face.Area),
                "unfold_rotation_deg": float(angle_deg),
                "normal_alignment_residual": normal_residual,
                "shared_edge_residual_mm": shared_residual,
                "panel_plane_residual_mm": plane_residual,
                "main_overlap_area_mm2": overlap_area,
                "line_2d_mm": [_project_point(point, basis[0], basis[1], basis[2]) for point in endpoints],
            }
        )

    panel_names = ["main", *[fold["name"] for fold in contract["folds"]]]
    raw_panels = [
        {
            "name": name,
            "area_mm2": float(flat_faces[name].Area),
            "wires_2d_mm": _face_wires_2d(flat_faces[name], basis),
        }
        for name in panel_names
    ]
    all_points = [point for panel in raw_panels for wire in panel["wires_2d_mm"] for point in wire]
    min_x = min(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)

    def normalize(point: list[float]) -> list[float]:
        return [float(point[0] - min_x), float(point[1] - min_y)]

    panels = [
        {
            **panel,
            "wires_2d_mm": [[normalize(point) for point in wire] for wire in panel["wires_2d_mm"]],
        }
        for panel in raw_panels
    ]
    for fold_record in fold_records:
        fold_record["line_2d_mm"] = [normalize(point) for point in fold_record["line_2d_mm"]]
    panel_wires = [wire for panel in panels for wire in panel["wires_2d_mm"]]
    cut_wires = _boundary_wires(panel_wires)
    pairwise_overlaps = []
    for left_position, left_panel in enumerate(panels):
        left_shape = _polygon_face_xy(left_panel["wires_2d_mm"])
        for right_panel in panels[left_position + 1 :]:
            right_shape = _polygon_face_xy(right_panel["wires_2d_mm"])
            area = float(abs(left_shape.common(right_shape).Area))
            pairwise_overlaps.append({"left": left_panel["name"], "right": right_panel["name"], "area_mm2": area})
            if area > AREA_TOLERANCE_MM2:
                raise RuntimeError(
                    f"{side}: flat panels overlap: {left_panel['name']}/{right_panel['name']}={area}"
                )
    normalized_points = [point for wire in panel_wires for point in wire]
    width = max(point[0] for point in normalized_points)
    height = max(point[1] for point in normalized_points)
    maximum_plane_residual = max(
        abs(float(vertex.Point.sub(basis[0]).dot(main_normal)))
        for face in flat_faces.values()
        for vertex in face.Vertexes
    )

    reference_origin = main_face.CenterOfMass

    def local_point(point: Any) -> list[float]:
        return [
            float(point.x - reference_origin.x),
            float(point.y - reference_origin.y),
            float(point.z - reference_origin.z),
        ]

    review_mesh = MeshPart.meshFromShape(
        Shape=source_object.Shape,
        LinearDeflection=REVIEW_LINEAR_DEFLECTION_MM,
        AngularDeflection=REVIEW_ANGULAR_DEFLECTION_RAD,
        Relative=False,
    )
    mesh_vertices, mesh_facets = review_mesh.Topology
    wrapped_review = {
        "reference_origin_cad_global_mm": [
            float(reference_origin.x),
            float(reference_origin.y),
            float(reference_origin.z),
        ],
        "projection_basis_cad_global": {
            "length_axis": [float(basis[1].x), float(basis[1].y), float(basis[1].z)],
            "vertical_axis": [float(basis[2].x), float(basis[2].y), float(basis[2].z)],
            "main_normal": [float(main_normal.x), float(main_normal.y), float(main_normal.z)],
        },
        "finger_mesh": {
            "vertices_local_mm": [local_point(point) for point in mesh_vertices],
            "triangles": [[int(index) for index in facet] for facet in mesh_facets],
            "linear_deflection_mm": REVIEW_LINEAR_DEFLECTION_MM,
            "angular_deflection_deg": REVIEW_ANGULAR_DEFLECTION_DEG,
            "relative": False,
        },
        "panels": [],
    }
    for name in panel_names:
        panel_mesh = MeshPart.meshFromShape(
            Shape=folded_faces[name],
            LinearDeflection=REVIEW_LINEAR_DEFLECTION_MM,
            AngularDeflection=REVIEW_ANGULAR_DEFLECTION_RAD,
            Relative=False,
        )
        panel_vertices, panel_facets = panel_mesh.Topology
        wrapped_review["panels"].append(
            {
                "name": name,
                "wires_local_3d_mm": [
                    [local_point(point) for point in _wire_points(wire)] for wire in folded_faces[name].Wires
                ],
                "mesh_vertices_local_mm": [local_point(point) for point in panel_vertices],
                "mesh_triangles": [[int(index) for index in facet] for facet in panel_facets],
            }
        )

    review_document = App.newDocument(f"AlohaSandpaper{side.title()}Review")
    parameters = review_document.addObject("App::FeaturePython", "ReviewParameters")
    parameters.Label = "REVIEW ONLY — zero-thickness parameters"
    parameters.addProperty("App::PropertyString", "SourceSha256", "Evidence")
    parameters.SourceSha256 = EXPECTED_SOURCE_SHA256
    parameters.addProperty("App::PropertyString", "LicenseStatus", "Evidence")
    parameters.LicenseStatus = "UNKNOWN_HARD_BLOCKER_LOCAL_ONLY"
    parameters.addProperty("App::PropertyLength", "MaterialTotalThickness", "Design")
    parameters.MaterialTotalThickness = 0.0
    parameters.addProperty("App::PropertyLength", "EdgeClearance", "Design")
    parameters.EdgeClearance = 0.0
    parameters.addProperty("App::PropertyBool", "OverlapTabs", "Design")
    parameters.OverlapTabs = False
    parameters.addProperty("App::PropertyInteger", "FoldCount", "Design")
    parameters.FoldCount = len(contract["folds"])
    parameters.addProperty("App::PropertyString", "FingerSide", "Design")
    parameters.FingerSide = side

    source_group = review_document.addObject("App::DocumentObjectGroup", "SourceFinger")
    coverage_group = review_document.addObject("App::DocumentObjectGroup", "WrappedCoverage")
    flat_group = review_document.addObject("App::DocumentObjectGroup", "FlatPattern")
    center_translation = App.Vector(
        -float(main_face.CenterOfMass.x),
        -float(main_face.CenterOfMass.y),
        -float(main_face.CenterOfMass.z),
    )
    finger_obj = _add_part(
        review_document,
        "InstalledFingerBRep",
        f"{side.title()} installed supplier finger — reference",
        _translated(source_object.Shape, center_translation),
        (0.72, 0.72, 0.72),
    )
    if finger_obj.ViewObject is not None:
        finger_obj.ViewObject.Transparency = 70
    source_group.addObject(finger_obj)

    panel_colors = {
        "main": (0.95, 0.65, 0.10),
        "outer_z_min": (0.20, 0.70, 0.35),
        "outer_z_max": (0.20, 0.70, 0.35),
    }
    for name in panel_names:
        normal = _face_normal(folded_faces[name])
        display_offset = App.Vector(normal)
        display_offset.multiply(DISPLAY_OFFSET_MM)
        display_translation = App.Vector(center_translation).add(display_offset)
        coverage = _add_part(
            review_document,
            f"Wrapped_{name}",
            f"Wrapped panel: {name}",
            _translated(folded_faces[name], display_translation),
            panel_colors[name],
        )
        if coverage.ViewObject is not None:
            coverage.ViewObject.Transparency = 5
        coverage_group.addObject(coverage)

    flat_x_offset = 65.0
    for panel in panels:
        flat_shape = _polygon_face_xy(panel["wires_2d_mm"], x_offset=flat_x_offset)
        flat_obj = _add_part(
            review_document,
            f"Flat_{panel['name']}",
            f"Flat panel: {panel['name']}",
            flat_shape,
            panel_colors[panel["name"]],
        )
        if flat_obj.ViewObject is not None:
            flat_obj.ViewObject.Transparency = 15
        flat_group.addObject(flat_obj)
    for fold in fold_records:
        start, end = fold["line_2d_mm"]
        edge = Part.makeLine(
            App.Vector(start[0] + flat_x_offset, start[1], 0.2),
            App.Vector(end[0] + flat_x_offset, end[1], 0.2),
        )
        fold_obj = _add_part(
            review_document,
            f"Fold_{fold['name']}",
            f"FOLD: {fold['name']}",
            edge,
            (0.05, 0.25, 0.95),
        )
        if fold_obj.ViewObject is not None:
            fold_obj.ViewObject.LineWidth = 4.0
        flat_group.addObject(fold_obj)

    review_document.recompute()
    fcstd_path = output_dir / f"aloha_sandpaper_{side}_zero_thickness_review.FCStd"
    review_document.saveAs(str(fcstd_path))
    App.closeDocument(review_document.Name)

    return {
        "object_name": contract["object_name"],
        "expected_label": contract["expected_label"],
        "mirror_applied": False,
        "main_face_index_1_based": main_index,
        "main_face_area_mm2": float(main_face.Area),
        "main_face_wire_count": len(main_face.Wires),
        "main_face_edge_count": len(main_face.Edges),
        "folds": fold_records,
        "flat_pattern": {
            "panels": panels,
            "cut_wires_2d_mm": cut_wires,
            "relief_cut_lines_2d_mm": [],
            "inner_overlap_resolution": "NOT_APPLICABLE_INNER_PANELS_EXCLUDED",
            "bounds_mm": [0.0, 0.0, float(width), float(height)],
            "width_mm": float(width),
            "height_mm": float(height),
            "panel_area_sum_mm2": float(sum(panel["area_mm2"] for panel in panels)),
            "maximum_panel_plane_residual_mm": float(maximum_plane_residual),
            "pairwise_panel_overlaps": pairwise_overlaps,
            "maximum_pairwise_overlap_area_mm2": max(record["area_mm2"] for record in pairwise_overlaps),
        },
        "wrapped_review": wrapped_review,
        "artifacts": {
            "fcstd": {
                "absolute_path": str(fcstd_path.resolve()),
                "sha256": _sha256(fcstd_path),
                "local_only": True,
            }
        },
    }


source_text = os.environ.get("ALOHA_SANDPAPER_STEP")
output_text = os.environ.get("ALOHA_SANDPAPER_OUTPUT_DIR")
if not source_text or not output_text:
    raise RuntimeError("ALOHA_SANDPAPER_STEP and ALOHA_SANDPAPER_OUTPUT_DIR are required")

source_path = Path(source_text).resolve(strict=True)
output_dir = Path(output_text).resolve()
output_dir.mkdir(parents=True, exist_ok=True)
source_sha256 = _sha256(source_path)
if source_sha256 != EXPECTED_SOURCE_SHA256:
    raise RuntimeError(f"unexpected supplier STEP hash: {source_sha256}")
source_mode = source_path.stat().st_mode
if source_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
    raise RuntimeError("supplier STEP is not filesystem read-only")

freecad_version = ".".join(str(value) for value in App.Version()[:3])
if freecad_version != EXPECTED_FREECAD_VERSION:
    raise RuntimeError(f"unexpected FreeCAD version: {freecad_version}")
if str(Part.OCC_VERSION) != EXPECTED_OPENCASCADE_VERSION:
    raise RuntimeError(f"unexpected OpenCascade version: {Part.OCC_VERSION}")

source_document = _open_import_document(source_path)
sides = {side: _build_side(source_document, side, contract, output_dir) for side, contract in FINGER_CONTRACTS.items()}
report = {
    "schema_version": 1,
    "status": "PASS",
    "classification": "LOCAL_ONLY_ZERO_THICKNESS_SANDPAPER_REVIEW",
    "source": {
        "absolute_path": str(source_path),
        "sha256": source_sha256,
        "size_bytes": source_path.stat().st_size,
        "read_only": True,
        "license_status": "UNKNOWN_HARD_BLOCKER_LOCAL_ONLY",
        "redistribution": "PROHIBITED_PENDING_LICENSE_EVIDENCE",
    },
    "toolchain": {
        "freecad_version": freecad_version,
        "freecad_commit": str(App.Version()[7]),
        "python_version": ".".join(str(value) for value in sys.version_info[:3]),
        "opencascade_version": str(Part.OCC_VERSION),
    },
    "design": {
        "phase": "FIRST_GEOMETRY_REVIEW",
        "material_total_thickness_mm": 0.0,
        "bend_compensation": "NOT_APPLIED_PENDING_USER_GEOMETRY_APPROVAL_AND_MEASUREMENT",
        "edge_clearance_mm": 0.0,
        "one_piece_per_finger": True,
        "overlap_tabs": False,
        "fold_count_per_finger": len(FINGER_CONTRACTS["left"]["folds"]),
        "coverage": "FULL_INNER_PROFILE_PLUS_TWO_OUTER_LONGITUDINAL_PANELS",
        "photo_used_for_dimensions": False,
        "abrasive_side": "TOWARD_BOTTLE_AND_INWARD_GRIPPING_SURFACE",
    },
    "sides": sides,
    "acceptance": {
        "geometry_review_required": True,
        "final_print_template": False,
        "physical_robot_control_performed": False,
        "maximum_allowed_plane_residual_mm": POINT_TOLERANCE_MM,
        "maximum_allowed_overlap_area_mm2": AREA_TOLERANCE_MM2,
    },
}
validate_review_report(report)
report_path = output_dir / "aloha_sandpaper_zero_thickness_review.json"
report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"status": "PASS", "report": str(report_path)}, sort_keys=True))
