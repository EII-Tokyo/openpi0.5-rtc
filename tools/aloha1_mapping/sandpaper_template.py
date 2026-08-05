"""Contracts and local-only render helpers for the ALOHA sandpaper template.

The supplier STEP has no confirmed redistribution license.  This module may be
committed because it contains only selection contracts and generic rendering
logic; generated geometry must remain in the ignored ``.codex/artifacts``
tree until redistribution is explicitly cleared.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
from typing import Any

EXPECTED_SOURCE_SHA256 = "337862418769d4ea8b801d26e68930c4412f870050e60769bbf91765194dc571"
EXPECTED_MAIN_FACE_AREA_MM2 = 2020.89740871246
EXPECTED_FREECAD_VERSION = "1.1.1"
EXPECTED_OPENCASCADE_VERSION = "7.8.1"

# Face/edge indices refer to the two installed handed members in the frozen
# supplier assembly.  Neither member is created by mirroring the other.
FINGER_CONTRACTS: dict[str, dict[str, Any]] = {
    "left": {
        "object_name": "Part__Feature007",
        "expected_label": "Aloha VX Fingers 2024-4-21 v2",
        "main_face_index_1_based": 117,
        "mirror_applied": False,
        "folds": (
            {
                "name": "outer_z_min",
                "main_edge_index_1_based": 4,
                "adjacent_face_index_1_based": 132,
            },
            {
                "name": "outer_z_max",
                "main_edge_index_1_based": 6,
                "adjacent_face_index_1_based": 123,
            },
        ),
    },
    "right": {
        "object_name": "Part__Feature008",
        "expected_label": "Aloha VX Fingers 2024-4-21 v001",
        "main_face_index_1_based": 128,
        "mirror_applied": False,
        "folds": (
            {
                "name": "outer_z_max",
                "main_edge_index_1_based": 4,
                "adjacent_face_index_1_based": 143,
            },
            {
                "name": "outer_z_min",
                "main_edge_index_1_based": 6,
                "adjacent_face_index_1_based": 134,
            },
        ),
    },
}


def _require(condition: bool, message: str) -> None:  # noqa: FBT001
    if not condition:
        raise ValueError(message)


def _finite_points(points: Sequence[Sequence[float]]) -> bool:
    return bool(points) and all(
        len(point) == 2 and all(math.isfinite(float(value)) for value in point) for point in points
    )


def validate_review_report(report: Mapping[str, Any]) -> None:
    """Validate the immutable-source and zero-thickness review contract."""

    _require(report.get("status") == "PASS", "review report status is not PASS")
    _require(
        report.get("classification") == "LOCAL_ONLY_ZERO_THICKNESS_SANDPAPER_REVIEW",
        "unexpected review classification",
    )
    source = report.get("source", {})
    _require(source.get("sha256") == EXPECTED_SOURCE_SHA256, "unexpected source hash")
    _require(source.get("read_only") is True, "source is not read-only")
    _require(
        source.get("license_status") == "UNKNOWN_HARD_BLOCKER_LOCAL_ONLY",
        "generated geometry is not explicitly local-only",
    )
    toolchain = report.get("toolchain", {})
    _require(
        toolchain.get("freecad_version") == EXPECTED_FREECAD_VERSION,
        "unexpected FreeCAD version",
    )
    _require(
        toolchain.get("opencascade_version") == EXPECTED_OPENCASCADE_VERSION,
        "unexpected OpenCascade version",
    )
    design = report.get("design", {})
    _require(design.get("material_total_thickness_mm") == 0.0, "review must be zero-thickness")
    _require(design.get("one_piece_per_finger") is True, "expected one piece per finger")
    _require(design.get("overlap_tabs") is False, "review unexpectedly contains overlap tabs")
    _require(design.get("fold_count_per_finger") == 2, "expected two folds per finger")
    _require(
        design.get("coverage") == "FULL_INNER_PROFILE_PLUS_TWO_OUTER_LONGITUDINAL_PANELS",
        "unexpected coverage contract",
    )

    sides = report.get("sides", {})
    _require(set(sides) == set(FINGER_CONTRACTS), "left/right review sides are incomplete")
    for side, contract in FINGER_CONTRACTS.items():
        record = sides[side]
        _require(record.get("object_name") == contract["object_name"], f"{side}: wrong object")
        _require(
            record.get("main_face_index_1_based") == contract["main_face_index_1_based"],
            f"{side}: wrong main face",
        )
        _require(
            abs(float(record.get("main_face_area_mm2", 0.0)) - EXPECTED_MAIN_FACE_AREA_MM2) < 1e-6,
            f"{side}: unexpected main face area",
        )
        folds = record.get("folds", [])
        _require(len(folds) == 2, f"{side}: expected two fold records")
        by_name = {fold.get("name"): fold for fold in folds}
        for expected in contract["folds"]:
            fold = by_name.get(expected["name"], {})
            _require(
                fold.get("main_edge_index_1_based") == expected["main_edge_index_1_based"],
                f"{side}/{expected['name']}: wrong fold edge",
            )
            _require(
                fold.get("adjacent_face_index_1_based") == expected["adjacent_face_index_1_based"],
                f"{side}/{expected['name']}: wrong adjacent face",
            )
            _require(_finite_points(fold.get("line_2d_mm", [])), f"{side}: invalid fold line")
            _require(
                float(fold.get("normal_alignment_residual", math.inf)) <= 1e-9,
                f"{side}/{expected['name']}: unfolded normal is not aligned",
            )
            _require(
                float(fold.get("shared_edge_residual_mm", math.inf)) <= 1e-8,
                f"{side}/{expected['name']}: shared edge moved during unfold",
            )
        flat = record.get("flat_pattern", {})
        _require(len(flat.get("panels", [])) == 3, f"{side}: expected three flat panels")
        _require(
            {panel.get("name") for panel in flat.get("panels", [])}
            == {"main", "outer_z_min", "outer_z_max"},
            f"{side}: unexpected flat panel set",
        )
        _require(not flat.get("relief_cut_lines_2d_mm", []), f"{side}: unexpected relief cuts")
        _require(
            float(flat.get("maximum_panel_plane_residual_mm", math.inf)) <= 1e-8,
            f"{side}: unfolded panels are not coplanar",
        )


def _path_data(wire: Sequence[Sequence[float]]) -> str:
    if not _finite_points(wire):
        raise ValueError("wire contains invalid points")
    commands = [f"M {float(wire[0][0]):.6f} {float(wire[0][1]):.6f}"]
    commands.extend(f"L {float(x):.6f} {float(y):.6f}" for x, y in wire[1:])
    commands.append("Z")
    return " ".join(commands)


def render_flat_pattern_svg(
    report: Mapping[str, Any],
    *,
    side: str,
    output_path: Path,
) -> None:
    """Render one A4 review SVG in physical millimetre units.

    This is intentionally marked as a review drawing.  Material-thickness
    compensation is not applied until the user approves the wrapped geometry.
    """

    validate_review_report(report)
    if side not in FINGER_CONTRACTS:
        raise ValueError(f"unknown finger side: {side}")
    record = report["sides"][side]
    flat = record["flat_pattern"]
    min_x, min_y, max_x, max_y = [float(value) for value in flat["bounds_mm"]]
    pattern_width = max_x - min_x
    pattern_height = max_y - min_y
    if pattern_width > 180.0 or pattern_height > 220.0:
        raise ValueError("flat review pattern does not fit the reserved A4 area")
    translate_x = 105.0 - 0.5 * (min_x + max_x)
    translate_y = 160.0 + 0.5 * (min_y + max_y)

    cut_wires = flat.get("cut_wires_2d_mm")
    if not cut_wires:
        cut_wires = [wire for panel in flat["panels"] for wire in panel.get("wires_2d_mm", [])]
    cut_paths = "\n".join(f'      <path class="cut" d="{_path_data(wire)}"/>' for wire in cut_wires)
    fold_paths = "\n".join(
        (
            '      <path class="fold" d="M '
            f"{float(fold['line_2d_mm'][0][0]):.6f} "
            f"{float(fold['line_2d_mm'][0][1]):.6f} L "
            f"{float(fold['line_2d_mm'][1][0]):.6f} "
            f'{float(fold["line_2d_mm"][1][1]):.6f}"/>'
        )
        for fold in record["folds"]
    )
    side_label = side.upper()
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="297mm" viewBox="0 0 210 297">
  <style>
    .cut {{ fill: none; stroke: #111; stroke-width: 0.35; }}
    .fold {{ fill: none; stroke: #1565c0; stroke-width: 0.3; stroke-dasharray: 3 2; }}
    .warning {{ fill: #b71c1c; font: bold 5px sans-serif; }}
    .label {{ fill: #111; font: 4px sans-serif; }}
    .small {{ fill: #333; font: 3px sans-serif; }}
  </style>
  <text class="warning" x="15" y="16">ZERO-THICKNESS REVIEW — NOT FINAL PRINT TEMPLATE</text>
  <text class="label" x="15" y="25">ALOHA {side_label} FINGER — ONE PIECE / TWO FOLDS / NO TABS</text>
  <text class="small" x="15" y="31">Black: outer cut. Blue dashed: CAD-derived outer fold.</text>
  <g transform="translate({translate_x:.6f} {translate_y:.6f}) scale(1 -1)">
{cut_paths}
{fold_paths}
  </g>
  <rect x="15" y="232" width="50" height="50" fill="none" stroke="#666" stroke-width="0.25"/>
  <text class="small" x="69" y="247">50 x 50 mm geometry check</text>
  <text class="small" x="69" y="253">Review only; bend compensation = 0.00 mm</text>
  <text class="small" x="69" y="259">Abrasive side faces the bottle / inward surface</text>
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")
