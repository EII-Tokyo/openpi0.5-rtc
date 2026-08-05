#!/usr/bin/env python3
"""Export user-approved, zero-thickness ALOHA sandpaper print templates.

The supplier CAD redistribution license is unresolved, so all generated PDF,
DXF, and manifest files remain local-only under ``.codex/artifacts``.
"""

from __future__ import annotations

import argparse
from datetime import UTC
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from tools.aloha1_mapping.sandpaper_template import FINGER_CONTRACTS
from tools.aloha1_mapping.sandpaper_template import validate_review_report

A4_SIZE_MM = (210.0, 297.0)
CALIBRATION_SQUARE_MM = 50.0
MM_TO_POINTS = 72.0 / 25.4
FIXED_PDF_DATE = datetime(2000, 1, 1, tzinfo=UTC)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cut_wires(flat: dict[str, Any]) -> list[list[list[float]]]:
    wires = flat.get("cut_wires_2d_mm")
    if wires:
        return wires
    return [wire for panel in flat["panels"] for wire in panel.get("wires_2d_mm", [])]


def _pattern_dimensions(flat: dict[str, Any]) -> tuple[float, float]:
    min_x, min_y, max_x, max_y = [float(value) for value in flat["bounds_mm"]]
    return max_x - min_x, max_y - min_y


def _print_layout(flat: dict[str, Any]) -> tuple[float, float]:
    min_x, min_y, max_x, max_y = [float(value) for value in flat["bounds_mm"]]
    width, height = max_x - min_x, max_y - min_y
    if width > 180.0 or height > 185.0:
        raise ValueError("approved pattern does not fit the reserved A4 print area")
    return 105.0 - 0.5 * (min_x + max_x), 160.0 + 0.5 * (min_y + max_y)


def _render_pdf(report: dict[str, Any], *, side: str, output_path: Path) -> dict[str, Any]:
    record = report["sides"][side]
    flat = record["flat_pattern"]
    translate_x, translate_y = _print_layout(flat)
    width, height = _pattern_dimensions(flat)

    figure = Figure(figsize=(A4_SIZE_MM[0] / 25.4, A4_SIZE_MM[1] / 25.4))
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.set_xlim(0.0, A4_SIZE_MM[0])
    axis.set_ylim(A4_SIZE_MM[1], 0.0)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")

    for wire in _cut_wires(flat):
        closed = [*wire, wire[0]]
        axis.plot(
            [translate_x + float(point[0]) for point in closed],
            [translate_y - float(point[1]) for point in closed],
            color="#111111",
            linewidth=0.35 * MM_TO_POINTS,
            solid_joinstyle="miter",
        )
    for fold in record["folds"]:
        start, end = fold["line_2d_mm"]
        line = axis.plot(
            [translate_x + float(start[0]), translate_x + float(end[0])],
            [translate_y - float(start[1]), translate_y - float(end[1])],
            color="#1565c0",
            linewidth=0.30 * MM_TO_POINTS,
        )[0]
        line.set_dashes((3.0 * MM_TO_POINTS, 2.0 * MM_TO_POINTS))

    axis.text(15.0, 16.0, "ALOHA SANDPAPER 1:1 PRINT TEMPLATE", fontsize=14, weight="bold", color="#111111")
    axis.text(
        15.0,
        25.0,
        f"{side.upper()} FINGER | ONE PIECE | TWO OUTER FOLDS | ZERO-THICKNESS APPROXIMATION",
        fontsize=8.5,
        color="#111111",
    )
    axis.text(15.0, 31.0, "Black: cut. Blue dashed: fold. Print at Actual Size / 100%.", fontsize=8, color="#333333")
    axis.text(15.0, 37.0, f"Pattern bounds: {width:.3f} x {height:.3f} mm", fontsize=8, color="#333333")
    axis.add_patch(
        Rectangle(
            (15.0, 232.0),
            CALIBRATION_SQUARE_MM,
            CALIBRATION_SQUARE_MM,
            fill=False,
            edgecolor="#444444",
            linewidth=0.25 * MM_TO_POINTS,
        )
    )
    axis.text(69.0, 247.0, "50 x 50 mm calibration square", fontsize=8, color="#333333")
    axis.text(69.0, 253.0, "Measure after printing; reject if not 50.0 mm.", fontsize=8, color="#333333")
    axis.text(69.0, 259.0, "Disable Fit, Shrink, and Scale-to-page options.", fontsize=8, color="#333333")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        format="pdf",
        dpi=72,
        metadata={
            "Title": f"ALOHA {side} finger sandpaper 1:1 print template",
            "Author": "openpi0.5-rtc-reward-learning",
            "Creator": "deterministic ALOHA sandpaper exporter",
            "Producer": f"Matplotlib {mpl.__version__}",
            "CreationDate": FIXED_PDF_DATE,
            "ModDate": FIXED_PDF_DATE,
        },
    )
    return {
        "absolute_path": str(output_path.resolve()),
        "sha256": _sha256(output_path),
        "page_size_mm": list(A4_SIZE_MM),
        "print_scale": 1.0,
        "calibration_square_mm": [CALIBRATION_SQUARE_MM, CALIBRATION_SQUARE_MM],
        "local_only": True,
    }


def _dxf_pair(code: int, value: str | int | float) -> list[str]:
    return [str(code), str(value)]


def _dxf_polyline(points: list[list[float]], *, layer: str, closed: bool) -> list[str]:
    values = [
        *_dxf_pair(0, "LWPOLYLINE"),
        *_dxf_pair(100, "AcDbEntity"),
        *_dxf_pair(8, layer),
        *_dxf_pair(100, "AcDbPolyline"),
        *_dxf_pair(90, len(points)),
        *_dxf_pair(70, 1 if closed else 0),
    ]
    for x_value, y_value in points:
        values.extend(_dxf_pair(10, f"{float(x_value):.9f}"))
        values.extend(_dxf_pair(20, f"{float(y_value):.9f}"))
    return values


def _dxf_line(start: list[float], end: list[float], *, layer: str) -> list[str]:
    return [
        *_dxf_pair(0, "LINE"),
        *_dxf_pair(100, "AcDbEntity"),
        *_dxf_pair(8, layer),
        *_dxf_pair(100, "AcDbLine"),
        *_dxf_pair(10, f"{float(start[0]):.9f}"),
        *_dxf_pair(20, f"{float(start[1]):.9f}"),
        *_dxf_pair(30, "0.0"),
        *_dxf_pair(11, f"{float(end[0]):.9f}"),
        *_dxf_pair(21, f"{float(end[1]):.9f}"),
        *_dxf_pair(31, "0.0"),
    ]


def _render_dxf(report: dict[str, Any], *, side: str, output_path: Path) -> dict[str, Any]:
    record = report["sides"][side]
    flat = record["flat_pattern"]
    layers = (("CUT", 7, "CONTINUOUS"), ("FOLD", 5, "DASHED"), ("REFERENCE", 8, "CONTINUOUS"))
    values = [
        *_dxf_pair(0, "SECTION"),
        *_dxf_pair(2, "HEADER"),
        *_dxf_pair(9, "$ACADVER"),
        *_dxf_pair(1, "AC1024"),
        *_dxf_pair(9, "$INSUNITS"),
        *_dxf_pair(70, 4),
        *_dxf_pair(0, "ENDSEC"),
        *_dxf_pair(0, "SECTION"),
        *_dxf_pair(2, "TABLES"),
        *_dxf_pair(0, "TABLE"),
        *_dxf_pair(2, "LTYPE"),
        *_dxf_pair(70, 1),
        *_dxf_pair(0, "LTYPE"),
        *_dxf_pair(2, "DASHED"),
        *_dxf_pair(70, 0),
        *_dxf_pair(3, "Dashed fold line"),
        *_dxf_pair(72, 65),
        *_dxf_pair(73, 2),
        *_dxf_pair(40, 5.0),
        *_dxf_pair(49, 3.0),
        *_dxf_pair(74, 0),
        *_dxf_pair(49, -2.0),
        *_dxf_pair(74, 0),
        *_dxf_pair(0, "ENDTAB"),
        *_dxf_pair(0, "TABLE"),
        *_dxf_pair(2, "LAYER"),
        *_dxf_pair(70, len(layers)),
    ]
    for name, color, line_type in layers:
        values.extend(
            [
                *_dxf_pair(0, "LAYER"),
                *_dxf_pair(2, name),
                *_dxf_pair(70, 0),
                *_dxf_pair(62, color),
                *_dxf_pair(6, line_type),
            ]
        )
    values.extend(
        [
            *_dxf_pair(0, "ENDTAB"),
            *_dxf_pair(0, "ENDSEC"),
            *_dxf_pair(0, "SECTION"),
            *_dxf_pair(2, "ENTITIES"),
        ]
    )
    for wire in _cut_wires(flat):
        values.extend(_dxf_polyline(wire, layer="CUT", closed=True))
    for fold in record["folds"]:
        values.extend(_dxf_line(fold["line_2d_mm"][0], fold["line_2d_mm"][1], layer="FOLD"))
    reference_square = [[120.0, 0.0], [170.0, 0.0], [170.0, 50.0], [120.0, 50.0]]
    values.extend(_dxf_polyline(reference_square, layer="REFERENCE", closed=True))
    values.extend([*_dxf_pair(0, "ENDSEC"), *_dxf_pair(0, "EOF")])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(values) + "\n", encoding="ascii")
    return {
        "absolute_path": str(output_path.resolve()),
        "sha256": _sha256(output_path),
        "units": "mm",
        "layers": [layer[0] for layer in layers],
        "cut_wire_count": len(_cut_wires(flat)),
        "fold_line_count": len(record["folds"]),
        "calibration_square_mm": [CALIBRATION_SQUARE_MM, CALIBRATION_SQUARE_MM],
        "local_only": True,
    }


def export_print_template_set(*, report_path: Path, output_dir: Path) -> dict[str, Any]:
    report_path = report_path.resolve(strict=True)
    output_dir = output_dir.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_review_report(report)
    output_dir.mkdir(parents=True, exist_ok=True)
    sides: dict[str, Any] = {}
    for side in FINGER_CONTRACTS:
        pdf_path = output_dir / f"aloha_sandpaper_{side}_print_1to1.pdf"
        dxf_path = output_dir / f"aloha_sandpaper_{side}_print_1to1.dxf"
        width, height = _pattern_dimensions(report["sides"][side]["flat_pattern"])
        sides[side] = {
            "pattern_bounds_mm": [width, height],
            "artifacts": {
                "pdf": _render_pdf(report, side=side, output_path=pdf_path),
                "dxf": _render_dxf(report, side=side, output_path=dxf_path),
            },
        }
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "classification": "LOCAL_ONLY_APPROVED_ZERO_THICKNESS_PRINT_TEMPLATE",
        "source_report": {"absolute_path": str(report_path), "sha256": _sha256(report_path)},
        "final_print_template": True,
        "print_scale": 1.0,
        "material_total_thickness_mm": 0.0,
        "bend_compensation_mm": 0.0,
        "material_assumption": "USER_APPROVED_VERY_THIN_ZERO_THICKNESS",
        "calibration_square_mm": [CALIBRATION_SQUARE_MM, CALIBRATION_SQUARE_MM],
        "print_instruction": "ACTUAL_SIZE_100_PERCENT_DISABLE_FIT_AND_SHRINK",
        "redistribution": "PROHIBITED_PENDING_SUPPLIER_CAD_LICENSE_EVIDENCE",
        "sides": sides,
    }
    manifest_path = output_dir / "aloha_sandpaper_print_1to1_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--approved-zero-thickness", action="store_true", required=True)
    args = parser.parse_args()
    manifest = export_print_template_set(report_path=args.report, output_dir=args.output_dir)
    print(json.dumps({"status": manifest["status"], "output_dir": str(args.output_dir.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
