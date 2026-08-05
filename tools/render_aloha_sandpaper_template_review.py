#!/usr/bin/env python3
"""Render bounded local review evidence for the ALOHA sandpaper template."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from tools.aloha1_mapping.sandpaper_template import render_flat_pattern_svg
from tools.aloha1_mapping.sandpaper_template import validate_review_report

PANEL_COLORS = {
    "main": "#f2a900",
    "outer_z_min": "#2eaf62",
    "outer_z_max": "#2eaf62",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_points(points: list[list[float]], basis: dict[str, Any]) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    length_axis = np.asarray(basis["length_axis"], dtype=np.float64)
    normal_axis = np.asarray(basis["main_normal"], dtype=np.float64)
    vertical_axis = np.asarray(basis["vertical_axis"], dtype=np.float64)
    return np.column_stack((values @ length_axis, values @ normal_axis, values @ vertical_axis))


def _equal_3d_axes(axis: Any, points: np.ndarray) -> None:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = 0.5 * (minimum + maximum)
    radius = max(float((maximum - minimum).max()) * 0.56, 1.0)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def _render_wrapped(axis: Any, side_record: dict[str, Any], side: str) -> None:
    wrapped = side_record["wrapped_review"]
    basis = wrapped["projection_basis_cad_global"]
    finger = wrapped["finger_mesh"]
    finger_points = _display_points(finger["vertices_local_mm"], basis)
    finger_triangles = finger_points[np.asarray(finger["triangles"], dtype=np.int64)]
    axis.add_collection3d(
        Poly3DCollection(
            finger_triangles,
            facecolor="#b7bec8",
            edgecolor="#7d8794",
            linewidth=0.08,
            alpha=0.20,
        )
    )
    all_points = [finger_points]
    for panel in wrapped["panels"]:
        panel_points = _display_points(panel["mesh_vertices_local_mm"], basis)
        triangles = panel_points[np.asarray(panel["mesh_triangles"], dtype=np.int64)]
        axis.add_collection3d(
            Poly3DCollection(
                triangles,
                facecolor=PANEL_COLORS[panel["name"]],
                edgecolor="#202020",
                linewidth=0.35,
                alpha=0.90,
            )
        )
        all_points.append(panel_points)
    combined = np.concatenate(all_points, axis=0)
    _equal_3d_axes(axis, combined)
    axis.view_init(elev=21.0, azim=-63.0 if side == "left" else -117.0)
    axis.set_xlabel("finger length (mm)", fontsize=8)
    axis.set_ylabel("surface depth (mm)", fontsize=8)
    axis.set_zlabel("finger Z (mm)", fontsize=8)
    axis.tick_params(labelsize=7)
    axis.set_title(f"{side.upper()} — wrapped CAD coverage", fontsize=11, weight="bold")


def _render_flat(axis: Any, side_record: dict[str, Any], side: str) -> None:
    flat = side_record["flat_pattern"]
    for panel in flat["panels"]:
        wires = panel["wires_2d_mm"]
        outer_index = max(
            range(len(wires)),
            key=lambda index: abs(
                sum(
                    wires[index][point][0] * wires[index][(point + 1) % len(wires[index])][1]
                    - wires[index][(point + 1) % len(wires[index])][0] * wires[index][point][1]
                    for point in range(len(wires[index]))
                )
            ),
        )
        axis.add_collection(
            PolyCollection(
                [wires[outer_index]],
                facecolor=PANEL_COLORS[panel["name"]],
                edgecolor="none",
                alpha=0.62,
            )
        )
        for wire_index, wire in enumerate(wires):
            if wire_index != outer_index:
                axis.add_collection(PolyCollection([wire], facecolor="white", edgecolor="none", alpha=1.0))
    for wire in flat["cut_wires_2d_mm"]:
        closed = np.asarray([*wire, wire[0]], dtype=np.float64)
        axis.plot(closed[:, 0], closed[:, 1], color="#151515", linewidth=1.5)
    for fold in side_record["folds"]:
        line = np.asarray(fold["line_2d_mm"], dtype=np.float64)
        axis.plot(line[:, 0], line[:, 1], color="#1565c0", linewidth=1.4, linestyle=(0, (5, 3)))
    width = float(flat["width_mm"])
    height = float(flat["height_mm"])
    margin = 7.0
    axis.set_xlim(-margin, width + margin)
    axis.set_ylim(-margin, height + margin)
    axis.set_aspect("equal")
    axis.grid(visible=True, color="#dddddd", linewidth=0.45)
    axis.set_xlabel("mm")
    axis.set_ylabel("mm")
    axis.set_title(
        f"{side.upper()} — flat pattern {width:.3f} x {height:.3f} mm",
        fontsize=11,
        weight="bold",
    )
    axis.text(
        0.0,
        -5.2,
        "black=cut   blue dashed=outer fold",
        fontsize=8,
        color="#333333",
    )


def render_review(report: dict[str, Any], output_path: Path) -> None:
    validate_review_report(report)
    figure = plt.figure(figsize=(15.5, 11.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    for row, side in enumerate(("left", "right")):
        wrapped_axis = figure.add_subplot(grid[row, 0], projection="3d")
        flat_axis = figure.add_subplot(grid[row, 1])
        _render_wrapped(wrapped_axis, report["sides"][side], side)
        _render_flat(flat_axis, report["sides"][side], side)
    figure.suptitle(
        "ALOHA single-finger sandpaper wrap — first geometry review\n"
        "ZERO THICKNESS · NO BEND COMPENSATION · NOT FINAL PRINT TEMPLATE",
        fontsize=15,
        color="#9d1414",
        weight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.report.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_review_report(report)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        svg_path = output_dir / f"aloha_sandpaper_{side}_zero_thickness_review.svg"
        render_flat_pattern_svg(report, side=side, output_path=svg_path)
        artifacts[f"{side}_svg"] = {
            "absolute_path": str(svg_path.resolve()),
            "sha256": _sha256(svg_path),
            "local_only": True,
        }
    png_path = output_dir / "aloha_sandpaper_first_geometry_review.png"
    render_review(report, png_path)
    artifacts["review_png"] = {
        "absolute_path": str(png_path.resolve()),
        "sha256": _sha256(png_path),
        "local_only": True,
    }
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "classification": "LOCAL_ONLY_SANDPAPER_FIRST_GEOMETRY_REVIEW_RENDER",
        "source_report": {
            "absolute_path": str(report_path),
            "sha256": _sha256(report_path),
        },
        "artifacts": artifacts,
        "final_print_template": False,
        "material_total_thickness_mm": 0.0,
    }
    manifest_path = output_dir / "aloha_sandpaper_review_render_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "manifest": str(manifest_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
