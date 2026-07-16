#!/usr/bin/env python3
"""Render front, top, right, and isometric review images for Iteration 000.

This renderer avoids matplotlib because the local system matplotlib is linked
against an older NumPy ABI. It creates deterministic 2D projection PNGs from
the same mesh references used to build the FCStd file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import mjcf_reference


ROOT = mjcf_reference.REPO_ROOT
WORKDIR = ROOT / "scene_reconstruction" / "cad" / "aloha_incremental"
ITER_DIR = WORKDIR / "iterations" / "iter_000_reference"
CANVAS = (1600, 1100)
MARGIN = 95


COLORS = {
    "robot": (37, 47, 60),
    "frame": (15, 15, 17),
    "table": (125, 92, 62),
    "camera": (0, 0, 0),
    "scene": (65, 65, 70),
}


def _view_basis(name: str):
    if name == "front":
        return np.array([1, 0, 0.0]), np.array([0, 0, 1.0]), "+X right, +Z up; looking along -Y"
    if name == "top":
        return np.array([1, 0, 0.0]), np.array([0, 1, 0.0]), "+X right, +Y forward; looking down -Z"
    if name == "right":
        return np.array([0, 1, 0.0]), np.array([0, 0, 1.0]), "+Y forward, +Z up; looking along -X"
    if name == "isometric":
        right = np.array([0.82, 0.57, 0.0])
        up = np.array([-0.30, 0.43, 0.85])
        right = right / np.linalg.norm(right)
        up = up / np.linalg.norm(up)
        return right, up, "isometric; X/Y/Z axes shown"
    raise ValueError(name)


def _load_projected_segments():
    refs = mjcf_reference.load_current_isaac_reference_meshes()
    segments_by_category: list[tuple[str, np.ndarray]] = []
    all_points = []
    for ref in refs:
        tris = mjcf_reference.read_mesh_triangles(ref.mesh_path)
        max_faces = 1800 if ref.category == "robot" else 1200
        if len(tris) > max_faces:
            step = max(1, len(tris) // max_faces)
            tris = tris[::step][:max_faces]
        pts = mjcf_reference.transform_points(tris.reshape((-1, 3)), ref.matrix_m).reshape((-1, 3, 3)) * 1000.0
        all_points.append(pts.reshape((-1, 3)))
        edges = np.concatenate(
            [
                pts[:, [0, 1], :],
                pts[:, [1, 2], :],
                pts[:, [2, 0], :],
            ],
            axis=0,
        )
        segments_by_category.append((ref.category, edges))
    return segments_by_category, np.concatenate(all_points, axis=0)


def _project(points: np.ndarray, right: np.ndarray, up: np.ndarray):
    return np.column_stack((points @ right, points @ up))


def _world_to_canvas(projected, mins, maxs):
    w, h = CANVAS
    span = np.maximum(maxs - mins, 1e-6)
    scale = min((w - 2 * MARGIN) / span[0], (h - 2 * MARGIN) / span[1])
    x = MARGIN + (projected[..., 0] - mins[0]) * scale
    y = h - MARGIN - (projected[..., 1] - mins[1]) * scale
    return np.stack([x, y], axis=-1)


def _draw_axis(draw: ImageDraw.ImageDraw, origin, vector, label, color, mins, maxs, right, up):
    pts = np.asarray([origin, origin + vector], dtype=float)
    xy = _world_to_canvas(_project(pts, right, up), mins, maxs)
    draw.line([tuple(xy[0]), tuple(xy[1])], fill=color, width=5)
    end = xy[1]
    draw.ellipse([end[0] - 5, end[1] - 5, end[0] + 5, end[1] + 5], fill=color)
    draw.text((end[0] + 8, end[1] - 14), label, fill=color)


def _render_one(name: str, segments_by_category, all_points):
    right, up, subtitle = _view_basis(name)
    projected_all = _project(all_points, right, up)
    mins = projected_all.min(axis=0)
    maxs = projected_all.max(axis=0)
    pad = (maxs - mins) * 0.08 + 80
    mins -= pad
    maxs += pad

    img = Image.new("RGB", CANVAS, "white")
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle([0, 0, CANVAS[0] - 1, CANVAS[1] - 1], outline=(220, 225, 230), width=2)
    draw.text((36, 24), f"iter_000_reference - {name} view", fill=(25, 30, 36))
    draw.text((36, 50), subtitle, fill=(80, 88, 100))

    for category, edges in segments_by_category:
        projected = _project(edges.reshape((-1, 3)), right, up).reshape((-1, 2, 2))
        canvas = _world_to_canvas(projected, mins, maxs)
        color = COLORS.get(category, COLORS["scene"])
        rgba = (*color, 120 if category == "table" else 165)
        for seg in canvas:
            p0 = tuple(seg[0])
            p1 = tuple(seg[1])
            draw.line([p0, p1], fill=rgba, width=1)

    origin = np.array([0.0, 0.0, 0.0])
    _draw_axis(draw, origin, np.array([250.0, 0.0, 0.0]), "+X", (210, 45, 45), mins, maxs, right, up)
    _draw_axis(draw, origin, np.array([0.0, 250.0, 0.0]), "+Y", (40, 150, 70), mins, maxs, right, up)
    _draw_axis(draw, origin, np.array([0.0, 0.0, 250.0]), "+Z", (55, 90, 210), mins, maxs, right, up)

    legend_x = CANVAS[0] - 300
    legend_y = 34
    for idx, (label, color) in enumerate(
        [
            ("robot reference", COLORS["robot"]),
            ("table reference", COLORS["table"]),
            ("frame reference", COLORS["frame"]),
            ("scene camera/mount", COLORS["camera"]),
        ]
    ):
        y = legend_y + idx * 28
        draw.rectangle([legend_x, y + 3, legend_x + 18, y + 18], fill=(*color, 190))
        draw.text((legend_x + 28, y), label, fill=(45, 50, 58))

    out = ITER_DIR / f"{name}.png"
    img.save(out)
    return out


def main():
    ITER_DIR.mkdir(parents=True, exist_ok=True)
    segments_by_category, all_points = _load_projected_segments()
    for name in ["front", "top", "right", "isometric"]:
        print(_render_one(name, segments_by_category, all_points).relative_to(ROOT))


if __name__ == "__main__":
    main()
