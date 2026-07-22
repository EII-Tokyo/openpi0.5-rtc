#!/usr/bin/env python3
"""Render A5 support-frame measurement guide SVGs.

These drawings are measurement worksheets, not evidence of calibrated CAD.
They intentionally use labeled points so the user can reply with values such
as "AB=122.0 cm" or "MN=61.0 cm".
"""

from __future__ import annotations

import argparse
from pathlib import Path


OUTER_LENGTH_M = 1.220
OUTER_WIDTH_M = 0.625
PROFILE_WIDTH_M = 0.020
PIPE_LENGTH_M = 0.260
CAM_LOW_VERTICAL_SUPPORT_HEIGHT_M = 0.100
RAIL_Y_M = 0.323856
RAIL_Z_M = PROFILE_WIDTH_M / 2
EXTENSION_HALF_WIDTH_M = OUTER_LENGTH_M / 2
EXTENSION_INNER_GAP_M = 0.260
EXTENSION_DEPTH_M = EXTENSION_INNER_GAP_M + PROFILE_WIDTH_M
CAM_LOW = (0.030, OUTER_WIDTH_M / 2 + EXTENSION_INNER_GAP_M + PROFILE_WIDTH_M / 2)
CAM_HIGH = (0.0, -0.360)
BASE_EDGE_NEAR_CAM_LOW_Y_M = OUTER_WIDTH_M / 2 - 0.180
BASE_EDGE_NEAR_CAM_HIGH_Y_M = -OUTER_WIDTH_M / 2 + 0.235

SCALE = 520.0
MARGIN = 170.0


def _svg_header(width: float, height: float, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        '<rect width="100%" height="100%" fill="#fbfaf5"/>',
        "<defs>",
        '<marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L0,9 L8,4.5 z" fill="#d7263d"/></marker>',
        '<marker id="arrow-start" markerWidth="9" markerHeight="9" refX="1" refY="4.5" orient="auto"><path d="M8,0 L8,9 L0,4.5 z" fill="#d7263d"/></marker>',
        "</defs>",
        "<style>",
        "text{font-family:'Noto Sans CJK SC','Microsoft YaHei',Arial,sans-serif;fill:#1f2933}",
        ".title{font-size:23px;font-weight:700}",
        ".note{font-size:13px;fill:#526173}",
        ".label{font-size:15px;font-weight:700}",
        ".dim{font-size:14px;font-weight:700;fill:#d7263d}",
        ".small{font-size:12px;fill:#526173}",
        ".pt{font-size:13px;font-weight:700;fill:#ffffff}",
        "</style>",
        f'<text x="26" y="34" class="title">{title}</text>',
        f'<text x="26" y="56" class="note">{subtitle}</text>',
    ]


def _finish(parts: list[str], path: Path) -> None:
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _point(parts: list[str], x: float, y: float, name: str, color: str = "#23395b") -> None:
    parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="{color}"/>')
    parts.append(f'<text x="{x-4:.1f}" y="{y+4:.1f}" class="pt">{name}</text>')


def _dimension(parts: list[str], x1: float, y1: float, x2: float, y2: float, label: str, tx: float, ty: float) -> None:
    parts.append(
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        'stroke="#d7263d" stroke-width="2.2" marker-start="url(#arrow-start)" marker-end="url(#arrow)"/>'
    )
    parts.append(f'<text x="{tx:.1f}" y="{ty:.1f}" class="dim">{label}</text>')


def _note(parts: list[str], x: float, y: float, text: str) -> None:
    parts.append(f'<text x="{x:.1f}" y="{y:.1f}" class="small">{text}</text>')


def _map_top(x_m: float, y_m: float, width: float, height: float) -> tuple[float, float]:
    return width / 2 + x_m * SCALE, height / 2 + 70.0 - y_m * SCALE


def render_top(output: Path) -> None:
    width = OUTER_LENGTH_M * SCALE + MARGIN * 2
    height = max((OUTER_WIDTH_M + EXTENSION_DEPTH_M) * SCALE + MARGIN * 2, 820)
    parts = _svg_header(
        width,
        height,
        "A5 俯视图：支撑架与扩展区",
        "从上往下看：+Y 朝低位相机扩展区。实线是原始框，虚线是扩展框。",
    )
    left, front = _map_top(-OUTER_LENGTH_M / 2, -OUTER_WIDTH_M / 2, width, height)
    right, back = _map_top(OUTER_LENGTH_M / 2, OUTER_WIDTH_M / 2, width, height)
    frame_x = left
    frame_y = back
    frame_w = right - left
    frame_h = front - back
    parts.append(f'<rect x="{frame_x:.1f}" y="{frame_y:.1f}" width="{frame_w:.1f}" height="{frame_h:.1f}" fill="#d8c9a8" fill-opacity="0.20" stroke="#1a9b45" stroke-width="4"/>')

    # Extra camera-low-side extension. This is intentionally dashed and marked
    # as unknown so it cannot be mistaken for a measured final frame.
    ext_y0 = OUTER_WIDTH_M / 2
    ext_y1 = OUTER_WIDTH_M / 2 + EXTENSION_DEPTH_M
    ext_x0 = -EXTENSION_HALF_WIDTH_M
    ext_x1 = EXTENSION_HALF_WIDTH_M
    ex0, ey0 = _map_top(ext_x0, ext_y1, width, height)
    ex1, ey1 = _map_top(ext_x1, ext_y0, width, height)
    parts.append(
        f'<rect x="{ex0:.1f}" y="{ey0:.1f}" width="{ex1-ex0:.1f}" height="{ey1-ey0:.1f}" '
        'fill="#00b83a" fill-opacity="0.08" stroke="#00a34a" stroke-width="3" stroke-dasharray="10 8"/>'
    )
    parts.append(f'<text x="{ex0+10:.1f}" y="{ey0+24:.1f}" class="label">低位相机扩展框</text>')

    rail_left, rail_y = _map_top(-OUTER_LENGTH_M / 2, RAIL_Y_M, width, height)
    rail_right, _ = _map_top(OUTER_LENGTH_M / 2, RAIL_Y_M, width, height)
    parts.append(f'<rect x="{rail_left:.1f}" y="{rail_y - PROFILE_WIDTH_M * SCALE / 2:.1f}" width="{rail_right - rail_left:.1f}" height="{PROFILE_WIDTH_M * SCALE:.1f}" fill="#3d444d" fill-opacity="0.88"/>')

    # Measured Y range of both ALOHA bases. This does not define the base X
    # width or mesh, only the front/back Y edges provided by the user.
    for label, y_m in [
        ("底座边：距低位相机侧 18cm", BASE_EDGE_NEAR_CAM_LOW_Y_M),
        ("底座边：距高位相机侧 23.5cm", BASE_EDGE_NEAR_CAM_HIGH_Y_M),
    ]:
        x1, yy = _map_top(-OUTER_LENGTH_M / 2, y_m, width, height)
        x2, _ = _map_top(OUTER_LENGTH_M / 2, y_m, width, height)
        parts.append(f'<line x1="{x1:.1f}" y1="{yy:.1f}" x2="{x2:.1f}" y2="{yy:.1f}" stroke="#ff9f1c" stroke-width="3" stroke-dasharray="8 7"/>')
        _note(parts, x1 + 14, yy - 8, label)

    for label, (x, y), color, dx, dy in [
        ("低位相机", CAM_LOW, "#00b83a", 18, -22),
        ("高位相机", CAM_HIGH, "#2857ff", 18, -30),
    ]:
        px, py = _map_top(x, y, width, height)
        parts.append(f'<rect x="{px-13:.1f}" y="{py-13:.1f}" width="26" height="26" rx="4" fill="{color}"/>')
        parts.append(f'<text x="{px+dx:.1f}" y="{py+dy:.1f}" class="label">{label}</text>')

    points = {
        "A": (-OUTER_LENGTH_M / 2, -OUTER_WIDTH_M / 2),
        "B": (OUTER_LENGTH_M / 2, -OUTER_WIDTH_M / 2),
        "C": (OUTER_LENGTH_M / 2, OUTER_WIDTH_M / 2),
        "D": (-OUTER_LENGTH_M / 2, OUTER_WIDTH_M / 2),
        "E": (ext_x0, ext_y0),
        "F": (ext_x1, ext_y0),
        "G": (ext_x1, ext_y1),
        "H": (ext_x0, ext_y1),
        "I": CAM_LOW,
        "J": CAM_HIGH,
        "K": (-OUTER_LENGTH_M / 2, CAM_LOW[1]),
        "L": CAM_LOW,
        "P": (-OUTER_LENGTH_M / 2, RAIL_Y_M),
        "Q": (-OUTER_LENGTH_M / 2, RAIL_Y_M - PROFILE_WIDTH_M),
        "R": (-OUTER_LENGTH_M / 2, BASE_EDGE_NEAR_CAM_LOW_Y_M),
        "S": (-OUTER_LENGTH_M / 2, BASE_EDGE_NEAR_CAM_HIGH_Y_M),
    }
    for name, xy in points.items():
        _point(parts, *_map_top(*xy, width, height), name)

    ax, ay = _map_top(*points["A"], width, height)
    bx, by = _map_top(*points["B"], width, height)
    cx, cy = _map_top(*points["C"], width, height)
    ex, ey = _map_top(*points["E"], width, height)
    fx, fy = _map_top(*points["F"], width, height)
    gx, gy = _map_top(*points["G"], width, height)
    hx, hy = _map_top(*points["H"], width, height)
    ix, iy = _map_top(*points["I"], width, height)
    jx, jy = _map_top(*points["J"], width, height)
    kx, ky = _map_top(*points["K"], width, height)
    px, py = _map_top(*points["P"], width, height)
    qx, qy = _map_top(*points["Q"], width, height)
    rx, ry = _map_top(*points["R"], width, height)
    sx, sy = _map_top(*points["S"], width, height)

    _dimension(parts, ax, ay + 32, bx, by + 32, "AB 原框长 122cm", (ax + bx) / 2 - 70, ay + 58)
    _dimension(parts, cx + 34, cy, bx + 34, by, "BC 原框宽 62.5cm", cx - 142, (cy + by) / 2)
    _dimension(parts, ex, ey - 28, fx, fy - 28, "EF 接口宽 122cm", (ex + fx) / 2 - 70, ey - 42)
    _dimension(parts, fx + 42, fy, gx + 42, gy, "FG 外深 28cm", fx - 116, (fy + gy) / 2 - 8)
    _dimension(parts, hx, hy - 74, gx, gy - 74, "HG 外边长 122cm", (hx + gx) / 2 - 78, hy - 88)
    _dimension(parts, ix + 72, iy, jx + 72, jy, "IJ 相机距待测", ix + 82, (iy + jy) / 2)
    _dimension(parts, kx, iy + 34, ix, iy + 34, "KL 左边到低位相机 64cm", (kx + ix) / 2 - 100, iy + 56)
    _dimension(parts, px - 42, py, qx - 42, qy, "PQ 管宽待测", px - 150, py - 24)
    _dimension(parts, rx - 54, ry, sx - 54, sy, "RS 底座深 21cm", rx + 16, (ry + sy) / 2)

    ox, oy = _map_top(0.0, 0.0, width, height)
    _point(parts, ox, oy, "O", "#7a4cc2")
    parts.append(f'<line x1="{ox:.1f}" y1="{oy:.1f}" x2="{ox+90:.1f}" y2="{oy:.1f}" stroke="#7a4cc2" stroke-width="2"/>')
    parts.append(f'<line x1="{ox:.1f}" y1="{oy:.1f}" x2="{ox:.1f}" y2="{oy-90:.1f}" stroke="#7a4cc2" stroke-width="2"/>')
    _note(parts, ox + 96, oy + 5, "+X")
    _note(parts, ox + 7, oy - 96, "+Y 到低位相机扩展区")

    _note(parts, 26, height - 30, "反馈例子：AB=122cm, BC=62.5cm, FG外深=28cm。注意：26cm 是两根方管之间的内空，PQ=2cm方管。")
    _finish(parts, output)


def render_front(output: Path) -> None:
    width = OUTER_LENGTH_M * SCALE + MARGIN * 2
    height = RAIL_Z_M * SCALE + MARGIN * 2
    parts = _svg_header(
        width,
        height,
        "A5 正视图：只看高度",
        "从高位相机侧看向低位相机。这个视图看不出前后深度，只用于确认横向长度和高度。",
    )

    def map_front(x_m: float, z_m: float) -> tuple[float, float]:
        return width / 2 + x_m * SCALE, height - MARGIN - z_m * SCALE

    ground_y = height - MARGIN
    rail_z = RAIL_Z_M
    rail_left, rail_y = map_front(-OUTER_LENGTH_M / 2, rail_z)
    rail_right, _ = map_front(OUTER_LENGTH_M / 2, rail_z)

    parts.append(f'<line x1="{MARGIN:.1f}" y1="{ground_y:.1f}" x2="{width-MARGIN:.1f}" y2="{ground_y:.1f}" stroke="#a39271" stroke-width="3"/>')
    _note(parts, MARGIN, ground_y + 24, "桌面/高度参考")
    parts.append(f'<rect x="{rail_left:.1f}" y="{rail_y - PROFILE_WIDTH_M * SCALE / 2:.1f}" width="{rail_right - rail_left:.1f}" height="{PROFILE_WIDTH_M * SCALE:.1f}" fill="#3d444d" fill-opacity="0.88"/>')
    parts.append(f'<text x="{rail_left + 10:.1f}" y="{rail_y - 16:.1f}" class="label">顶部横梁投影，长 1220mm</text>')

    for idx, x in enumerate([-0.600, -0.433554, 0.433554, 0.600], start=1):
        px, py = map_front(x, rail_z)
        parts.append(f'<rect x="{px - PROFILE_WIDTH_M * SCALE / 2:.1f}" y="{py - 74:.1f}" width="{PROFILE_WIDTH_M * SCALE:.1f}" height="74" fill="#5c6670" fill-opacity="0.84"/>')
        parts.append(f'<text x="{px - 10:.1f}" y="{py - 84:.1f}" class="small">P{idx}</text>')

    points = {
        "M": (-OUTER_LENGTH_M / 2, 0.0),
        "N": (-OUTER_LENGTH_M / 2, rail_z),
        "O": (-OUTER_LENGTH_M / 2, rail_z),
        "P": (OUTER_LENGTH_M / 2, rail_z),
    }
    for name, xz in points.items():
        _point(parts, *map_front(*xz), name)
    mx, my = map_front(*points["M"])
    nx, ny = map_front(*points["N"])
    ox, oy = map_front(*points["O"])
    px, py = map_front(*points["P"])
    _dimension(parts, mx - 42, my, nx - 42, ny, "MN 横梁离桌面高度？", mx + 20, (my + ny) / 2)
    _dimension(parts, ox, oy - 40, px, py - 40, "OP 横梁长 122cm", (ox + px) / 2 - 70, oy - 58)
    _note(parts, 26, height - 60, "注意：这个视图看不出低位相机方向的内空和外边方管。")
    _note(parts, 26, height - 30, "反馈例子：MN=61cm, OP=122cm。若高度不同，反馈 MN=...cm。")
    _finish(parts, output)


def render_side(output: Path) -> None:
    local_scale = 1800.0
    local_margin = 130.0
    width = (PROFILE_WIDTH_M + EXTENSION_INNER_GAP_M + PROFILE_WIDTH_M) * local_scale + local_margin * 2 + 170.0
    height = (PROFILE_WIDTH_M + CAM_LOW_VERTICAL_SUPPORT_HEIGHT_M + 0.055) * local_scale + local_margin * 2 + 90.0
    parts = _svg_header(
        width,
        height,
        "A5 侧视图：低位相机支架 L 形",
        "从桌子侧面平看：水平外伸 26cm 内空，外边方管上立起 10cm 支撑。",
    )

    def map_side(y_m: float, z_m: float) -> tuple[float, float]:
        return local_margin + (y_m + PROFILE_WIDTH_M) * local_scale, height - local_margin - z_m * local_scale

    # Local Y origin is the outward face of the original cam_low-side tube.
    orig_tube_y0 = -PROFILE_WIDTH_M
    orig_tube_y1 = 0.0
    gap_y0 = 0.0
    gap_y1 = EXTENSION_INNER_GAP_M
    outer_tube_y0 = gap_y1
    outer_tube_y1 = gap_y1 + PROFILE_WIDTH_M
    support_center_y = outer_tube_y0 + PROFILE_WIDTH_M / 2
    tube_z0 = 0.0
    tube_z1 = PROFILE_WIDTH_M
    support_z0 = tube_z1
    support_z1 = tube_z1 + CAM_LOW_VERTICAL_SUPPORT_HEIGHT_M

    ox0, oz1 = map_side(orig_tube_y0, tube_z1)
    ox1, oz0 = map_side(orig_tube_y1, tube_z0)
    gx0, _ = map_side(gap_y0, tube_z0)
    gx1, _ = map_side(gap_y1, tube_z0)
    tx0, _ = map_side(outer_tube_y0, tube_z0)
    tx1, _ = map_side(outer_tube_y1, tube_z0)

    parts.append(f'<rect x="{ox0:.1f}" y="{oz1:.1f}" width="{ox1 - ox0:.1f}" height="{oz0 - oz1:.1f}" fill="#3d444d" fill-opacity="0.88"/>')
    parts.append(f'<rect x="{gx0:.1f}" y="{oz1:.1f}" width="{gx1 - gx0:.1f}" height="{oz0 - oz1:.1f}" fill="#5c6670" fill-opacity="0.70"/>')
    parts.append(f'<rect x="{tx0:.1f}" y="{oz1:.1f}" width="{tx1 - tx0:.1f}" height="{oz0 - oz1:.1f}" fill="#00a34a" fill-opacity="0.82"/>')
    _note(parts, ox0 - 44, oz0 + 24, "原始方管")
    _note(parts, tx1 + 8, oz0 + 24, "外边方管")
    _note(parts, (gx0 + gx1) / 2 - 42, oz1 - 12, "水平外伸管")

    support_x, support_top_y = map_side(support_center_y, support_z1)
    _, support_bottom_y = map_side(support_center_y, support_z0)
    support_w = PROFILE_WIDTH_M * local_scale
    parts.append(
        f'<rect x="{support_x - support_w / 2:.1f}" y="{support_top_y:.1f}" '
        f'width="{support_w:.1f}" height="{support_bottom_y - support_top_y:.1f}" '
        'fill="#00a34a" fill-opacity="0.82" stroke="#087f3f" stroke-width="2"/>'
    )
    parts.append(f'<rect x="{support_x-18:.1f}" y="{support_top_y-26:.1f}" width="36" height="20" rx="3" fill="#00b83a"/>')
    parts.append(f'<text x="{support_x+24:.1f}" y="{support_top_y-11:.1f}" class="label">低位相机底座</text>')

    points = {
        "A": (orig_tube_y0, tube_z0),
        "B": (orig_tube_y1, tube_z0),
        "U": (gap_y0, tube_z1),
        "V": (gap_y1, tube_z1),
        "W": (outer_tube_y1, support_z0),
        "CL": (outer_tube_y1, support_z1),
    }
    for name, yz in points.items():
        _point(parts, *map_side(*yz), name)
    ax, ay = map_side(*points["A"])
    bx, by = map_side(*points["B"])
    ux, uy = map_side(*points["U"])
    vx, vy = map_side(*points["V"])
    wx, wy = map_side(*points["W"])
    clx, cly = map_side(*points["CL"])
    _note(parts, ax - 8, ay + 34, "方管 2cm")
    _dimension(parts, ux, uy - 44, vx, vy - 44, "UV 内空 26cm", (ux + vx) / 2 - 62, uy - 60)
    _dimension(parts, wx + 62, wy, clx + 62, cly, "W-CL 支撑高 10cm", wx + 84, (wy + cly) / 2 + 4)
    _note(parts, 26, height - 60, "这是一张局部侧视图，只画低位相机扩展区，不画整个原始框宽。")
    _note(parts, 26, height - 30, "反馈例子：AB=2cm, UV内空=26cm, 外边方管=2cm, W-CL=10cm。若相机离支撑顶还有高度，请补充。")
    _finish(parts, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("aloha_isaac_rebuild/artifacts/screenshots"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_top(args.output_dir / "a5_support_frame_measurement_top.svg")
    render_front(args.output_dir / "a5_support_frame_measurement_front.svg")
    render_side(args.output_dir / "a5_support_frame_measurement_side.svg")
    print(args.output_dir)


if __name__ == "__main__":
    main()
