#!/usr/bin/env python3
"""Render a static top-down SVG overview of A4 camera visual markers."""

from __future__ import annotations

import argparse
from pathlib import Path


SUPPORT_FRAME_LENGTH_M = 1.220
SUPPORT_FRAME_WIDTH_M = 0.625
TABLE_LENGTH_M = 1.2192
TABLE_WIDTH_M = 0.7490
SCALE = 620.0
MARGIN = 90.0

MARKERS = {
    "cam_high": {
        "position": (0.0, -0.360),
        "color": "#2659ff",
        "direction_end": (0.0, -0.300),
    },
    "cam_low": {
        "position": (0.0, 0.360),
        "color": "#00cc40",
        "direction_end": (0.0, 0.300),
    },
    "cam_left_wrist": {
        "position": (-0.360, 0.0),
        "color": "#ff8c0a",
        "direction_end": (-0.300, 0.0),
    },
    "cam_right_wrist": {
        "position": (0.360, 0.0),
        "color": "#cc26ff",
        "direction_end": (0.300, 0.0),
    },
}


def _map_point(x_m: float, y_m: float, width_px: float, height_px: float) -> tuple[float, float]:
    return (width_px / 2.0 + x_m * SCALE, height_px / 2.0 - y_m * SCALE)


def _rect(center_x: float, center_y: float, size_x: float, size_y: float, width_px: float, height_px: float) -> str:
    x, y = _map_point(center_x - size_x / 2.0, center_y + size_y / 2.0, width_px, height_px)
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{size_x * SCALE:.1f}" '
        f'height="{size_y * SCALE:.1f}" fill="none" />'
    )


def render(output_path: Path) -> None:
    width_px = TABLE_LENGTH_M * SCALE + MARGIN * 2.0
    height_px = TABLE_WIDTH_M * SCALE + MARGIN * 2.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_px:.0f}" height="{height_px:.0f}" viewBox="0 0 {width_px:.0f} {height_px:.0f}">',
        '<rect width="100%" height="100%" fill="#fbfaf5"/>',
        '<style>text{font-family:Arial, sans-serif; fill:#1f2933} .small{font-size:13px} .label{font-size:15px;font-weight:600}</style>',
        '<text x="28" y="34" class="label">A4 camera marker overview: schematic visual markers only</text>',
        '<text x="28" y="56" class="small">Marker coordinates are not calibrated camera extrinsics.</text>',
    ]

    table = _rect(0.0, 0.0, TABLE_LENGTH_M, TABLE_WIDTH_M, width_px, height_px)
    parts.append(table.replace('fill="none"', 'fill="#d8c9a8" fill-opacity="0.22" stroke="#a39271" stroke-width="2"'))
    frame = _rect(0.0, 0.0, SUPPORT_FRAME_LENGTH_M, SUPPORT_FRAME_WIDTH_M, width_px, height_px)
    parts.append(frame.replace('fill="none"', 'fill="none" stroke="#1a9b45" stroke-width="5"'))

    # Approximate base placeholders: intentionally not robot geometry.
    for name, x in [("left_base_link", -0.38), ("right_base_link", 0.38)]:
        cx, cy = _map_point(x, 0.0, width_px, height_px)
        parts.append(f'<rect x="{cx-40:.1f}" y="{cy-22:.1f}" width="80" height="44" rx="6" fill="#111" fill-opacity="0.18" stroke="#111" stroke-width="1"/>')
        parts.append(f'<text x="{cx-48:.1f}" y="{cy+42:.1f}" class="small">{name}</text>')

    for name, spec in MARKERS.items():
        x, y = spec["position"]
        dx, dy = spec["direction_end"]
        px, py = _map_point(x, y, width_px, height_px)
        qx, qy = _map_point(dx, dy, width_px, height_px)
        parts.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{qx:.1f}" y2="{qy:.1f}" stroke="{spec["color"]}" stroke-width="4" marker-end="url(#arrow)"/>')
        parts.append(f'<rect x="{px-13:.1f}" y="{py-13:.1f}" width="26" height="26" rx="4" fill="{spec["color"]}" fill-opacity="0.95"/>')
        parts.append(f'<text x="{px+18:.1f}" y="{py+5:.1f}" class="label">{name}</text>')

    parts.insert(
        3,
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="currentColor"/></marker></defs>',
    )
    parts.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("aloha_isaac_rebuild/artifacts/screenshots/a4_camera_marker_topdown.svg"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
