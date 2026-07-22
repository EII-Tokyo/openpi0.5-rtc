#!/usr/bin/env python3
"""Render A11 base X-separation measurement worksheet SVGs."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


DEFAULT_CONFIG = Path("aloha_isaac_rebuild/configs/physical_reconstruction/base_anchor_measurement_worksheet.yaml")
DEFAULT_OUTPUT_DIR = Path("aloha_isaac_rebuild/artifacts/screenshots")

SCALE = 560.0
MARGIN_X = 160.0
MARGIN_Y = 155.0


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
        ".pt{font-size:12px;font-weight:700;fill:#ffffff}",
        "</style>",
        f'<text x="26" y="34" class="title">{title}</text>',
        f'<text x="26" y="58" class="note">{subtitle}</text>',
    ]


def _finish(parts: list[str], path: Path) -> None:
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _point(parts: list[str], x: float, y: float, name: str, color: str = "#23395b") -> None:
    parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{color}"/>')
    parts.append(f'<text x="{x-4:.1f}" y="{y+4:.1f}" class="pt">{name}</text>')


def _dimension(parts: list[str], x1: float, y1: float, x2: float, y2: float, label: str, tx: float, ty: float) -> None:
    parts.append(
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        'stroke="#d7263d" stroke-width="2.1" marker-start="url(#arrow-start)" marker-end="url(#arrow)"/>'
    )
    parts.append(f'<text x="{tx:.1f}" y="{ty:.1f}" class="dim">{label}</text>')


def _note(parts: list[str], x: float, y: float, text: str) -> None:
    parts.append(f'<text x="{x:.1f}" y="{y:.1f}" class="small">{text}</text>')


def _value(config: dict, name: str) -> float:
    return float(config["known_measured_values"][name]["value"])


def render_top(config: dict, output: Path) -> None:
    outer_x = _value(config, "support_frame_outer_length_m")
    outer_y = _value(config, "support_frame_outer_width_m")
    base_low_y = _value(config, "base_edge_near_cam_low_y_m")
    base_high_y = _value(config, "base_edge_near_cam_high_y_m")
    base_center_y = (base_low_y + base_high_y) / 2
    width = outer_x * SCALE + MARGIN_X * 2
    height = outer_y * SCALE + MARGIN_Y * 2
    parts = _svg_header(
        width,
        height,
        "A11 俯视测量图：只测双底座 X 距离",
        "底座高度、宽度和视觉形状复用原始 ALOHA1 USD；不要重新量这些几何。",
    )

    def m(x_m: float, y_m: float) -> tuple[float, float]:
        return width / 2 + x_m * SCALE, height / 2 - y_m * SCALE

    left_x = -outer_x / 2
    right_x = outer_x / 2
    low_y = outer_y / 2
    high_y = -outer_y / 2
    lx, ly = m(left_x, low_y)
    rx, ry = m(right_x, high_y)
    parts.append(f'<rect x="{lx:.1f}" y="{ly:.1f}" width="{rx-lx:.1f}" height="{ry-ly:.1f}" fill="#d8c9a8" fill-opacity="0.20" stroke="#1a9b45" stroke-width="4"/>')
    _note(parts, lx + 10, ly - 12, "支撑架外框：122cm x 62.5cm")

    bx0, by0 = m(left_x, base_low_y)
    bx1, by1 = m(right_x, base_high_y)
    parts.append(f'<rect x="{bx0:.1f}" y="{by0:.1f}" width="{bx1-bx0:.1f}" height="{by1-by0:.1f}" fill="#ff9f1c" fill-opacity="0.15" stroke="#ff9f1c" stroke-width="2" stroke-dasharray="9 7"/>')
    _note(parts, bx0 + 12, by0 + 22, "已测底座 Y 带：18cm / 23.5cm 推导")

    left_base_x = -0.315
    right_base_x = 0.315
    base_w = 0.185
    base_d = base_low_y - base_high_y
    for name, cx, color, label in [
        ("LC", left_base_x, "#2878d7", "左底座：用原始 USD 视觉几何"),
        ("RC", right_base_x, "#8e52c7", "右底座：用原始 USD 视觉几何"),
    ]:
        x, y = m(cx - base_w / 2, base_low_y)
        x2, y2 = m(cx + base_w / 2, base_high_y)
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{x2-x:.1f}" height="{y2-y:.1f}" fill="{color}" fill-opacity="0.18" stroke="{color}" stroke-width="2"/>')
        _point(parts, *m(cx, base_center_y), name, color=color)
        _note(parts, x - 12, y - 10, label)

    lcx, lcy = m(left_base_x, base_center_y)
    rcx, rcy = m(right_base_x, base_center_y)
    _dimension(parts, lcx, lcy - 72, rcx, rcy - 72, "需要实测：LC-RC 双底座 X 距离", (lcx + rcx) / 2 - 128, lcy - 92)

    left_inner_x = left_base_x + base_w / 2
    right_inner_x = right_base_x - base_w / 2
    lix, liy = m(left_inner_x, base_center_y)
    rix, riy = m(right_inner_x, base_center_y)
    _dimension(parts, lix, liy + 70, rix, riy + 70, "可选：两个内侧边之间的 X 间隙", (lix + rix) / 2 - 112, liy + 94)

    _dimension(parts, lx, ly + (ry - ly) + 46, rx, ry + 46, "支撑架总长 122cm", (lx + rx) / 2 - 58, ry + 70)
    _note(parts, 26, height - 78, "首选回复：inter_base_anchor_x_distance_cm=...")
    _note(parts, 26, height - 52, "如果中心不好量，可回复：inter_base_inner_edge_gap_x_cm=... 或 inter_base_outer_span_x_cm=...")
    _note(parts, 26, height - 26, "注意：底座高度、宽度、视觉形状、底座轮廓均复用原始 ALOHA1 USD，不再作为本步骤测量项。")
    _finish(parts, output)


def render_side(config: dict, output: Path) -> None:
    width = 860
    height = 520
    parts = _svg_header(
        width,
        height,
        "A11 侧视说明：底座视觉几何复用原始 USD",
        "本图只记录纠偏结论：侧向高度和底座形状不再重测。",
    )
    table_y = 360
    parts.append(f'<line x1="90" y1="{table_y}" x2="760" y2="{table_y}" stroke="#8b7e66" stroke-width="4"/>')
    _note(parts, 92, table_y + 25, "桌面/安装平面示意")
    parts.append('<rect x="230" y="250" width="145" height="78" fill="#2878d7" fill-opacity="0.18" stroke="#2878d7" stroke-width="2"/>')
    parts.append('<rect x="485" y="250" width="145" height="78" fill="#8e52c7" fill-opacity="0.18" stroke="#8e52c7" stroke-width="2"/>')
    _note(parts, 220, 226, "左底座视觉：复用原始 ALOHA1 USD")
    _note(parts, 474, 226, "右底座视觉：复用原始 ALOHA1 USD")
    parts.append('<text x="120" y="124" class="label">不再测量：</text>')
    _note(parts, 120, 154, "1. 底座高度")
    _note(parts, 120, 180, "2. 底座宽度")
    _note(parts, 120, 206, "3. 底座视觉 footprint")
    parts.append('<text x="430" y="124" class="label">仍需测量：</text>')
    _note(parts, 430, 154, "两个底座在 X 方向的相对距离")
    _note(parts, 430, 180, "首选测 LC 到 RC，即两个底座参考中心距离")
    _note(parts, 430, 206, "量不到中心时，再用内侧边间距或外侧总跨度")
    _note(parts, 90, 455, "这个纠偏避免把已经正确的 ALOHA1 USD 底座视觉重新建错。")
    _finish(parts, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    render_top(config, args.output_dir / "a11_base_anchor_measurement_top.svg")
    render_side(config, args.output_dir / "a11_base_anchor_measurement_side.svg")
    print(args.output_dir / "a11_base_anchor_measurement_top.svg")
    print(args.output_dir / "a11_base_anchor_measurement_side.svg")


if __name__ == "__main__":
    main()
