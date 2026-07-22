#!/usr/bin/env python3
"""Render A12 pipe/bottle measurement worksheet SVGs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml


DEFAULT_CONFIG = Path("aloha_isaac_rebuild/configs/physical_reconstruction/pipe_bottle_measurement_worksheet.yaml")
DEFAULT_OUTPUT_DIR = Path("aloha_isaac_rebuild/artifacts/screenshots")


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


def _candidate(config: dict, section: str, name: str) -> float:
    return float(config[section][name]["value"])


def render_pipe(config: dict, output: Path) -> None:
    width, height = 980, 650
    parts = _svg_header(
        width,
        height,
        "A12 水管测量图：只记录几何，不做碰撞",
        "历史值先作为候选：长度约22.5cm、外径约5mm、角度约44度；都需要现场复核。",
    )
    table_y = 460
    origin_x = 150
    base_x = 500
    base_top_y = table_y - 70
    pipe_len_px = 260
    angle = math.radians(_candidate(config, "pipe_candidate_values", "angle_deg"))
    pipe_dx = -math.cos(angle) * pipe_len_px
    pipe_dy = -math.sin(angle) * pipe_len_px
    entry_x = base_x + pipe_dx
    entry_y = base_top_y + pipe_dy

    parts.append(f'<line x1="80" y1="{table_y}" x2="900" y2="{table_y}" stroke="#8b7e66" stroke-width="4"/>')
    _note(parts, 82, table_y + 24, "桌面/安装平面")
    parts.append(f'<rect x="{base_x-42}" y="{base_top_y}" width="84" height="70" fill="#8a8f94" fill-opacity="0.72" stroke="#51565c" stroke-width="2"/>')
    _note(parts, base_x - 55, base_top_y - 12, "水管底座")
    parts.append(f'<line x1="{base_x:.1f}" y1="{base_top_y:.1f}" x2="{entry_x:.1f}" y2="{entry_y:.1f}" stroke="#d7263d" stroke-width="8" stroke-linecap="round"/>')
    parts.append(f'<line x1="{base_x:.1f}" y1="{base_top_y:.1f}" x2="{entry_x:.1f}" y2="{entry_y:.1f}" stroke="#ffffff" stroke-width="2" stroke-linecap="round"/>')
    _point(parts, base_x, base_top_y, "P0", "#6b7280")
    _point(parts, entry_x, entry_y, "PE", "#d7263d")
    _dimension(parts, base_x + 70, table_y, base_x + 70, base_top_y, "H 高度约7cm？", base_x + 84, (table_y + base_top_y) / 2)
    _dimension(parts, base_x, base_top_y - 38, entry_x, entry_y - 38, "L 水管长度约22.5cm？", min(base_x, entry_x) + 34, entry_y - 58)
    _dimension(parts, base_x - 15, base_top_y + 22, base_x - 210, base_top_y + 22, "D 底座到边约9.5cm？", base_x - 205, base_top_y + 44)
    _note(parts, 90, 110, "需要回复：pipe_outer_diameter_mm=..., pipe_inner_diameter_mm=...")
    _note(parts, 90, 138, "需要回复：pipe_length_cm=..., pipe_angle_deg=..., pipe_points_toward=...")
    _note(parts, 90, 166, "需要回复：pipe_entry_x_from_left_edge_cm=..., pipe_entry_y_from_cam_high_edge_cm=..., pipe_entry_z_above_table_cm=...")
    _note(parts, 90, 595, "P0 是底座/水管连接点，PE 是水管入口端中心。请确认方向确实朝左臂/瓶口插入方向。")
    _finish(parts, output)


def render_bottle(config: dict, output: Path) -> None:
    width, height = 980, 650
    parts = _svg_header(
        width,
        height,
        "A12 瓶子测量图：真实瓶子参数待确认",
        "现有 Bottle500 资产候选值：高约20.6cm、最大直径约68mm、瓶口内径约25mm。",
    )
    cx, cy = 460, 345
    body_w, body_h = 310, 92
    neck_w, neck_h = 84, 46
    mouth_w = 42
    parts.append(f'<rect x="{cx-body_w/2:.1f}" y="{cy-body_h/2:.1f}" width="{body_w:.1f}" height="{body_h:.1f}" rx="44" fill="#bde8f4" fill-opacity="0.55" stroke="#3b93a8" stroke-width="3"/>')
    parts.append(f'<rect x="{cx+body_w/2-10:.1f}" y="{cy-neck_h/2:.1f}" width="{neck_w:.1f}" height="{neck_h:.1f}" rx="14" fill="#bde8f4" fill-opacity="0.55" stroke="#3b93a8" stroke-width="3"/>')
    parts.append(f'<rect x="{cx+body_w/2+neck_w-10:.1f}" y="{cy-mouth_w/2:.1f}" width="26" height="{mouth_w:.1f}" rx="5" fill="#e7fbff" stroke="#3b93a8" stroke-width="3"/>')
    parts.append(f'<rect x="{cx-70:.1f}" y="{cy-body_h/2-25:.1f}" width="95" height="{body_h+50:.1f}" fill="#ff9f1c" fill-opacity="0.16" stroke="#e76f00" stroke-width="2" stroke-dasharray="8 6"/>')
    _note(parts, cx - 86, cy - body_h/2 - 36, "常见夹持区待测")
    _dimension(parts, cx-body_w/2, cy+body_h/2+48, cx+body_w/2+neck_w+16, cy+body_h/2+48, "瓶子总长/高度？", cx - 98, cy + body_h/2 + 72)
    _dimension(parts, cx-120, cy-body_h/2, cx-120, cy+body_h/2, "瓶身直径？", cx - 246, cy + 6)
    _dimension(parts, cx+body_w/2+neck_w+50, cy-mouth_w/2, cx+body_w/2+neck_w+50, cy+mouth_w/2, "瓶口内外径？", cx + body_w/2 + neck_w + 62, cy + 6)
    _dimension(parts, cx-body_w/2, cy-body_h/2-48, cx-22, cy-body_h/2-48, "瓶底到抓取中心？", cx - 142, cy - body_h/2 - 66)
    _note(parts, 90, 108, "需要回复：bottle_empty_mass_g=...")
    _note(parts, 90, 136, "需要回复：bottle_max_diameter_mm=..., bottle_grasp_diameter_mm=...")
    _note(parts, 90, 164, "需要回复：bottle_mouth_outer_diameter_mm=..., bottle_mouth_inner_diameter_mm=...")
    _note(parts, 90, 192, "需要回复：bottle_bottom_to_grasp_center_cm=...")
    _note(parts, 90, 592, "PET 瓶会被夹扁：真实 grasp_diameter 可能小于自由状态最大直径，需要单独记录。")
    _finish(parts, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    render_pipe(config, args.output_dir / "a12_pipe_measurement.svg")
    render_bottle(config, args.output_dir / "a12_bottle_measurement.svg")
    print(args.output_dir / "a12_pipe_measurement.svg")
    print(args.output_dir / "a12_bottle_measurement.svg")


if __name__ == "__main__":
    main()
