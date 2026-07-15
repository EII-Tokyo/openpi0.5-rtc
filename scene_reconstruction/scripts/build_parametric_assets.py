#!/usr/bin/env python3
"""Build parametric CAD-style drawings and OpenUSD proxy assets from YAML."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "scene_reconstruction"
CONFIG = OUT / "config/scene_parameters.yaml"
CAD = OUT / "cad"
DRAWINGS = CAD / "drawings"
USD = OUT / "usd"
BASE_ALOHA_USD = (
    REPO_ROOT
    / "local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose"
    / "aloha2_menagerie_scene_deep_black_real_start_pose.usd"
)


def pget(params: dict[str, Any], key: str) -> Any:
    return params["parameters"][key]["value"]


def svg_header(width: int = 1000, height: int = 700) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f5ef"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:18px;fill:#222} .dim{font-size:14px;fill:#555}</style>',
    ]


def line(x1: float, y1: float, x2: float, y2: float, color: str = "#222", width: float = 3, dash: str = "") -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{dash_attr}/>'


def rect(cx: float, cy: float, w: float, h: float, fill: str, stroke: str = "#222", opacity: float = 1.0) -> str:
    return (
        f'<rect x="{cx - w/2:.1f}" y="{cy - h/2:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="2" opacity="{opacity}"/>'
    )


def circle(cx: float, cy: float, r: float, fill: str, stroke: str = "#222") -> str:
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'


def save_svg(path: Path, parts: list[str]) -> None:
    path.write_text("\n".join(parts + ["</svg>\n"]), encoding="utf-8")


def world_to_top(x: float, y: float) -> tuple[float, float]:
    scale = 560.0
    return 500 + x * scale, 360 - y * scale


def world_to_front(x: float, z: float) -> tuple[float, float]:
    scale = 560.0
    return 500 + x * scale, 560 - z * scale


def world_to_side(y: float, z: float) -> tuple[float, float]:
    scale = 560.0
    return 500 + y * scale, 560 - z * scale


def build_drawings(params: dict[str, Any]) -> None:
    DRAWINGS.mkdir(parents=True, exist_ok=True)
    table = pget(params, "table.size")
    rack_outer = pget(params, "rack.outer_size")
    pipe_start = pget(params, "pipe.axis_start")
    pipe_end = pget(params, "pipe.axis_end")
    cam_low = pget(params, "camera.cam_low.pose")["translation"]
    cam_right = pget(params, "camera.cam_right_wrist_hint.pose")["translation"]

    # Top view.
    parts = svg_header()
    parts.append("<text x='40' y='45'>Top view: table, rack, pipe axis and proxy cameras</text>")
    tx, ty = world_to_top(0, 0)
    parts.append(rect(tx, ty, table[0] * 560, table[1] * 560, "#d8c28a", "#8a6d2f", 0.85))
    parts.append(rect(tx, ty, rack_outer[0] * 560, rack_outer[1] * 560, "none", "#111", 1.0))
    for x in (-rack_outer[0] / 2, rack_outer[0] / 2):
        for y in (-rack_outer[1] / 2, rack_outer[1] / 2):
            px, py = world_to_top(x, y)
            parts.append(rect(px, py, 18, 18, "#111", "#111"))
    sx, sy = world_to_top(pipe_start[0], pipe_start[1])
    ex, ey = world_to_top(pipe_end[0], pipe_end[1])
    parts.append(line(sx, sy, ex, ey, "#d7191c", 6))
    parts.append(circle(sx, sy, 8, "#d7191c"))
    parts.append(circle(ex, ey, 10, "#ffb3b3", "#d7191c"))
    for label, cam, color in (("cam_low", cam_low, "#2563eb"), ("right_hint", cam_right, "#7c3aed")):
        cx, cy = world_to_top(cam[0], cam[1])
        parts.append(rect(cx, cy, 34, 22, color, "#222"))
        parts.append(f"<text x='{cx + 18:.1f}' y='{cy - 10:.1f}' class='dim'>{label}</text>")
    parts.append("<text x='40' y='665' class='dim'>Measured: table and pipe. Estimated: rack and exact camera positions.</text>")
    save_svg(DRAWINGS / "top.svg", parts)

    # Front view: x-z.
    parts = svg_header()
    parts.append("<text x='40' y='45'>Front view: rack height and pipe tilt</text>")
    parts.append(line(*world_to_front(-table[0] / 2, 0), *world_to_front(table[0] / 2, 0), "#8a6d2f", 5))
    for x in (-rack_outer[0] / 2, rack_outer[0] / 2):
        x0, z0 = world_to_front(x, 0)
        x1, z1 = world_to_front(x, rack_outer[2])
        parts.append(line(x0, z0, x1, z1, "#111", 8))
    parts.append(line(*world_to_front(-rack_outer[0] / 2, rack_outer[2]), *world_to_front(rack_outer[0] / 2, rack_outer[2]), "#111", 8))
    parts.append(line(*world_to_front(pipe_start[0], pipe_start[2]), *world_to_front(pipe_end[0], pipe_end[2]), "#d7191c", 6))
    parts.append("<text x='40' y='665' class='dim'>Pipe length and tilt use user measurement; rack height is estimated.</text>")
    save_svg(DRAWINGS / "front.svg", parts)

    # Side view: y-z.
    parts = svg_header()
    parts.append("<text x='40' y='45'>Side view: table edge, pipe outside offset and camera height</text>")
    parts.append(line(*world_to_side(-table[1] / 2, 0), *world_to_side(table[1] / 2, 0), "#8a6d2f", 5))
    parts.append(line(*world_to_side(table[1] / 2, 0), *world_to_side(pipe_start[1], 0), "#666", 3, "6 4"))
    parts.append(line(*world_to_side(pipe_start[1], pipe_start[2]), *world_to_side(pipe_end[1], pipe_end[2]), "#d7191c", 6))
    for label, cam, color in (("cam_low", cam_low, "#2563eb"), ("right_hint", cam_right, "#7c3aed")):
        cx, cy = world_to_side(cam[1], cam[2])
        parts.append(rect(cx, cy, 34, 22, color, "#222"))
        parts.append(f"<text x='{cx + 18:.1f}' y='{cy - 10:.1f}' class='dim'>{label}</text>")
    save_svg(DRAWINGS / "side.svg", parts)

    # Simple isometric PNG.
    img = Image.new("RGB", (1200, 820), (246, 244, 238))
    draw = ImageDraw.Draw(img)

    def iso(x: float, y: float, z: float) -> tuple[int, int]:
        scale = 420
        return int(600 + (x - y) * scale * 0.72), int(530 - z * scale + (x + y) * scale * 0.22)

    corners = [(-table[0] / 2, -table[1] / 2, 0), (table[0] / 2, -table[1] / 2, 0), (table[0] / 2, table[1] / 2, 0), (-table[0] / 2, table[1] / 2, 0)]
    draw.polygon([iso(*c) for c in corners], fill=(214, 194, 138), outline=(120, 95, 43))
    rack = [(-rack_outer[0] / 2, -rack_outer[1] / 2), (rack_outer[0] / 2, -rack_outer[1] / 2), (rack_outer[0] / 2, rack_outer[1] / 2), (-rack_outer[0] / 2, rack_outer[1] / 2)]
    for x, y in rack:
        draw.line([iso(x, y, 0), iso(x, y, rack_outer[2])], fill=(15, 15, 15), width=8)
    for a, b in zip(rack, rack[1:] + rack[:1], strict=True):
        draw.line([iso(a[0], a[1], rack_outer[2]), iso(b[0], b[1], rack_outer[2])], fill=(15, 15, 15), width=8)
    draw.line([iso(*pipe_start), iso(*pipe_end)], fill=(210, 20, 20), width=8)
    for cam, color in ((cam_low, (37, 99, 235)), (cam_right, (124, 58, 237))):
        cx, cy = iso(*cam)
        draw.rectangle([cx - 14, cy - 10, cx + 14, cy + 10], fill=color, outline=(20, 20, 20))
    draw.text((34, 30), "Isometric proxy CAD view: measured table/pipe, estimated rack/cameras", fill=(20, 20, 20))
    img.save(DRAWINGS / "isometric.png")


def usd_header(sub_layers: list[str] | None = None) -> str:
    if not sub_layers:
        return '#usda 1.0\n(\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n\n'
    entries = ",\n".join(f"        @{layer}@" for layer in sub_layers)
    return f'#usda 1.0\n(\n    metersPerUnit = 1\n    upAxis = "Z"\n    subLayers = [\n{entries}\n    ]\n)\n\n'


def cube_prim(path: str, translation: list[float], scale: list[float], color: tuple[float, float, float]) -> str:
    x, y, z = translation
    sx, sy, sz = scale
    return f'''    def Cube "{path}" {{
        color3f[] primvars:displayColor = [({color[0]}, {color[1]}, {color[2]})]
        double size = 1
        double3 xformOp:translate = ({x}, {y}, {z})
        double3 xformOp:scale = ({sx}, {sy}, {sz})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }}
'''


def camera_prim(name: str, pose: dict[str, Any], focal_length: float) -> str:
    x, y, z = pose["translation"]
    # Keep proxy camera transform simple and evidence-linked; detailed extrinsics remain unknown.
    return f'''    def Camera "{name}" {{
        double focalLength = {focal_length}
        double horizontalAperture = 20.955
        double verticalAperture = 15.2908
        double2 clippingRange = (0.01, 100.0)
        double3 xformOp:translate = ({x}, {y}, {z})
        uniform token[] xformOpOrder = ["xformOp:translate"]
        custom string source_status = "proxy_from_scene_parameters"
    }}
'''


def cylinder_matrix(start: list[float], end: list[float]) -> tuple[list[list[float]], float]:
    sx, sy, sz = start
    ex, ey, ez = end
    axis = [ex - sx, ey - sy, ez - sz]
    length = math.sqrt(sum(v * v for v in axis))
    z_axis = [v / length for v in axis]
    up = [0.0, 1.0, 0.0] if abs(z_axis[1]) < 0.95 else [1.0, 0.0, 0.0]
    x_axis = [
        up[1] * z_axis[2] - up[2] * z_axis[1],
        up[2] * z_axis[0] - up[0] * z_axis[2],
        up[0] * z_axis[1] - up[1] * z_axis[0],
    ]
    norm = math.sqrt(sum(v * v for v in x_axis))
    x_axis = [v / norm for v in x_axis]
    y_axis = [
        z_axis[1] * x_axis[2] - z_axis[2] * x_axis[1],
        z_axis[2] * x_axis[0] - z_axis[0] * x_axis[2],
        z_axis[0] * x_axis[1] - z_axis[1] * x_axis[0],
    ]
    mid = [(sx + ex) / 2.0, (sy + ey) / 2.0, (sz + ez) / 2.0]
    return [x_axis + [0.0], y_axis + [0.0], z_axis + [0.0], mid + [1.0]], length


def cylinder_prim(name: str, start: list[float], end: list[float], radius: float, color: tuple[float, float, float]) -> str:
    matrix, length = cylinder_matrix(start, end)
    rows = ",\n            ".join("(" + ", ".join(f"{v:.8f}" for v in row) + ")" for row in matrix)
    return f'''    def Cylinder "{name}" {{
        color3f[] primvars:displayColor = [({color[0]}, {color[1]}, {color[2]})]
        double height = {length:.8f}
        double radius = {radius:.8f}
        matrix4d xformOp:transform = ({rows})
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}
'''


def build_usd(params: dict[str, Any]) -> None:
    USD.mkdir(parents=True, exist_ok=True)
    table = pget(params, "table.size")
    rack_outer = pget(params, "rack.outer_size")
    rack_section = pget(params, "rack.profile_section")
    pipe_start = pget(params, "pipe.axis_start")
    pipe_end = pget(params, "pipe.axis_end")
    pipe_radius = pget(params, "pipe.radius")
    cam_low = pget(params, "camera.cam_low.pose")
    cam_right = pget(params, "camera.cam_right_wrist_hint.pose")
    focal = pget(params, "camera.focal_length")

    rack_parts = [usd_header(), 'def Xform "World" {\n', '  def Xform "CameraRack" {\n']
    # Four posts.
    for i, (x, y, z) in enumerate(pget(params, "rack.post_positions"), start=1):
        rack_parts.append(cube_prim(f"post_{i:02d}", [x, y, z], [rack_section[0], rack_section[1], rack_outer[2]], (0.02, 0.02, 0.02)))
    # Top beams.
    z = rack_outer[2]
    rack_parts.append(cube_prim("front_top_beam", [0, -rack_outer[1] / 2, z], [rack_outer[0], rack_section[1], rack_section[0]], (0.02, 0.02, 0.02)))
    rack_parts.append(cube_prim("rear_top_beam", [0, rack_outer[1] / 2, z], [rack_outer[0], rack_section[1], rack_section[0]], (0.02, 0.02, 0.02)))
    rack_parts.append(cube_prim("left_top_beam", [-rack_outer[0] / 2, 0, z], [rack_section[0], rack_outer[1], rack_section[0]], (0.02, 0.02, 0.02)))
    rack_parts.append(cube_prim("right_top_beam", [rack_outer[0] / 2, 0, z], [rack_section[0], rack_outer[1], rack_section[0]], (0.02, 0.02, 0.02)))
    rack_parts.append(cube_prim("cam_low_mount_plate", cam_low["translation"], [0.08, 0.025, 0.055], (0.12, 0.28, 0.85)))
    rack_parts.append(cube_prim("cam_right_mount_plate", cam_right["translation"], [0.08, 0.025, 0.055], (0.45, 0.20, 0.85)))
    rack_parts.append("  }\n}\n")
    (USD / "camera_rack.usda").write_text("".join(rack_parts), encoding="utf-8")

    pipe_parts = [
        usd_header(),
        'def Xform "World" {\n',
        '  def Xform "PipeAssembly" {\n',
        cube_prim("table_reference", [0, 0, -table[2] / 2], table, (0.80, 0.70, 0.42)),
        cube_prim("pipe_base_estimated", [pipe_start[0], pipe_start[1], 0.025], [0.12, 0.08, 0.05], (0.28, 0.28, 0.28)),
        cylinder_prim("pipe_axis", pipe_start, pipe_end, pipe_radius, (0.80, 0.80, 0.78)),
        cylinder_prim("pipe_centerline_red", pipe_start, pipe_end, 0.0012, (1.0, 0.05, 0.05)),
        '  }\n}\n',
    ]
    (USD / "pipe.usda").write_text("".join(pipe_parts), encoding="utf-8")

    layout_parts = [
        usd_header(),
        'def Xform "World" {\n',
        '  def Xform "ReconstructionCameras" {\n',
        camera_prim("cam_low_proxy", cam_low, focal),
        camera_prim("cam_right_wrist_hint_proxy", cam_right, focal),
        '  }\n',
        '}\n',
    ]
    (USD / "real_layout_override.usda").write_text("".join(layout_parts), encoding="utf-8")

    base_rel = Path(os.path.relpath(BASE_ALOHA_USD, USD)).as_posix()
    (USD / "aloha_real_scene.usda").write_text(
        usd_header([base_rel, "camera_rack.usda", "pipe.usda", "real_layout_override.usda"]).rstrip() + "\n",
        encoding="utf-8",
    )

    rack_params = {
        "route": "OpenUSD proxy geometry",
        "freecad_available": False,
        "source": "scene_reconstruction/config/scene_parameters.yaml",
        "outputs": {
            "camera_rack_usd": "scene_reconstruction/usd/camera_rack.usda",
            "pipe_usd": "scene_reconstruction/usd/pipe.usda",
            "scene_usd": "scene_reconstruction/usd/aloha_real_scene.usda",
        },
    }
    (CAD / "rack_parameters.yaml").write_text(yaml.safe_dump(rack_params, sort_keys=False), encoding="utf-8")


def main() -> None:
    params = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    build_drawings(params)
    build_usd(params)
    print(
        "\n".join(
            [
                "generated:",
                str(DRAWINGS / "top.svg"),
                str(DRAWINGS / "front.svg"),
                str(DRAWINGS / "side.svg"),
                str(DRAWINGS / "isometric.png"),
                str(USD / "aloha_real_scene.usda"),
            ]
        )
    )


if __name__ == "__main__":
    main()
