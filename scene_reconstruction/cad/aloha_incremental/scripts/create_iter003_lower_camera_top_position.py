"""Create Iteration 003 with the original lower camera moved to the top-frame position.

The user's drawing is interpreted as a top/plan-view measurement in millimeters:

* 640 mm from the left outside edge to the new camera center.
* 580 mm from the new camera center to the right outside edge.
* 260 mm is a Y-direction inner clearance/extension, not a Z height.

The original `REF_SCENE_frame_wormseye_mount_30` mesh is moved and rotated.
No green block is used as a camera substitute.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import FreeCAD
from FreeCAD import Base
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/home/eii/project/openpi0.5-rtc-reward-learning")
WORKDIR = ROOT / "scene_reconstruction" / "cad" / "aloha_incremental"
ITER2_DIR = WORKDIR / "iterations" / "iter_002_measured_aloha_y_offset"
ITER3_DIR = WORKDIR / "iterations" / "iter_003_lower_camera_top_position"
INPUT_FCSTD = ITER2_DIR / "iter_002_measured_aloha_y_offset.FCStd"
OUTPUT_FCSTD = ITER3_DIR / "iter_003_lower_camera_top_position.FCStd"

LOWER_CAMERA_MOUNT_OBJECT = "REF_SCENE_frame_wormseye_mount_30"

LEFT_OUTER_DISTANCE_MM = 640.0
RIGHT_OUTER_DISTANCE_MM = 580.0
INNER_CLEARANCE_Y_MM = 260.0
STEEL_PROFILE_MM = 20.0
CAMERA_ROTATION_ABOUT_Z_RAD = math.pi
SUPPORT_PIPE_REFERENCE_OBJECTS = (
    "REF_SCENE_frame_extrusion_1000_25",
    "REF_SCENE_frame_extrusion_1000_16",
    "REF_SCENE_frame_extrusion_1000_21",
    "REF_SCENE_frame_extrusion_1000_22",
)
SUPPORT_PIPE_Z_REFERENCE_OBJECTS = (
    "REF_SCENE_frame_extrusion_150_3",
    "REF_SCENE_frame_extrusion_150_6",
)


def _bbox(obj):
    if hasattr(obj, "Mesh"):
        return obj.Mesh.BoundBox
    if hasattr(obj, "Shape"):
        return obj.Shape.BoundBox
    raise TypeError(f"Object {obj.Name} has no Mesh or Shape bbox")


def _center(box) -> tuple[float, float, float]:
    return (
        (box.XMin + box.XMax) / 2.0,
        (box.YMin + box.YMax) / 2.0,
        (box.ZMin + box.ZMax) / 2.0,
    )


def _round3(values) -> list[float]:
    return [round(float(v), 3) for v in values]


def _set_color(obj, color: tuple[float, float, float], transparency: int = 0) -> None:
    try:
        obj.ViewObject.ShapeColor = color
        obj.ViewObject.Transparency = transparency
    except Exception:
        pass


def _move_mesh_center_and_rotate_z(
    obj,
    target_center: tuple[float, float, float],
    rotation_rad: float,
) -> dict[str, object]:
    box = _bbox(obj)
    before = _center(box)
    cos_v = math.cos(rotation_rad)
    sin_v = math.sin(rotation_rad)
    matrix = Base.Matrix()
    matrix.A11 = cos_v
    matrix.A12 = -sin_v
    matrix.A21 = sin_v
    matrix.A22 = cos_v
    matrix.A33 = 1.0
    # Rotate around the original mesh center, then translate to target_center.
    matrix.A14 = target_center[0] - (cos_v * before[0] - sin_v * before[1])
    matrix.A24 = target_center[1] - (sin_v * before[0] + cos_v * before[1])
    matrix.A34 = target_center[2] - before[2]
    moved_mesh = obj.Mesh.copy()
    moved_mesh.transform(matrix)
    obj.Mesh = moved_mesh
    after_box = _bbox(obj)
    after = _center(after_box)
    return {
        "center_before_mm": _round3(before),
        "center_after_mm": _round3(after),
        "delta_center_mm": _round3(
            (
                target_center[0] - before[0],
                target_center[1] - before[1],
                target_center[2] - before[2],
            )
        ),
        "rotation_about_z_deg": round(math.degrees(rotation_rad), 3),
        "direction_after_move": "camera/mount is rotated 180 deg about Z so it faces negative Y from the positive-Y side",
        "bbox_after_mm": _round3(
            [after_box.XMin, after_box.XMax, after_box.YMin, after_box.YMax, after_box.ZMin, after_box.ZMax]
        ),
    }


def _add_box(
    doc,
    name: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    transparency: int = 0,
):
    obj = doc.addObject("Part::Box", name)
    obj.Length, obj.Width, obj.Height = size
    obj.Placement.Base = Base.Vector(
        center[0] - size[0] / 2.0,
        center[1] - size[1] / 2.0,
        center[2] - size[2] / 2.0,
    )
    _set_color(obj, color, transparency)
    try:
        obj.addProperty("App::PropertyBool", "ReferenceLocked", "Reference")
        obj.ReferenceLocked = True
        obj.addProperty("App::PropertyString", "SourceAsset", "Reference")
        obj.SourceAsset = "user measured construction proxy, not original ALOHA geometry"
    except Exception:
        pass
    return obj


def _write_review_png(report: dict[str, object]) -> None:
    out = ITER3_DIR / "top_dimension_review.png"
    w, h = 1400, 900
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.load_default()

    table_x_min, table_x_max = report["reference_edges"]["x_outer_edges_mm"]
    table_y_min, table_y_max = report["reference_edges"]["y_outer_edges_mm"]
    camera_center = report["new_camera_position"]["center_mm"]
    mount_before = report["moved_mount"]["center_before_mm"]

    x_min = table_x_min - 110
    x_max = table_x_max + 110
    y_min = table_y_min - 80
    y_max = table_y_max + 80

    def sx(x):
        return 90 + (x - x_min) / (x_max - x_min) * (w - 180)

    def sy(y):
        return h - 90 - (y - y_min) / (y_max - y_min) * (h - 180)

    # Table.
    draw.rectangle([sx(table_x_min), sy(table_y_max), sx(table_x_max), sy(table_y_min)], fill=(190, 232, 242, 210), outline=(120, 140, 150, 255), width=3)
    draw.text((sx(table_x_min), sy(table_y_max) + 8), "table top view: X-Y plane", fill=(40, 60, 70), font=font)

    # Opposite side rail line.
    draw.line([sx(table_x_min), sy(camera_center[1]), sx(table_x_max), sy(camera_center[1])], fill=(35, 35, 38, 220), width=8)
    draw.text((sx(table_x_min), sy(camera_center[1]) - 28), "opposite-side steel frame line", fill=(25, 25, 28), font=font)

    # Four 260 mm Y-direction support pipes.
    for pipe in report["support_pipes_260mm"]:
        cx, cy, _ = pipe["center_mm"]
        sx0, sx1 = sx(cx - STEEL_PROFILE_MM / 2), sx(cx + STEEL_PROFILE_MM / 2)
        sy0, sy1 = sy(cy + INNER_CLEARANCE_Y_MM / 2), sy(cy - INNER_CLEARANCE_Y_MM / 2)
        draw.rectangle([sx0, sy0, sx1, sy1], fill=(35, 35, 38, 220), outline=(10, 10, 10, 255), width=2)

    # Moved real camera/mount footprint.
    draw.ellipse([sx(camera_center[0]) - 28, sy(camera_center[1]) - 28, sx(camera_center[0]) + 28, sy(camera_center[1]) + 28], fill=(0, 210, 50, 210), outline=(0, 130, 20, 255), width=4)
    draw.text((sx(camera_center[0]) + 34, sy(camera_center[1]) - 42), "moved original lower camera", fill=(0, 110, 20), font=font)
    draw.line([sx(camera_center[0]), sy(camera_center[1]), sx(camera_center[0]), sy(camera_center[1] - 90)], fill=(0, 160, 40, 255), width=4)
    draw.text((sx(camera_center[0]) + 12, sy(camera_center[1] - 90)), "faces -Y", fill=(0, 110, 20), font=font)

    # Old mount location in this front projection.
    draw.ellipse([sx(mount_before[0]) - 18, sy(mount_before[1]) - 18, sx(mount_before[0]) + 18, sy(mount_before[1]) + 18], outline=(220, 30, 30, 255), width=5)
    draw.text((sx(mount_before[0]) + 24, sy(mount_before[1]) + 8), "old mount", fill=(190, 30, 30), font=font)
    draw.line([sx(mount_before[0]), sy(mount_before[1]), sx(camera_center[0]), sy(camera_center[1])], fill=(0, 220, 40, 180), width=3)

    # Dimension arrows, simplified.
    dim_y = sy(camera_center[1]) - 45
    draw.line([sx(table_x_min), dim_y, sx(camera_center[0]), dim_y], fill=(230, 20, 20, 255), width=4)
    draw.line([sx(camera_center[0]), dim_y, sx(table_x_max), dim_y], fill=(230, 20, 20, 255), width=4)
    draw.text(((sx(table_x_min) + sx(camera_center[0])) / 2 - 18, dim_y - 28), "640", fill=(20, 20, 20), font=font)
    draw.text(((sx(camera_center[0]) + sx(table_x_max)) / 2 - 18, dim_y - 28), "580", fill=(20, 20, 20), font=font)

    dim_x = sx(camera_center[0]) - 105
    draw.line([dim_x, sy(camera_center[1]), dim_x, sy(camera_center[1] - INNER_CLEARANCE_Y_MM)], fill=(230, 20, 20, 255), width=4)
    draw.text((dim_x + 10, (sy(camera_center[1]) + sy(camera_center[1] - INNER_CLEARANCE_Y_MM)) / 2 - 8), "260 along Y", fill=(20, 20, 20), font=font)

    draw.text((40, 28), "iter_003_lower_camera_top_position top-view review", fill=(20, 25, 30), font=font)
    draw.text((40, 52), "+X horizontal, +Y depth; Z remains at the original lower-camera height", fill=(70, 76, 85), font=font)
    img.save(out)


def _write_tutorial(report: dict[str, object]) -> None:
    tutorial = ITER3_DIR / "freecad_operation_tutorial.md"
    tutorial.write_text(
        f"""# FreeCAD 初学者操作教程：把 lower camera 移到图中绿色位置

本教程对应 `iter_003_lower_camera_top_position.FCStd`。

## 1. 这次修改到底做了什么

你给的示意图这次按俯视/平面图理解：横向是 X，图中上下方向是 Y，不是 Z。

- `640 mm`：从左侧外边到新相机中心。
- `580 mm`：从新相机中心到右侧外边。
- `260 mm`：是 Y 轴方向的内边距/延伸，不是相机高度。
- 相机的 Z 高度保持原 lower camera 的高度，不再抬到 `270 mm`。
- 原始 lower camera mesh 会移动到目标位置，并绕 Z 轴旋转 `180 deg`，使它在对侧朝向 `-Y`。
- 新增 4 根沿 `+Y` 方向延伸的 `260 mm` 长安装钢管，表示你图中蓝色竖向钢管；CAD 中是深灰色，不是蓝色。
- 这 4 根钢管放在上方相机架高度，而不是桌面底层框架高度；否则会和原有底层框架重叠，看起来像没有新增。
- 新相机中心位置是 `{report["new_camera_position"]["center_mm"]}` mm。

因此，核心计算是：

```text
x = left_outer_edge + 640 = right_outer_edge - 580
y = mirror original lower-camera side to the opposite table side
z = original lower-camera z
rotation about Z = 180 deg
four support pipes = 20 x 260 x 20 mm, along +Y
```

在当前模型中：

```text
x = -610 + 640 = 30 mm
x =  610 - 580 = 30 mm
y = +323.856 mm
z = 17.429 mm
```

## 2. 如何打开这个结果

在终端运行：

```bash
/snap/bin/freecad scene_reconstruction/cad/aloha_incremental/scripts/open_iter003_lower_camera_top_position.py
```

不要直接双击 `.FCStd`。当前 snap 版 FreeCAD 直接打开文件有时不可见；脚本会打开文件、切到 MeshWorkbench、选中移动后的原始相机并 fit all。

打开后建议先看俯视图或用脚本默认视角检查 X-Y 位置。

## 3. 如何找到新相机

左侧模型树中找到：

```text
REF_SCENE_frame_wormseye_mount_30
```

这就是原来的 lower / worms-eye camera 安装件。iter_003 不再创建绿色方块来替代相机。

## 4. 不会拖动物体时，最稳的方法：改 Placement 数值

如果对象是普通 Part 对象：

1. 在左侧模型树点击对象。
2. 下方属性区域选择 `Data`。
3. 找到 `Placement`。
4. 展开 `Position`。
5. 修改 `x / y / z` 数值。
6. 按回车。
7. 点击 `View → Fit all` 检查。

优点：不会手抖，适合精确尺寸。

缺点：当前 `REF_SCENE_frame_wormseye_mount_30` 是导入 mesh，脚本中为了保持其真实几何，是直接平移 mesh 顶点；普通手工操作时更适合用下面的 Transform 或 Draft Move。

## 5. 用鼠标拖动物体：Transform

1. 在模型树选中对象。
2. 右键对象。
3. 选择 `Transform`。
4. 视图中会出现红、绿、蓝方向箭头。
5. 拖动箭头：
   - 红色通常表示 X。
   - 绿色通常表示 Y。
   - 蓝色通常表示 Z。
6. 右侧或任务面板中点击 `OK` 完成。

如果你只是粗略摆位置，可以这样做；如果要精确到毫米，建议最终仍然检查 Placement 或用脚本。

## 6. 如何复制物体

方式 A：复制粘贴

1. 在模型树选中对象。
2. 按 `Ctrl+C`。
3. 如果弹出依赖对象窗口，默认全选即可。
4. 按 `Ctrl+V`。
5. 新对象通常和原对象重叠。
6. 选中新对象，用 `Transform` 或 `Placement` 移开。

方式 B：Draft Move 的复制模式

1. 切到 `Draft` 工作台。
2. 选中对象。
3. 使用 `Move`。
4. 在任务面板里打开 `Copy`。
5. 选择起点和终点。

这相当于“移动时留下原件，生成副本”。

## 7. 这次如果你手工复现，应怎么做

最清晰的手工流程：

1. 打开 `iter_002_measured_aloha_y_offset.FCStd`。
2. 另存为 `iter_003_lower_camera_top_position.FCStd`。
3. 选中 `REF_SCENE_frame_wormseye_mount_30`。
4. 把它移动到：

```text
X = 30 mm
Y = 323.856 mm
Z = 17.429 mm
```

5. 绕 Z 轴旋转 `180 deg`，让它朝向 `-Y`。

6. 新建 4 根 `20 x 260 x 20 mm` 钢管，沿 `+Y`，对齐左右外侧和内侧钢架。

7. 检查俯视图中：
   - 左边外距到相机中心是 `640 mm`。
   - 相机中心到右边外距是 `580 mm`。
   - `260 mm` 是 Y 方向的内边距，不是 Z 高度。
   - 4 根安装钢管都沿 `+Y`，长度都是 `260 mm`。

## 8. 常见错误

- 不要把蓝色当成真实颜色。图里的蓝色只是你画图时表示钢架。
- 不要把 `260` 当成 Z 高度。它在这里是 Y 轴方向的尺寸。
- 不要用绿色方块替代真实相机；应该移动原始相机对象。
- 不要漏掉 4 根 `260 mm` 安装钢管，它们不是高度尺寸，而是俯视图中的 Y 方向延伸。
- 相机从负 Y 侧移动到正 Y 侧后，方向必须反过来，面向 `-Y`。
- 不要只看 3D 透视图判断位置。先用俯视图确认 X/Y，再用侧视图确认 Z 没被错误抬高。
- 复制对象后如果看不到变化，多半是副本和原件重叠了，需要移动副本。

## 9. 本次修改的验证数据

```json
{json.dumps(report, indent=2)}
```

## 10. 参考来源

- FreeCAD 官方文档 GitHub 镜像 `Draft_Move.md`：<https://github.com/FreeCAD/FreeCAD-documentation/blob/main/wiki/Draft_Move.md>
  - 该页说明 Draft Move 可以移动或复制选中对象，也支持脚本 `Draft.move(objects, vector, copy=False)`。
- FreeCAD 官方文档 GitHub 镜像 `Object_API.md`：<https://github.com/FreeCAD/FreeCAD-documentation/blob/main/wiki/Object_API.md>
  - 该页说明 FreeCAD 文档对象有 `Placement` 属性，用于描述对象的位置和姿态。

""",
        encoding="utf-8",
    )


def main() -> None:
    ITER3_DIR.mkdir(parents=True, exist_ok=True)
    doc = FreeCAD.openDocument(str(INPUT_FCSTD))
    doc.Label = "iter_003_lower_camera_top_position"

    table = doc.getObject("REF_TABLE_DESKTOP_PLANE")
    if table is None:
        raise RuntimeError("Missing REF_TABLE_DESKTOP_PLANE")
    table_box = _bbox(table)
    table_x_min = float(table_box.XMin)
    table_x_max = float(table_box.XMax)
    table_y_min = float(table_box.YMin)
    table_y_max = float(table_box.YMax)
    table_z_top = float(table_box.ZMax)
    outer_width = table_x_max - table_x_min
    expected_width = LEFT_OUTER_DISTANCE_MM + RIGHT_OUTER_DISTANCE_MM
    if abs(outer_width - expected_width) > 1e-6:
        raise RuntimeError(f"Outer distances do not match table width: {outer_width} vs {expected_width}")

    target_x_left = table_x_min + LEFT_OUTER_DISTANCE_MM
    target_x_right = table_x_max - RIGHT_OUTER_DISTANCE_MM
    if abs(target_x_left - target_x_right) > 1e-6:
        raise RuntimeError(f"Left/right target X mismatch: {target_x_left} vs {target_x_right}")
    target_x = (target_x_left + target_x_right) / 2.0

    mount = doc.getObject(LOWER_CAMERA_MOUNT_OBJECT)
    if mount is None or not hasattr(mount, "Mesh"):
        raise RuntimeError(f"Missing mesh object {LOWER_CAMERA_MOUNT_OBJECT}")
    mount_center_before = _center(_bbox(mount))

    # Move the original lower camera from the near negative-Y side to the
    # opposite positive-Y side. Keep its original Z height; the user's 260 mm
    # measurement is a Y-direction support-pipe length, not a Z value.
    target_y = abs(mount_center_before[1])
    target_z = mount_center_before[2]
    target_center = (target_x, target_y, target_z)

    move_result = _move_mesh_center_and_rotate_z(mount, target_center, CAMERA_ROTATION_ABOUT_Z_RAD)
    try:
        mount.addProperty("App::PropertyString", "MeasuredCameraPosition", "Reference")
    except Exception:
        pass
    try:
        mount.MeasuredCameraPosition = (
            "Moved to user measured top-frame position: "
            f"x={target_x:.3f}, y={target_y:.3f}, z={target_z:.3f} mm; "
            "rotated 180 deg about Z to face -Y."
        )
    except Exception:
        pass

    rail = _add_box(
        doc,
        "MEASURED_TOP_STEEL_RAIL_DARK_REFERENCE",
        center=(0.0, target_y, table_z_top + STEEL_PROFILE_MM / 2.0),
        size=(outer_width, STEEL_PROFILE_MM, STEEL_PROFILE_MM),
        color=(0.08, 0.08, 0.08),
        transparency=25,
    )
    support_pipe_objects = []
    support_y_center = target_y - INNER_CLEARANCE_Y_MM / 2.0
    z_refs = []
    for ref_name in SUPPORT_PIPE_Z_REFERENCE_OBJECTS:
        ref = doc.getObject(ref_name)
        if ref is None:
            raise RuntimeError(f"Missing support-pipe Z reference object {ref_name}")
        z_refs.append(_center(_bbox(ref))[2])
    support_z_center = sum(z_refs) / len(z_refs)
    for index, ref_name in enumerate(SUPPORT_PIPE_REFERENCE_OBJECTS, start=1):
        ref = doc.getObject(ref_name)
        if ref is None:
            raise RuntimeError(f"Missing support-pipe reference object {ref_name}")
        ref_center = _center(_bbox(ref))
        pipe = _add_box(
            doc,
            f"MEASURED_CAMERA_SUPPORT_PIPE_260MM_{index}",
            center=(ref_center[0], support_y_center, support_z_center),
            size=(STEEL_PROFILE_MM, INNER_CLEARANCE_Y_MM, STEEL_PROFILE_MM),
            color=(0.08, 0.08, 0.08),
            transparency=15,
        )
        support_pipe_objects.append(pipe)

    doc.recompute()
    doc.saveAs(str(OUTPUT_FCSTD))

    report = {
        "iteration": "iter_003_lower_camera_top_position",
        "units": "mm",
        "source_iteration": str(INPUT_FCSTD.relative_to(ROOT)),
        "output_freecad_file": str(OUTPUT_FCSTD.relative_to(ROOT)),
        "user_measurements": {
            "left_outer_distance_to_camera_center_mm": LEFT_OUTER_DISTANCE_MM,
            "right_outer_distance_to_camera_center_mm": RIGHT_OUTER_DISTANCE_MM,
            "support_pipe_y_extension_mm": INNER_CLEARANCE_Y_MM,
            "steel_profile_assumed_mm": STEEL_PROFILE_MM,
            "note": "640 and 580 are X-direction outer distances. 260 is a Y-direction support-pipe extension, not a Z height. The blue rail in the sketch is only a visual annotation, not the real CAD color.",
        },
        "reference_edges": {
            "x_outer_edges_mm": _round3([table_x_min, table_x_max]),
            "y_outer_edges_mm": _round3([table_y_min, table_y_max]),
            "outer_width_mm": round(outer_width, 3),
            "table_top_z_mm": round(table_z_top, 3),
        },
        "new_camera_position": {
            "center_mm": _round3(target_center),
            "x_from_left_outer_mm": round(target_x - table_x_min, 3),
            "x_from_right_outer_mm": round(table_x_max - target_x, 3),
            "z_policy": "kept from original lower camera; 260 mm is not used as Z",
            "faces": "-Y after 180 deg Z rotation",
        },
        "moved_mount": {
            "object": LOWER_CAMERA_MOUNT_OBJECT,
            **move_result,
        },
        "top_steel_rail_proxy": {
            "object": rail.Name,
            "center_mm": _round3(_center(_bbox(rail))),
            "size_mm": _round3([rail.Length.Value, rail.Width.Value, rail.Height.Value]),
            "color": "dark gray, not sketch blue",
        },
        "support_pipes_260mm": [
            {
                "object": pipe.Name,
                "center_mm": _round3(_center(_bbox(pipe))),
                "size_mm": _round3([pipe.Length.Value, pipe.Width.Value, pipe.Height.Value]),
                "axis": "+Y",
                "length_mm": INNER_CLEARANCE_Y_MM,
                "z_source": "aligned to existing upper camera-frame rails",
            }
            for pipe in support_pipe_objects
        ],
    }
    (ITER3_DIR / "bbox_and_dimensions.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (ITER3_DIR / "changes.md").write_text(
        "\n".join(
            [
                "# Iteration 003 Lower Camera Top Position",
                "",
                "## What Changed",
                "",
                f"- Started from `{INPUT_FCSTD.relative_to(ROOT)}`.",
                f"- Moved the original `{LOWER_CAMERA_MOUNT_OBJECT}` to the measured target; no green cube substitute is created.",
                "- Rotated the moved camera/mount 180 deg about Z so it faces -Y.",
                "- Added `MEASURED_TOP_STEEL_RAIL_DARK_REFERENCE` as a dark gray rail proxy.",
                "- Added four `MEASURED_CAMERA_SUPPORT_PIPE_260MM_*` pipes, each 260 mm long along +Y and aligned to the upper camera-frame height.",
                "",
                "## Measurement Interpretation",
                "",
                "- `640 mm` and `580 mm` are outer horizontal distances.",
                "- `260 mm` is a Y-direction support-pipe length, not a Z height.",
                "- The lower-camera Z height is preserved from the original camera mesh.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _write_review_png(report)
    _write_tutorial(report)
    print(json.dumps(report, indent=2))


main()
