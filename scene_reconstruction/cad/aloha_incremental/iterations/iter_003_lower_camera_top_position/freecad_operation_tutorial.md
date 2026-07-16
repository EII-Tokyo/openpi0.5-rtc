# FreeCAD 初学者操作教程：把 lower camera 移到图中绿色位置

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
- 新相机中心位置是 `[30.0, 323.856, 17.429]` mm。

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
{
  "iteration": "iter_003_lower_camera_top_position",
  "units": "mm",
  "source_iteration": "scene_reconstruction/cad/aloha_incremental/iterations/iter_002_measured_aloha_y_offset/iter_002_measured_aloha_y_offset.FCStd",
  "output_freecad_file": "scene_reconstruction/cad/aloha_incremental/iterations/iter_003_lower_camera_top_position/iter_003_lower_camera_top_position.FCStd",
  "user_measurements": {
    "left_outer_distance_to_camera_center_mm": 640.0,
    "right_outer_distance_to_camera_center_mm": 580.0,
    "support_pipe_y_extension_mm": 260.0,
    "steel_profile_assumed_mm": 20.0,
    "note": "640 and 580 are X-direction outer distances. 260 is a Y-direction support-pipe extension, not a Z height. The blue rail in the sketch is only a visual annotation, not the real CAD color."
  },
  "reference_edges": {
    "x_outer_edges_mm": [
      -610.0,
      610.0
    ],
    "y_outer_edges_mm": [
      -312.5,
      312.5
    ],
    "outer_width_mm": 1220.0,
    "table_top_z_mm": 0.0
  },
  "new_camera_position": {
    "center_mm": [
      30.0,
      323.856,
      17.429
    ],
    "x_from_left_outer_mm": 640.0,
    "x_from_right_outer_mm": 580.0,
    "z_policy": "kept from original lower camera; 260 mm is not used as Z",
    "faces": "-Y after 180 deg Z rotation"
  },
  "moved_mount": {
    "object": "REF_SCENE_frame_wormseye_mount_30",
    "center_before_mm": [
      0.0,
      -323.856,
      17.429
    ],
    "center_after_mm": [
      30.0,
      323.856,
      17.429
    ],
    "delta_center_mm": [
      30.0,
      647.712,
      0.0
    ],
    "rotation_about_z_deg": 180.0,
    "direction_after_move": "camera/mount is rotated 180 deg about Z so it faces negative Y from the positive-Y side",
    "bbox_after_mm": [
      -10.331,
      70.331,
      308.727,
      338.986,
      -20.0,
      54.858
    ]
  },
  "top_steel_rail_proxy": {
    "object": "MEASURED_TOP_STEEL_RAIL_DARK_REFERENCE",
    "center_mm": [
      0.0,
      323.856,
      10.0
    ],
    "size_mm": [
      1220.0,
      20.0,
      20.0
    ],
    "color": "dark gray, not sketch blue"
  },
  "support_pipes_260mm": [
    {
      "object": "MEASURED_CAMERA_SUPPORT_PIPE_260MM_1",
      "center_mm": [
        -604.959,
        193.856,
        610.0
      ],
      "size_mm": [
        20.0,
        260.0,
        20.0
      ],
      "axis": "+Y",
      "length_mm": 260.0,
      "z_source": "aligned to existing upper camera-frame rails"
    },
    {
      "object": "MEASURED_CAMERA_SUPPORT_PIPE_260MM_2",
      "center_mm": [
        -433.554,
        193.856,
        610.0
      ],
      "size_mm": [
        20.0,
        260.0,
        20.0
      ],
      "axis": "+Y",
      "length_mm": 260.0,
      "z_source": "aligned to existing upper camera-frame rails"
    },
    {
      "object": "MEASURED_CAMERA_SUPPORT_PIPE_260MM_3",
      "center_mm": [
        433.554,
        193.856,
        610.0
      ],
      "size_mm": [
        20.0,
        260.0,
        20.0
      ],
      "axis": "+Y",
      "length_mm": 260.0,
      "z_source": "aligned to existing upper camera-frame rails"
    },
    {
      "object": "MEASURED_CAMERA_SUPPORT_PIPE_260MM_4",
      "center_mm": [
        604.959,
        193.856,
        610.0
      ],
      "size_mm": [
        20.0,
        260.0,
        20.0
      ],
      "axis": "+Y",
      "length_mm": 260.0,
      "z_source": "aligned to existing upper camera-frame rails"
    }
  ]
}
```

## 10. 参考来源

- FreeCAD 官方文档 GitHub 镜像 `Draft_Move.md`：<https://github.com/FreeCAD/FreeCAD-documentation/blob/main/wiki/Draft_Move.md>
  - 该页说明 Draft Move 可以移动或复制选中对象，也支持脚本 `Draft.move(objects, vector, copy=False)`。
- FreeCAD 官方文档 GitHub 镜像 `Object_API.md`：<https://github.com/FreeCAD/FreeCAD-documentation/blob/main/wiki/Object_API.md>
  - 该页说明 FreeCAD 文档对象有 `Placement` 属性，用于描述对象的位置和姿态。

