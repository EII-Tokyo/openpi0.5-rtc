# Iteration 003 研究与实现报告：移动原 lower camera 并补齐 260 mm 支撑钢管

## 调查目标

根据用户给出的示意图，把原始 lower / worms-eye camera 安装件移动到新的目标位置，并补齐图中沿 Y 方向延伸的 4 根 `260 mm` 安装钢管。

用户给出的尺寸单位均为 `mm`：

- `640`：左侧外边距到新相机中心。
- `580`：新相机中心到右侧外边距。
- `260`：Y 轴方向的安装钢管长度/内边距，不是 Z 轴高度。
- 蓝色只是用户在图中用于表示钢架的绘图颜色，不代表真实钢架颜色。

## 环境与版本

- 本机 FreeCAD：`FreeCAD 1.1.1 Revision: 44227 +647 (Git)`。
- 当前输入模型：`iter_002_measured_aloha_y_offset.FCStd`。
- 新输出模型：`iter_003_lower_camera_top_position.FCStd`。

## 关键结论

1. 当前 `iter_002` 的 scene-level lower camera 没有独立的 `d405_solid` 相机对象；可直接移动的对象是 lower / worms-eye camera 的安装件：

```text
REF_SCENE_frame_wormseye_mount_30
```

2. `640 + 580 = 1220`，正好等于当前桌面外边距宽度。因此新相机中心的横向坐标是：

```text
x = -610 + 640 = 30 mm
x =  610 - 580 = 30 mm
```

3. `260` 不是高度，不能写进 Z。相机的 Z 高度保持原 lower camera 高度：

```text
z = 17.429 mm
```

4. 原相机从负 Y 侧移动到正 Y 侧后，需要绕 Z 轴旋转 `180 deg`，使相机面向 `-Y`。

最终移动后的相机中心：

```text
[30.0, 323.856, 17.429] mm
```

## 已实现内容

新增/更新脚本：

```text
scene_reconstruction/cad/aloha_incremental/scripts/create_iter003_lower_camera_top_position.py
scene_reconstruction/cad/aloha_incremental/scripts/open_iter003_lower_camera_top_position.py
```

生成结果：

```text
scene_reconstruction/cad/aloha_incremental/iterations/iter_003_lower_camera_top_position/
```

主要文件：

- `iter_003_lower_camera_top_position.FCStd`
- `bbox_and_dimensions.json`
- `top_dimension_review.png`
- `freecad_operation_tutorial.md`
- `changes.md`

新增或修改的 CAD 对象：

```text
REF_SCENE_frame_wormseye_mount_30
MEASURED_TOP_STEEL_RAIL_DARK_REFERENCE
MEASURED_CAMERA_SUPPORT_PIPE_260MM_1
MEASURED_CAMERA_SUPPORT_PIPE_260MM_2
MEASURED_CAMERA_SUPPORT_PIPE_260MM_3
MEASURED_CAMERA_SUPPORT_PIPE_260MM_4
```

其中：

- `REF_SCENE_frame_wormseye_mount_30` 是实际被移动的 lower camera mount。
- 不再创建 `NEW_LOWER_CAMERA_POSITION_GREEN` 方块；相机位置由真实 mount 本身表示。
- 4 根 `MEASURED_CAMERA_SUPPORT_PIPE_260MM_*` 都是深灰色钢管，尺寸为 `20 x 260 x 20 mm`，沿 `+Y` 方向，并对齐到上方相机架高度 `Z=610 mm`。

## 验证结果

FreeCAD 重新生成 `iter_003_lower_camera_top_position.FCStd` 后验证：

```json
{
  "REF_SCENE_frame_wormseye_mount_30": {
    "center_mm": [30.0, 323.856, 17.429],
    "rotation_about_z_deg": 180.0
  },
  "support_pipe_count": 4,
  "support_pipe_size_mm": [20.0, 260.0, 20.0],
  "support_pipe_center_z_mm": 610.0,
  "green_marker_created": false
}
```

对应尺寸：

- 左外距：`640 mm`。
- 右外距：`580 mm`。
- 4 根安装钢管长度：`260 mm`，沿 `+Y`。
- 4 根安装钢管中心高度：`Z=610 mm`，对齐现有上方相机架。
- 相机方向：移动后面向 `-Y`。

## 证据与来源

### 本地模型证据

- `REF_TABLE_DESKTOP_PLANE` 的 X 外边界是 `[-610, 610] mm`，外宽 `1220 mm`。
- `REF_SCENE_frame_wormseye_mount_30` 在 `iter_002` 中的中心是 `[0.0, -323.856, 17.429] mm`。
- `REF_SCENE_frame_wormseye_mount_30` 在 `iter_003` 中的中心是 `[30.0, 323.856, 17.429] mm`。
- 4 根支撑钢管对齐当前模型中已有的四条纵向 frame X 位置：左外、左内、右内、右外。

### FreeCAD 操作依据

- FreeCAD 官方文档 GitHub 镜像 `Draft_Move.md`：
  - <https://github.com/FreeCAD/FreeCAD-documentation/blob/main/wiki/Draft_Move.md>
  - 该页说明 Draft Move 可以移动或复制对象，脚本接口是 `Draft.move(objectslist, vector, copy=False)`。
- FreeCAD 官方文档 GitHub 镜像 `Object_API.md`：
  - <https://github.com/FreeCAD/FreeCAD-documentation/blob/main/wiki/Object_API.md>
  - 该页说明 FreeCAD 文档对象具有 `Placement` 属性，用于表示对象的位置和姿态。

说明：FreeCAD wiki 页面当前可能被防护页拦截，因此采用官方 GitHub 文档镜像作为来源。

## 风险与未确认项

1. `20 mm` 钢架剖面来自当前模型中钢架对象的常见尺寸和现有 extrusion 的厚度推断。
2. 当前 FCStd 没有独立 scene-level D405 相机实体，所以移动的是 `wormseye_mount` 这个真实安装件 mesh。
3. 4 根 `260 mm` 钢管的位置按现有 CAD frame 的四条纵向 X 位置对齐，高度按现有上方相机架 `Z=610 mm` 对齐；如果真实钢架的孔位或高度不同，需要后续按实测参数调整。

## 如何继续微调

如果后续测得钢架剖面不是 `20 mm`，或 4 根支撑管 X 位置需要调整，修改脚本中的：

```text
STEEL_PROFILE_MM
SUPPORT_PIPE_REFERENCE_OBJECTS
```

然后重新运行：

```bash
/snap/bin/freecad.cmd -c "exec(open('/home/eii/project/openpi0.5-rtc-reward-learning/scene_reconstruction/cad/aloha_incremental/scripts/create_iter003_lower_camera_top_position.py').read())"
```
