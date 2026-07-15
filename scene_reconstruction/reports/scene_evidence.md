# Scene Evidence

## Source Separation

| Source class | Files / evidence | Status meaning |
| --- | --- | --- |
| User measurements | `/home/eii/Downloads/iphone/IMG_5334.JPG`, user text in this session | `measured` when the user gave a numeric value |
| Real photos | `/home/eii/Downloads/iphone/IMG_5335.JPG` through `IMG_5340.JPG` | `estimated` unless a ruler or known dimension is visible |
| Original ALOHA USD | `local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose/aloha2_menagerie_scene_deep_black_real_start_pose.usd` | `read_from_usd` for robot prim transforms and extents |
| Existing overlay USD | `local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose/aloha2_menagerie_scene_deep_black_real_start_pose_with_user_table_pipe.usda` | `read_from_usd` for existing proxy table, pipe and camera prims |
| Derived geometry | `scene_reconstruction/config/scene_parameters.yaml` | `derived` when computed from measured/read_from_usd values |

## Confirmed Facts

- Isaac/USD stage units are meters, with Z as up axis.
- The original confirmed ALOHA USD contains the robot but no real `UsdGeom.Camera` prim.
- The existing overlay USD contains proxy cameras:
  - `/World/Cameras/cam_low`
  - `/World/Cameras/cam_right_wrist_hint`
- The table is measured as `1.10 m x 0.60 m`.
- The pipe diameter is measured as `0.005 m`.
- The pipe base is `0.095 m` outside the `w1` table edge.
- The pipe base projection point is `0.58 m` from the left table edge.
- The pipe length is recorded as `0.225 m`, and side tilt as `44 deg` from the user's sketch.

## Derived Pipe Axis

The table frame used for the first-pass model puts the table center at the world origin.
The left table edge is at `x = -0.55 m`, and the `w1` edge is at `y = 0.30 m`.

Therefore the pipe base projection point is:

```text
A = (-0.55 + 0.58, 0.30, 0.0) = (0.03, 0.30, 0.0)
```

The pipe axis starts outside the table edge:

```text
axis_start = (0.03, 0.30 + 0.095, 0.07) = (0.03, 0.395, 0.07)
```

Using length `0.225 m` and side tilt `44 deg`, the horizontal component is approximately `0.162 m`,
and the vertical component is approximately `0.156 m`. The current model points the pipe parallel
to the table edge toward the left arm:

```text
axis_end = (0.03 - 0.162, 0.395, 0.07 + 0.156) = (-0.132, 0.395, 0.226)
```

## Estimated Items

The following must not be treated as measured:

- camera rack outer width/depth/height;
- camera rack profile section;
- exact camera optical center;
- real camera intrinsics;
- real camera extrinsics;
- exact ALOHA base-to-table transform;
- exact rack attachment locations.

## Visual Reading Notes

- `IMG_5339.JPG` and `IMG_5340.JPG` are the most useful for visual alignment because they show both ALOHA arms, the front rack, the pipe assembly, and the table surface.
- `IMG_5335.JPG` and `IMG_5336.JPG` are useful for top/oblique rack layout.
- `IMG_5334.JPG` is useful for room lighting, but not for geometric calibration.

## Minimum Measurements Still Needed

- Rack outer width, depth and height.
- Rack extrusion cross section.
- Real `cam_low` optical center relative to the table frame.
- Real `cam_right_wrist` optical center relative to the wrist or table frame.
- Camera intrinsics or ROS `camera_info`.
- ALOHA base positions relative to the table frame.
- Pipe holder/base precise CAD dimensions.
