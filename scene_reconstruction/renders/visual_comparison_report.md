# Visual Comparison Assets

These images are a first-pass proxy comparison, not a calibrated photogrammetry result.

| View | Real photo | Simulated/proxy | Overlay | Edges | Notes |
| --- | --- | --- | --- | --- | --- |
| `front_oblique` | `scene_reconstruction/renders/front_oblique/real.jpg` | `scene_reconstruction/renders/front_oblique/simulated.png` | `scene_reconstruction/renders/front_oblique/overlay.png` | `scene_reconstruction/renders/front_oblique/edges.png` | Proxy CAD layout only; camera pose is not photogrammetrically solved. |
| `rack_oblique` | `scene_reconstruction/renders/rack_oblique/real.jpg` | `scene_reconstruction/renders/rack_oblique/simulated.png` | `scene_reconstruction/renders/rack_oblique/overlay.png` | `scene_reconstruction/renders/rack_oblique/edges.png` | Proxy CAD layout only; camera pose is not photogrammetrically solved. |

## Interpretation

- Aligned evidence: table plane, black camera rack concept, pipe assembly, and camera proxy locations are represented.
- Known deviations: rack dimensions and exact camera brackets are estimated; the CAD view is not rendered from an optimized real camera pose.
- Next measurement needed: rack width/depth/height, camera optical center, camera pitch/yaw, and calibrated camera intrinsics.
