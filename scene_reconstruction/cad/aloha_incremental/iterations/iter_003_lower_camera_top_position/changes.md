# Iteration 003 Lower Camera Top Position

## What Changed

- Started from `scene_reconstruction/cad/aloha_incremental/iterations/iter_002_measured_aloha_y_offset/iter_002_measured_aloha_y_offset.FCStd`.
- Moved the original `REF_SCENE_frame_wormseye_mount_30` to the measured target; no green cube substitute is created.
- Rotated the moved camera/mount 180 deg about Z so it faces -Y.
- Added `MEASURED_TOP_STEEL_RAIL_DARK_REFERENCE` as a dark gray rail proxy.
- Added four `MEASURED_CAMERA_SUPPORT_PIPE_260MM_*` pipes, each 260 mm long along +Y and aligned to the upper camera-frame height.

## Measurement Interpretation

- `640 mm` and `580 mm` are outer horizontal distances.
- `260 mm` is a Y-direction support-pipe length, not a Z height.
- The lower-camera Z height is preserved from the original camera mesh.
