# Iteration 001 Measured Table/Rack

## What Changed

- Created a separate FreeCAD review model from `iter_000_reference.FCStd`.
- Applied the user's measured tabletop footprint: `1220 mm x 625 mm`.
- Scaled table/rack/camera-support scene meshes about the tabletop center: X scale `1.008264`, Y scale `0.822368`, Z scale `1.0`.
- Set `REF_TABLE_DESKTOP_PLANE` directly to `Length=1220 mm`, `Width=625 mm`, `Height=18 mm`.

## What Did Not Change

- The original Isaac/MJCF/USD assets were not modified.
- `iter_000_reference.FCStd` was not overwritten.
- `REF_ALOHA_*` robot meshes were not scaled.
- World axes were not scaled.

## Assumption

The measured tabletop center remains at the existing world origin. The rack and scene camera support structure follows the tabletop X/Y footprint, while the robot base geometry remains unchanged.
