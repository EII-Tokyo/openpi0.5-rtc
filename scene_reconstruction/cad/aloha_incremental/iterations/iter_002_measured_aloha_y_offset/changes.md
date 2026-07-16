# Iteration 002 Measured ALOHA Y Offset

## What Changed

- Started from `scene_reconstruction/cad/aloha_incremental/iterations/iter_001_measured_table_rack/iter_001_measured_table_rack.FCStd`.
- Translated all `REF_ALOHA_*` robot meshes by `dy=46.500 mm`.
- Kept the measured table/rack footprint from Iteration 001.

## Measurement Used

- Table width: `625 mm`.
- Physical top margin to robot base: `180 mm`.
- Physical bottom margin to robot base: `235 mm`.

## Important Note

The CAD robot base mesh is `204 mm` wide in Y, while the two measured margins imply a `210 mm` base footprint. This iteration does not scale the robot; it aligns the base center, giving approximately `183 mm` and `238 mm` margins.
