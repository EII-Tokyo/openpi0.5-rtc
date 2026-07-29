# ALOHA 500 mL bottle CAD source manifest

- Status: `PARTIAL`
- Selection: `GEOMETRY_REFERENCE_ONLY_NOT_DEFAULT_FOR_GRASP`
- Original: `/home/eii/Downloads/500mlbottle.step`
- Canonical local-only copy:
  `/home/eii/project/openpi0.5-rtc-reward-learning/local_eval_assets/aloha_bottle_cad/500mlbottle.step`
- SHA-256:
  `88a341eb493211b46ede5b1b5c448da06a9845d93b328613719521c242f36416`
- Size: `7,842,101 bytes`
- Raw STEP redistribution: `UNKNOWN_HARD_BLOCKER`
- Task 8: `NOT_RUN`

The source and canonical-copy hashes match. The canonical copy is below the
Git-ignored `local_eval_assets/` tree. The raw STEP is therefore available for
local audit but is not added to Git or asserted to be redistributable.

## Read-only CAD audit

The project-pinned FreeCAD executable read the source successfully:

- FreeCAD `1.1.1`, build `20260414 (Git shallow)`;
- OpenCascade `7.8.1`;
- Python `3.11.14`;
- AP214, millimetres, radians;
- one valid compound containing two valid solids;
- aggregate B-Rep SHA-256
  `84db52faedc8f42d8771e6e6b725522be511d4bdaad1475349fb82a17bc5d6d8`;
- `1,599` faces, `3,929` edges and `2,411` vertices;
- standard `Shape.BoundBox`:
  `65.43889555547 × 196.577258730698 × 64.445444477224 mm`;
- `Shape.optimalBoundingBox()`:
  `60.05492227750102 × 192.7344012859301 × 60.054922277501014 mm`;
- CAD long axis: `+Y`;
- aggregate CAD solid volume: `13,107.844274397974 mm³`.

The two solids are geometrically consistent with a cap-like part and a
body-like part, but the STEP contains generic translator labels. Those roles
remain engineering inferences, not source-authored semantic names.

For this B-Spline-heavy STEP, the ordinary `BoundBox` is a conservative
overbound. The optimal B-Rep box agrees with the independently tessellated
surface to within approximately `0.008 mm` on the radial extents and
`0.000005 mm` on the long-axis extent. The optimal box is therefore the
reported physical geometry extent; the ordinary box is retained only as audit
evidence.

The machine audit is
`/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-500ml-bottle-source/freecad_audit.json`.
The bounded clean-run log is
`/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-132355_20260729-aloha-500ml-bottle-optimal-bbox-audit-v8`.

## Selection and evidence boundary

This STEP is retained as a detailed geometry reference. It is **not** the
default input for future ALOHA digital bottle-grasp tests. The project-authored
`/assets/bottle_500ml` CAD is the selected primary bottle. This downloaded
STEP also does **not** define an accepted Isaac rigid body:

- `500 mL` is the user's designation and filename, not a volume measurement
  derived from the two CAD solids;
- mass, fill state, material and wall properties are not confirmed;
- CAD `+Y` was rotated `+90°` about X to display `+Z` for diagnostic
  screenshots only;
- deterministic visual tessellation passes, but collision design and Isaac
  runtime validation are `NOT_RUN`;
- the CAD solid volume must not be interpreted as liquid capacity or mass.

The project `/assets/bottle_500ml` CAD is now the primary geometry for future
grasp tests. Its `25 g` FCStd parameter remains uncalibrated and does not
override the existing `20 g` Task 5 diagnostic baseline without an explicit
test-profile decision.

## HARD_BLOCKER

- `BOTTLE_STEP_FORMAL_LICENSE_TEXT_MISSING`: blocks committing or
  redistributing the raw STEP.
- `BOTTLE_MASS_FILL_STATE_NOT_CONFIRMED`: blocks a calibrated static-hold or
  inertia claim.
- `BOTTLE_MATERIAL_AND_WALL_PROPERTIES_NOT_CONFIRMED`: blocks calibrated
  contact/compliance claims.
- `DOWNLOADED_REFERENCE_CAD_NOT_SELECTED_FOR_ISAAC_PROMOTION`: the reference
  STEP is not promoted.
- `PRIMARY_BOTTLE_COLLIDER_NOT_REVALIDATED_WITH_CURRENT_GRIPPER`: blocks
  canonical grasp-regression promotion.
