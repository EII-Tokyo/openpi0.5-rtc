# ALOHA1 Tabletop-Zero Root Metadata Fix Design

## Goal

Make the approved ALOHA diagnostic Stage compose as meter-scaled, Z-up USD so
Isaac Sim uses `+Z` as the world up direction without rotating any existing
geometry, articulation, joint, collider, or environment prim.

## Frozen Target

- Root layer:
  `assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_diagnostic.usda`
- Default prim: `/World`
- Existing sublayer order remains unchanged.
- Current pre-fix SHA-256:
  `eb3d2b12bb0903589856607c9f05212bf5c22182f539a413587162f4b1027459`

## Evidence and Root Cause

The root layer authors `defaultPrim` and two `subLayers` but does not author
`metersPerUnit` or `upAxis`. Opening the composed Stage with the same OpenUSD
runtime used by Isaac Sim returns:

```text
COMPOSED_UP_AXIS Y
METERS_PER_UNIT 0.01
```

The geometry is already authored as meter-scaled Z-up data. In particular, the
table thickness is on Z, both follower bounds use Z as height, and `/World` has
an identity transform. The mismatch is therefore Stage metadata at the root
layer, not a geometry transform problem.

NVIDIA's Isaac Sim convention is right-handed with world `+Z` up and `+X`
forward. The root layer must explicitly declare that convention.

## Considered Approaches

1. **Author root-layer metrics — selected.** Add `metersPerUnit = 1` and
   `upAxis = "Z"` next to `defaultPrim`. This fixes persistent Stage semantics
   at their source while preserving every prim transform.
2. **Rotate `/World` — rejected.** The geometry is already Z-up, so rotating the
   root would corrupt robot, table, collision, and articulation alignment.
3. **Use a session-layer/runtime override — rejected.** It would be transient,
   make future openings inconsistent, and leave the source asset defective.

## Exact Change

The root-layer header becomes:

```usda
#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    subLayers = [
        @../../table_support_alignment/1.0/configuration/aloha1_tabletop_world_zero.usda@,
        @aloha1_cad_derived_full_body_collider_gripper_decomposition_diagnostic.usda@
    ]
    upAxis = "Z"
)
```

No other line in the USD changes.

## Verification

1. A regression test must fail before the edit because the root header lacks
   the two declarations, then pass after the edit.
2. Open the composed Stage with the Isaac-bundled OpenUSD runtime and require
   `UsdGeom.GetStageUpAxis(stage) == "Z"` and
   `UsdGeom.GetStageMetersPerUnit(stage) == 1.0`.
3. Require `/World` to retain an identity local-to-world transform and verify
   the same two sublayer paths in the same order.
4. Record the intentional new root-layer SHA-256.
5. Stop only the verified currently running Isaac Sim Full process, relaunch
   Full with the existing reviewed left-Inspector startup script, and keep the
   main timeline stopped.
6. Verify the exact Stage URL, Perspective view, valid left articulation root,
   13 Inspector joint rows, non-`DISABLED` Inspector state, workspace index 2,
   and a viewport screenshot whose axis indicator treats Z as world up.

## Safety Boundaries

- No real robot connection or command.
- No joint value, target, velocity, force, or effort writes.
- No timeline playback.
- No Stage save from the Inspector session.
- No `/World` transform, geometry, physics, collider, or articulation edits.
- Preserve unrelated dirty worktree files.
