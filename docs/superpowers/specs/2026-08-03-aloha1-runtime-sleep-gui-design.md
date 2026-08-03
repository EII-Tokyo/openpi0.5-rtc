# ALOHA1 Runtime-Sleep GUI Design

## Goal

Open the frozen ALOHA1 diagnostic Stage in a new Isaac Sim 5.1 GUI process,
initialize `follower_left` at the previously verified runtime-measured Sleep
pose, move only that new window to workspace 2, and leave the timeline paused
for user review. No real-robot transport or motion command is permitted.

## Frozen inputs

- Stage: `assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_z_up_meters_diagnostic.usda`
- Stage SHA-256: `327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9`
- Manifest: `reports/aloha1_mapping/aloha1_runtime_measured_sleep_command_manifest.json`
- Manifest SHA-256: `d48047eadc6a02664efb01cba3e0345b523bf64052791491bb237639f24dad3c`
- Finger-limit layer SHA-256: `2547e6fb374c213b5c6c54f200c7ced37605ab0e1a11735d0a32c0a231fd260f`

## Design

The launcher reuses the already validated Stage/session-layer and
articulation initialization helpers. It verifies every frozen hash before
opening the Stage, applies only anonymous diagnostic layers, initializes both
articulations, writes the runtime Sleep vector to the left arm, renders enough
physics frames to synchronize visible transforms, then pauses both `World`
and timeline. The process remains alive with `app.update()` only.

The GUI window is moved by X11 window ID to one-based workspace 2 without
changing the user's active workspace. A READY report records versions, PID,
window/workspace result, Stage composition, left-arm target/readback/error,
timeline state, and the explicit zero-real-command boundary.

## Failure behavior

Hash mismatch, wrong sequence kind, wrong initial pose label, missing prim,
wrong DOF order, failed workspace move, non-paused timeline, or excessive
Sleep readback error prevents READY status. Source/default/final USD files are
never saved or edited.

## Acceptance

- Isaac Sim `5.1.0.0`, Kit `107.3.3`, PhysX `107.3.26` read back.
- The exact frozen Stage is loaded and its hash remains unchanged.
- `follower_left` readback is within `0.02 rad` of runtime-measured Sleep.
- Timeline is paused after visual transforms are refreshed.
- The Isaac window is on workspace 2 while the active workspace stays intact.
- Report says `READY_FOR_USER_REVIEW`, `real_motion_commands=0`, and
  `source_or_final_asset_modified=false`.
