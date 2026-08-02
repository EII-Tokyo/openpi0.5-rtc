# Stable Full Isaac Left Physics Inspector Startup Design

## Goal

Start Isaac Sim Full on workspace 3 with the approved ALOHA diagnostic Stage,
use the Perspective viewport camera, and leave Physics Inspector ready for the
left follower articulation without the `Structural changes detected` disabled
state.

## Frozen Runtime Identities

- Full experience: `isaacsim.exp.full.kit`
- Stage:
  `/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_diagnostic.usda`
- Expected Stage SHA-256:
  `eb3d2b12bb0903589856607c9f05212bf5c22182f539a413587162f4b1027459`
- Left articulation root:
  `/World/follower_left/vx300s_left/root_joint`
- User-facing workspace: GNOME workspace index `2` (workspace 3)

## Root Cause Addressed

The previous startup used a fixed number of UI updates before enabling Physics
Inspector and switched the viewport camera only after Inspector initialization.
The Inspector later entered its native `DISABLED` state after receiving a Stage
structural-change event, cleared its inspected path, and displayed
`Structural changes detected. Inspector needs parsing the stage again to
re-enable authoring`.

The revised startup must gate Inspector initialization on Stage loading
stability rather than an arbitrary update count and must provide one bounded
native recovery if the Inspector enters `DISABLED`.

## Startup Sequence

1. Launch the Full experience through the reviewed Dock wrapper with a Kit
   `--exec` script.
2. Stop the main timeline.
3. Open the frozen Stage path.
4. Switch the active viewport to the Perspective camera immediately after the
   Stage-open request.
5. Poll the USD context loading status until no files remain loading, then
   require several consecutive stable UI updates.
6. Validate that the left articulation root exists and has
   `PhysicsArticulationRootAPI`.
7. Enable `omni.physx.supportui` Physics Inspector through its registered native
   action.
8. Select the exact left articulation root and invoke the Inspector toolbar's
   native `Use current stage selection` handler.
9. Confirm that the Inspector window is visible, the selected label is the exact
   root, and the model contains the expected left-arm joint rows.
10. Keep the timeline stopped and leave the window on workspace 3.

## Bounded Recovery

Monitor the Inspector model state during a short startup acceptance window.

- If it never enters `DISABLED`, perform no recovery.
- On the first `DISABLED` observation, call the same native method used by the
  `Re-Enable authoring` button, wait for reparsing, reselect the exact left root,
  and invoke `Use current stage selection` again.
- If `DISABLED` is observed again, stop retrying, emit an explicit failure marker,
  and leave the application open for diagnosis.
- Never use an unbounded retry loop.

This recovery is startup-scoped. It does not silently undo structural edits the
user intentionally makes later in the session.

## Safety and Mutation Boundaries

- Do not connect to or command the real ALOHA robot.
- Do not start the main timeline.
- Do not set joint positions, drive targets, velocities, or efforts.
- Do not save the Stage.
- Do not rotate `/World` or change the Stage up-axis.
- Do not modify the Dock launcher because it already resolves to the Full
  experience.
- Preserve all unrelated dirty worktree files.

## Verification

Acceptance requires fresh evidence for all of the following:

- process command contains `isaacsim.exp.full.kit` and the revised `--exec`
  script;
- log identifies Isaac Sim Full and `omni.physx.supportui` startup;
- Stage URL equals the frozen absolute path;
- Stage loading reaches stable zero-pending status before Inspector enablement;
- viewport camera action is Perspective;
- left articulation root is valid and has `PhysicsArticulationRootAPI`;
- Inspector is visible, selected label equals the exact left root, and expected
  joint rows are populated;
- Inspector remains out of `DISABLED` throughout the startup acceptance window,
  or succeeds after exactly one bounded recovery;
- timeline is stopped;
- window is on workspace index `2`;
- final Stage SHA-256 is unchanged;
- startup script contains no joint-control calls.

Capture a final screenshot showing the Perspective label and the populated
Physics Inspector panel.
