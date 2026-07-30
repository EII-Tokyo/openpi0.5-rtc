# ALOHA1 Full-Gripper CAD Clearance Design

## Scope

Correct the follower-left horizontal Bottle500 grasp frame before any new
Grasp Editor, IK, randomized grasp, or video run. The correction is diagnostic
and session-only. It must not modify the supplier STEP, project Bottle500
FCStd/STEP, imported source USD, accepted Stage, physics/collider layers, or
Task 8 state.

## Root Cause

The rejected run13 frame used the centroid of each finger's entire largest
inward planar B-Rep face. That face spans most of the finger length, so its
centroid is not evidence of a usable distal contact station for a 68 mm
Bottle500 body. The runtime result placed the bottle in the gripper-bar
envelope, produced no valid bilateral finger contact, and was nevertheless
reported as successful by the native tester because its active finger did not
reach the configured fully closed value.

The supplier CAD was therefore used only for the handed finger pair, not for
the complete gripper-envelope clearance problem. The replacement method must
include the supplier gripper shell/sliding carriage and retain the runtime
URDF gripper-bar envelope as a separate, potentially conflicting source.

## Frozen Sources

- Supplier assembly:
  `.codex/artifacts/20260729-aloha-finger-palm-orientation/gdrive_source_readonly/Simple Aloha Viper 2024-5-13.step`
  with SHA-256
  `337862418769d4ea8b801d26e68930c4412f870050e60769bbf91765194dc571`.
- Supplier gripper shell/sliding carriage:
  `Part__Feature006`, label `Aloha VX Gripper 2024-4-19 v4`.
- Supplier left/right installed fingers:
  `Part__Feature007` and `Part__Feature008`, the embedded handed v2 pair.
- Project primary bottle:
  `assets/bottle_500ml/cad/bottle_500ml.FCStd`, SHA-256
  `3594f60200e54181bc8480a229484293a0d386c146d3f235b32e31a0c16bbf8a`.
- Exported bottle STEP:
  `assets/bottle_500ml/cad/bottle_500ml.step`, SHA-256
  `863001b4d939d7d8c879497b5054fe93f426662761e6fb7a80550096fd9bc780`.
- Generated follower-left URDF remains the kinematic source for link/joint
  names, axes, origins, and legal finger ranges.

## Geometry Contract

The supplier assembly is evaluated in its native CAD coordinates:

- closing line: CAD `X`;
- approach direction: CAD `-Y`, corresponding to gripper-link `+X`;
- bottle axis: CAD `Z`;
- no mirror or per-finger rotation is allowed.

The bottle remains horizontal in the task frame. Its CAD `+Z` longitudinal
axis is aligned with gripper CAD `Z`. The selected axial station remains the
current evidence-backed cylindrical-body station until episode 18 or a user
confirmation supersedes it.

The usable grasp station along the fingers is not a face centroid and is not
the first point that merely avoids intersection. It is the Chebyshev center
of the continuous, symmetric feasible interval on both audited inward pad
faces. This max-min rule maximizes the smaller of the distance to the
gripper/bar forbidden boundary and the distance to the distal pad boundary,
without introducing a guessed millimetre offset. Every point in the interval
must satisfy all exact B-Rep clearance gates:

1. the Bottle500 B-Rep at the selected axial station does not intersect the
   supplier gripper shell/sliding carriage;
2. it does not intersect the runtime URDF gripper-bar envelope;
3. the proposed left/right contact points remain on both audited inward faces;
4. both points have opposing inward normals;
5. the bottle center lies on the midpoint of the contact pair;
6. a recorded nonnegative clearance margin exists to every forbidden
   envelope.

If the supplier shell and runtime bar disagree, preserve both results and use
the more restrictive valid interval for this diagnostic. Do not silently
change either geometry. The old centroid is explicitly retained as rejected
runtime evidence even if its static hard clearance is slightly positive:
run13 proved that its small margin entered the runtime contact envelope after
joint drift.

## Grasp Frame

The corrected frame `G_pad` is defined from exact geometry:

- origin: midpoint of the two selected effective distal pad contact points;
- `+Y`: left-to-right finger contact line using the existing URDF convention;
- `+X`: gripper approach direction;
- `+Z`: right-handed completion, aligned with the directed bottle axis;
- determinant: `+1`;
- translation and rotation are recorded relative to `gripper_link`,
  `ee_gripper_link`, and the supplier CAD assembly.

The existing helper frame is retained and labelled `NOT_GRASP_CENTER`. The
new frame is session-only and cannot be promoted by this task.

## Static Evidence Gate

Before another Isaac run, produce a machine-readable report containing:

- all source paths and SHA-256 values;
- FreeCAD 1.1.1 / OCCT 7.8.1 readback;
- the complete candidate interval along the inner pad faces;
- exact B-Rep intersection/common volumes and minimum distances;
- supplier-shell and runtime-bar results separately;
- selected contact points, normals, midpoint, axes, transform matrices,
  determinant, and margins;
- a deterministic signature from two fresh FreeCAD processes;
- explicit rejection of the old whole-face centroid.

Produce true orthographic top and side raw/annotated images. Each must show
the complete gripper assembly, both fingers, the project Bottle500 envelope,
the helper EE point, the rejected centroid, the corrected frame, approach
axis, closing line, bottle axis, and clearance dimensions. Images must be
individually reviewed by the vision model.

Only a static geometry `PASS` and screenshot-review `PASS` authorize a new
Grasp Editor run. Grasp Editor, IK, five randomized positions, videos, and
Task 8 remain `NOT_RUN` until then.
