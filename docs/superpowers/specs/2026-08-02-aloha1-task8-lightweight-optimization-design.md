# ALOHA1 Task 8 Lightweight Optimization Design

## Authorization boundary

On 2026-08-02 the user explicitly authorized Task 8 to begin before the
remaining Task 7 asset-promotion findings are closed. This authorization does
not turn Task 7 into `PASS` and does not authorize promotion of a diagnostic
layer into a final/default asset.

The status contract is:

- Task 7 runtime grasp and finger-safety gates: `PASS`;
- Task 7 asset-promotion readiness: `PARTIAL`;
- Task 7 aggregate: `PARTIAL_ACCEPTED_FOR_TASK8`;
- Task 8: `AUTHORIZED_IN_PROGRESS`.

Task 8 may expose additional Task 7 defects. Such defects are recorded and
returned to the matching Task 7 root-cause scope instead of being hidden by an
optimization.

## Objective

Measure the frozen ALOHA1 diagnostic baseline, create an isolated low-risk
optimization candidate, and quantify whether visual/composition optimization
improves load/runtime cost without changing robot structure, collision,
physics, control mapping, or the accepted grasp trajectory.

## Frozen inputs

- baseline Stage:
  `assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_z_up_meters_diagnostic.usda`;
- baseline SHA-256:
  `327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9`;
- diagnostic finger-limit layer:
  `assets/Trossen/ALOHA1/1.0/diagnostics/finger_limit_pair_collision_candidate/1.0/configuration/finger_source_limits.usda`;
- finger-limit SHA-256:
  `2547e6fb374c213b5c6c54f200c7ced37605ab0e1a11735d0a32c0a231fd260f`.

All candidates live below a new isolated Task 8 diagnostic directory. The
frozen Stage, source layers, final/default assets and colliders are immutable.

## Optimization order

1. Inventory prims, meshes, points, faces, materials, references, payloads,
   instanceable prims, physics schemas and composed dependencies.
2. Measure stage-load time, memory and fixed-frame update/physics timings using
   the local Isaac Sim 5.1 benchmark implementation where practical.
3. Select only opportunities demonstrated by the inventory.
4. Prefer visual/composition-only changes. Do not begin with collider
   simplification, drive changes, timestep changes or solver changes.
5. Keep visual geometry and collision geometry separable and prove the
   candidate's physics/control signature is unchanged.

Instanceable authoring is not an automatic optimization: this project already
has verified local `omni.hydra.usdrt_delegate 7.5.1` prototype-resolution
failures. It is attempted only if an isolated candidate passes native render,
determinism and composition checks. Mesh merge is also conditional because it
can erase link/material granularity.

## Lightweight acceptance

Task 8 does not rerun the five accepted grasp videos by default. A candidate
must pass:

- frozen input hashes and dependency resolution;
- articulation count and roots;
- exact DOF names and order;
- drive/mimic and source finger-limit readback;
- unchanged collision and physics-schema signature;
- one gripper open/close smoke;
- one short horizontal Bottle500 grasp/lift/hold smoke when the composed Stage
  is changed;
- baseline/candidate performance comparison with method and variance recorded.

No candidate is promoted automatically.

## Failure-first visual evidence

The user explicitly prioritizes evidence of errors over a forced green report.
If a Task 8 run fails or exposes a regression, preserve:

- a raw and annotated image before the anomaly;
- a raw and annotated image at the first anomalous frame;
- a raw and annotated final failure image;
- a video showing the full arm, gripper, affected object and collision display;
- stage path/hash, camera pose, frame/time, target/readback, contact/collision
  telemetry and failure classification.

Annotations must show the faulty part, expected relationship, observed
relationship, relevant direction/offset and why the frame is a failure. Each
image and video is visually reviewed before acceptance. Every reproducible
Task 8 failure requires both the three screenshot phases and a full-arm,
collision-enabled video, including a render-only failure. Passing visual-only
comparisons need screenshots but do not require a new video.

## Result states

- `PASS_CANDIDATE_NOT_PROMOTED`: measurable improvement and all lightweight
  gates pass;
- `NO_MEASURABLE_IMPROVEMENT`: correct candidate but no meaningful gain;
- `REGRESSION_CAPTURED`: a regression is reproducible and has complete visual
  and machine evidence;
- `PARTIAL`: work is reproducible but an applicable gate or evidence item is
  incomplete;
- `HARD_BLOCKER`: promotion or a physical fact needs user authority/data.
