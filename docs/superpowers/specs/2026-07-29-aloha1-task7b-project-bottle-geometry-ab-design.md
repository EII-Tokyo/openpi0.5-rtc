# ALOHA1 Task 7B Project-Bottle Geometry A/B Design

## Objective

Determine whether replacing the already passing 65 mm procedural cylinder
with the project-authored Bottle500 geometry changes the supplier-CAD
follower gripper's ability to establish bilateral contact and statically hold
a 20 g digital bottle.

This is a controlled geometry experiment. It is not a friction calibration,
mass calibration, SimReady promotion, support-to-lift acceptance, insertion
task, or Task 8 optimization.

## Frozen inputs

### Robot and gripper

- Existing supplier-CAD follower-left Task 5 parent diagnostic:
  `assets/Trossen/ALOHA1/1.0/diagnostics/cad_finger_task5_arm_max_force_over_combined/aloha_viperx_supplier_cad_arm_max_force_over_combined.usda`
- Supplier assembly embedded v2 handed finger pair.
- Existing supplier-CAD convex-hull diagnostic collider.
- Explicit symmetric finger targets.
- Existing arm and finger drive configuration.

### A: procedural-cylinder baseline

- Shape: `UsdGeom.Cylinder`
- Diameter: `0.065 m`
- Height: `0.210 m`
- Mass: `0.020 kg`

### B: project-authored Bottle500

- CAD master:
  `assets/bottle_500ml/cad/bottle_500ml.FCStd`
- Isaac asset:
  `assets/bottle_500ml/isaac/bottle_500ml_sim.usd`
- Referenced product prim: `/Bottle500`
- Runtime-read root-layer default prim: `/World`. The root layer also contains
  a test gauge, so Task 7B must explicitly reference `/Bottle500`; it must not
  use the root layer default prim.
- Geometry: project-authored 68 mm maximum-diameter, 206 mm-high bottle.
- Collision: the existing 41-piece collision hierarchy is used unchanged.
- Diagnostic mass override: `0.020 kg`, authored only in the isolated
  Task 7B session/diagnostic layer. The asset's `0.025 kg` value remains
  `TEMPORARY_REQUIRES_MEASUREMENT` and is not modified.

The CAD bottle replaces only the bottle geometry/collider provider. It must
not alter the robot, finger geometry, drive, material coefficients, solver,
time step, trajectory, acceptance gate, or initial grasp-frame derivation.

## Shared physics and control parameters

- Static/dynamic friction: `0.7`
- Friction status: `TEMPORARY_UNCALIBRATED`
- Restitution: `0`
- Physics frequency: `60 Hz`
- `solve_articulation_contact_last = True`
- Self collision: disabled
- Finger maximum force: `5 N` left and right
- Open targets: `[0.057, -0.057] m`
- Closed targets: `[0.021, -0.021] m`
- Hold interval: `2 s` / `120` physics steps
- Maximum drop: `0.010 m`
- No SurfaceGripper
- No fixed joint
- No parent attachment
- Fixed/kinematic bottle is allowed only during bilateral-contact setup and
  cannot count as static-hold PASS.

## Architecture

### Pure experiment contract

Add a small pure-Python module that:

- defines the two geometry profiles;
- verifies that every non-geometry causal parameter matches;
- rejects simultaneous parameter changes;
- classifies each trial using the existing Task 5 gates;
- compares group results without relaxing the gate;
- outputs one of:
  `PROJECT_BOTTLE_MATCHES_BASELINE`,
  `PROJECT_BOTTLE_WORSENS_HOLD`,
  `PROJECT_BOTTLE_IMPROVES_HOLD`,
  or `INCONCLUSIVE`.

### Isaac runtime

Extend the existing supplier-CAD bottle validator instead of duplicating its
contact and hold logic.

- Profile A retains the existing procedural cylinder.
- Profile B creates an Xform at the same bottle session path and adds a USD
  reference explicitly to `/Bottle500`. The source layer default prim is
  `/World` and is intentionally not referenced.
- Profile B reads back the composed visual/collision hierarchy, rigid-body
  API, collision APIs, material bindings, AABB, and effective mass.
- A session-only/local diagnostic opinion overrides mass to `0.020 kg` and
  binds the same temporary bottle physics material used by A.
- The approved source Stage, project Bottle500 USD, source URDFs,
  configuration layers, and final/default colliders remain immutable.

### Isolated diagnostic asset

Create a wrapper only if the reference must be persisted for reproducibility:

`assets/Trossen/ALOHA1/1.0/diagnostics/task7b_project_bottle_geometry_ab/`

The wrapper may reference existing assets but may not copy, flatten, edit, or
promote the Bottle500 or robot source assets.

## Experiment sequence

For each profile:

1. Start a fresh Isaac Sim 5.1 process/world reset.
2. Load the same frozen parent diagnostic.
3. Open the fingers.
4. Derive the bottle pose from the same finger midpoint and closing-axis
   calculation.
5. Keep the bottle kinematic only while closing to establish bilateral
   contact.
6. Verify physical left and right finger contact, normal direction, finite
   impulse, penetration, and absence of unexpected gripper collisions.
7. Remove kinematic state without changing bottle pose.
8. Hold for 2 seconds under gravity.
9. Record contact continuity, bottle pose/velocity, angular velocity, drop,
   penetration, target/readback, and deterministic signature.
10. Reset completely before the next trial.

Run one smoke trial per profile, then 20 fresh-reset acceptance trials per
profile. Do not reuse the historical cylinder result as the new A group.

## Acceptance

Each trial uses the existing gates:

- both fingers establish physical contact before release;
- impulses and state values are finite;
- no persistent excessive penetration;
- no unexpected gripper-bar/internal collision;
- no fixed constraint, SurfaceGripper, or parent attachment;
- gravity is enabled after release;
- contact/pose/velocity records cover the full 2-second interval;
- maximum bottle drop is at most `0.010 m`.

Group PASS requires 20/20 passing trials and a single deterministic signature.
The project bottle is not accepted merely because its result is better than
the cylinder; it must pass the unchanged gate.

## Lift boundary

This A/B proves static free-bottle holding, not support-to-lift pickup.

After the geometry A/B:

- if B fails static hold, classify the failure before attempting lift;
- if B passes, a later Task 7B.2 may apply the already validated small-up arm
  signal;
- a claim that the bottle was “picked up” additionally requires evidence that
  it began on the user-confirmed support surface and subsequently left it;
- if no validated grasp pose places the project bottle on that surface, record
  `HARD_BLOCKER_SUPPORT_TO_GRASP_POSE` and do not describe a floating-bottle
  hold as pickup.

## Screenshot evidence

For each profile capture:

- open;
- bilateral contact while kinematic;
- release;
- hold end.

Each capture has raw and annotated versions. Annotations identify both
fingers, inner contact surfaces, bottle, contact points/normals, frame/time,
bottle Z, drop, contact state, profile, and PASS/FAIL. The same phase/profile
camera pose must be fixed where comparisons require it.

Every image is individually reviewed with the vision model. Occluded,
cropped, distant, mislabeled, or visually indistinguishable captures are
rejected and retaken. Screenshots remain auxiliary to numeric evidence.

## Reports

Generate:

- `configs/aloha1_task7b_bottle_geometry_ab.yaml`
- `reports/aloha1_mapping/aloha1_task7b_bottle_geometry_ab.json`
- `reports/aloha1_mapping/aloha1_task7b_bottle_geometry_ab.md`
- `reports/aloha1_mapping/aloha1_task7b_bottle_geometry_ab_trials.jsonl`
- `reports/aloha1_mapping/aloha1_task7b_bottle_geometry_ab_screenshot_review.json`
- `reports/aloha1_mapping/aloha1_task7b_bottle_geometry_ab_screenshot_review.md`

Full Isaac, pytest, annotation, and screenshot logs are stored under:

`.codex/artifacts/20260729-aloha1-task7b-project-bottle-geometry-ab/`

## Status boundaries

- Task 7A remains unchanged.
- Task 7B becomes PASS only if the explicitly scoped static-hold gate passes;
  otherwise it is FAIL or PARTIAL according to machine evidence.
- Asset promotion remains PARTIAL.
- Task 8 remains `NOT_RUN`.
- No real robot is connected and `192.168.1.103` is not accessed.
