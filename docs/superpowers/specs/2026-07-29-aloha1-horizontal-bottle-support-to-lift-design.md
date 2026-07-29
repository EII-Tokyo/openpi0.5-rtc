# ALOHA1 Horizontal Bottle Support-to-Lift Design

## Status

User-approved design, 2026-07-29.

This design replaces the upright-bottle and shoulder-sweep geometry in
`2026-07-29-aloha1-task7b2-support-to-lift-pickup-design.md` as the default
Task 7B.2 pickup task. The older design and its runtime evidence remain
historical records, but they do not contribute to acceptance of the
horizontal-bottle task.

Task 8 remains `NOT_RUN`.

## Goal

Use Isaac Sim 5.1 runtime evidence to determine whether the current
supplier-CAD follower gripper can:

1. approach the project Bottle500 vertically while it lies horizontally on
   the table under gravity;
2. establish physical bilateral contact on the bottle body;
3. lift the bottle away from the table; and
4. hold it for two seconds without a fabricated attachment.

This is an isolated digital diagnostic. It is not calibrated sim-to-real
dynamics, bottle-mouth insertion, a final collider promotion, or Task 8.

The runtime is strictly limited to:

- Isaac Sim `5.1.0.0`;
- Kit `107.3.3`; and
- PhysX `107.3.26`.

Before implementing or changing Isaac code, USD authoring, Stage contents, or
runtime physics behavior, use NVIDIA's official Isaac capability through the
project's MCPJungle Gateway to verify the local 5.1 API. Do not substitute a
latest or 6.0 API.

## Selected Approach

Use a hybrid evidence pipeline:

- CAD defines the bottle axis, body region, dimensions, and geometric grasp
  section.
- The user-confirmed episode 18 frame window defines the real-data action
  phases and checks that the task semantics match a horizontal bottle grasp.
- Version-verified Isaac Sim 5.1 FK/IK generates the digital Cartesian
  approach and lift trajectory.

This avoids two unsupported shortcuts:

- literal replay of the full 14-dimensional real-robot signal without
  calibrated table/base/camera transforms; and
- a CAD-only arm motion chosen without real-data phase correspondence.

## Approved And Frozen Inputs

### Project Bottle500

- CAD:
  `/home/eii/project/openpi0.5-rtc-reward-learning/assets/bottle_500ml/cad/bottle_500ml.FCStd`
- CAD SHA-256:
  `3594f60200e54181bc8480a229484293a0d386c146d3f235b32e31a0c16bbf8a`
- Isaac USD:
  `/home/eii/project/openpi0.5-rtc-reward-learning/assets/bottle_500ml/isaac/bottle_500ml_sim.usd`
- USD SHA-256:
  `16427135f152ec951de2321fd689366d745a2dd389cbe260976631783952533e`
- The CAD body is a surface of revolution around local `+Z`.
- The bottle length along the authored axis is `206 mm`.
- The approximately constant-radius body interval is
  `s = 18 mm` through `s = 120 mm`.
- The default grasp section is the midpoint of that CAD-derived interval:
  `s_grasp = 69 mm`.

The current `0.020 kg` diagnostic mass and the current friction inputs remain
`TEMPORARY_UNCALIBRATED`. They must not be described as measured physical
bottle parameters.

### Real-Data Window

- HDF5:
  `/home/eii/project/bottles_data/episode_18.hdf5`
- SHA-256:
  `f073a21c6a790e738e36085d791482924a82832ca6d80cece04a26353b9fc745`
- User-confirmed frames:
  `208-244` inclusive.

The window supplies phase, direction, and body-region evidence. It does not
supply an absolute world pose because the required camera extrinsics and
table-to-base calibration are not complete.

Frame interpretation remains indexed by frame number until acquisition-rate
provenance is verified. Action commands and joint-position readback are
separate evidence channels and must not be treated as equivalent.

### Isaac Diagnostic Stage

The candidate approved Stage is:

`/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/1.0/aloha1_signal_correspondence_workcell.usda`

Before any runtime mutation, the execution must recompute and freeze:

- the absolute Stage path and SHA-256;
- default/root prim;
- sublayers and references;
- follower articulation;
- supplier-CAD finger prims; and
- the user-confirmed table prim.

An older recorded hash is evidence, not permission to load a changed Stage.
All bottle and runtime diagnostic authoring remains in isolated session or
diagnostic layers. The source Stage is immutable.

## Bottle Coordinate Contract

Define:

- `A` as the bottle-bottom axis point at CAD axial coordinate `0 mm`;
- `B` as the bottle-mouth axis point at CAD axial coordinate `206 mm`;
- `AB = B - A`; and
- `ab_hat = normalize(AB)`.

After applying the runtime world transform, record:

- world-space `A` and `B`;
- `ab_hat`;
- the angle between `ab_hat` and world/table normal `+Z`;
- the bottle's world-space lowest point; and
- the gap between that lowest point and the table top.

For the default task, `AB` must be horizontal. The numerical diagnostic gate
is an angle of `90 degrees +/- 1 degree` relative to `+Z`. This is a
simulation acceptance tolerance, not physical measurement accuracy.

## Canonical Direction And Roll

Episode 18 cannot establish the sign of the absolute world-space directed
axis without calibrated camera extrinsics. The isolated diagnostic therefore
uses a reproducible convention:

1. choose the horizontal orientation that satisfies the gripper-line
   perpendicularity constraint;
2. select the directed solution for which
   `dot(ab_hat, world +X) >= 0`;
3. when that dot product is numerically zero, use
   `dot(ab_hat, world +Y) >= 0` as the tie breaker.

This direction is labelled
`DIAGNOSTIC_CANONICAL_NOT_REAL_CALIBRATION`.

Initial roll around `AB` uses the CAD-authored zero roll. Before dynamic
pickup, audit the radial symmetry of the collision representation. If its
discretization creates a material asymmetry, do not search arbitrary rolls.
Record the issue and, if needed, run a later one-variable roll sensitivity
diagnostic.

## Gripper Coordinate Contract

The gripper line joins the centers of the effective inward contact regions of
the supplier-CAD left and right fingers.

At the grasp waypoint:

- both inward surfaces face the bottle;
- the gripper-line projection onto table `XY` is perpendicular to `AB`;
- the diagnostic angular gate is `90 degrees +/- 3 degrees`;
- the midpoint of the two contact-region centers is aligned with the
  CAD-derived `s_grasp = 69 mm` section; and
- accepted finger/bottle contact points project into the CAD body interval
  `18-120 mm`.

The angular gate is a numerical path-acceptance tolerance, not a real
calibration measurement.

Do not mirror fingers, exchange handedness, add arbitrary 180-degree
rotations, or substitute the standalone 3D-A1 v3 or legacy finger geometry.

## Episode 18 Phase Extraction

The current evidence supports the following provisional frame-indexed
segments:

- frames `208-224`: open/approach candidate;
- frames `225-232`: closing-command transition;
- frame `229` onward: observed gripper-position response; and
- frames `237-244`: lift-onset candidate interval.

The implementation must refine these boundaries with:

- action transitions;
- qpos readback;
- version-pinned FK of the relevant end-effector/contact frame; and
- the extracted episode images.

It must not apply a qpos threshold to action data, or use an action threshold
as proof of physical finger motion.

Episode 18 provides task correspondence. It must not be converted directly
into final world joint targets without calibrated coordinate transforms.

## Cartesian Runtime Sequence

Each accepted trial starts from a fresh Isaac process or equivalent fresh
world reset:

1. Verify and freeze the approved Stage.
2. Compose Bottle500 only through an isolated diagnostic/session layer.
3. Establish a CAD-derived horizontal setup pose above the table.
4. Make the bottle dynamic and enable gravity before `support_settle`.
5. Require physical table support and finite, low linear/angular velocity.
6. Compute the world-space point on `AB` at `s_grasp = 69 mm`.
7. Compute an open-gripper pregrasp pose vertically above that point.
8. Solve a waypoint sequence that preserves gripper orientation and moves
   primarily along world `-Z`.
9. Descend with the gripper open.
10. Establish left and right physical bottle contact.
11. Apply the existing close/preload signal without changing collider,
    friction, drive, mimic, mass, timestep, or solver iterations.
12. Preserve the closed targets and solve a world `+Z` lift.
13. Require the bottle to leave table support.
14. Hold the lifted state for two seconds.

The pregrasp/descend distance must be derived from the composed bottle,
finger, table, and contact-envelope geometry. The world `+Z` lift displacement
must be derived from the validated episode 18 FK lift-onset interval and then
checked against the table-clearance gate. Neither distance may be selected by
visual trial and error. If the required FK or coordinate evidence cannot
determine one, report a `HARD_BLOCKER` and continue independent checks.

The descent-direction gate is derived from the Cartesian waypoint deltas. It
must demonstrate world `-Z` motion rather than a lateral sweep.

IK must:

- use only locally verified Isaac Sim 5.1 APIs;
- seed each waypoint from the previous accepted solution;
- enforce joint limits;
- reject discontinuities;
- verify the runtime target/readback result; and
- fail explicitly when the constrained path is unreachable.

Do not move the bottle or alter the task to accommodate an IK failure.

## Physics And Mutation Boundary

The first controlled comparison retains the current frozen diagnostic
parameters:

- supplier-CAD handed finger geometry;
- collider profile;
- friction `0.7`;
- restitution `0`;
- drive stiffness and damping;
- maximum force;
- mimic/explicit-control disposition;
- bottle mass and diameter;
- physics frequency `60 Hz`;
- solver iterations; and
- self-collision setting.

Do not change multiple parameters to obtain a passing report. If the
horizontal task exposes a new failure, preserve evidence and change at most
one diagnostic variable in a separately identified experiment.

Forbidden grasp fabrication includes:

- `SurfaceGripper`;
- fixed joints;
- parent attachment;
- runtime bottle teleport after the dynamic phase starts;
- disabling bottle gravity during settle/grasp/lift/hold; and
- abnormally high friction.

Finger/table contact is classified from body part, impulse, penetration,
duration, and task effect. It is not an unconditional failure.

## Validation Order

1. Stage and input manifest freeze.
2. Static CAD/USD geometry and collision audit.
3. Episode 18 phase extraction and FK report.
4. Horizontal bottle placement and support-settle audit.
5. Gripper-line/`AB` correspondence and IK feasibility report.
6. No-bottle open/close and waypoint structure check.
7. Non-acceptance kinematic path preview.
8. One fresh-reset dynamic smoke trial.
9. Repeated fresh-reset trials only after the smoke trial passes all physical
   gates.
10. True top and side screenshots with visual-model review.
11. Applicable Task 7 validators and regression tests.

A kinematic setup or path preview can provide geometry evidence but cannot
contribute to dynamic pickup `PASS`.

## Machine Acceptance

A dynamic trial is `PASS` only when all of the following are true:

- `A`, `B`, `ab_hat`, and the horizontal-angle gate are finite and pass.
- The bottle dynamically settles on the user-confirmed table.
- The gripper line is perpendicular to `AB` within the diagnostic gate.
- The pregrasp and descent trajectory is primarily world `-Z`.
- Both supplier-CAD fingers establish physical bottle contact before lift.
- Accepted contact points lie in the CAD-derived body interval.
- Contact normals and impulses are finite and physically directed.
- The bottle loses table support after lift begins.
- The bottle remains clear of the table at lift end and hold end.
- The bottle remains dynamically attached only through finger contact.
- Full-interval drop during the two-second hold is at most `0.010 m`.
- Bottle pose, linear/angular velocity, contact duration, target/readback,
  separation, penetration, and deterministic signature are finite.
- There is no persistent excessive penetration or numerical ejection.
- No forbidden constraint or attachment exists.

The group result requires the configured number of fresh resets and a stable
deterministic signature. A single smoke trial remains `PARTIAL`, even if its
physical result passes.

## Failure Classification

Each trial must use one primary classification:

- `support_settle_failed`
- `horizontal_geometry_failed`
- `gripper_axis_correspondence_failed`
- `vertical_ik_unreachable`
- `contact_not_established`
- `contact_lost_then_free_fall`
- `bilateral_contact_continuous_slip`
- `rotation_induced_escape`
- `normal_force_decay`
- `numerical_penetration_or_ejection`
- `support_clearance_failed`
- `forbidden_contact`
- `inconclusive`
- `stable_hold`

Aggregate status is limited to `PASS`, `FAIL`, `PARTIAL`, or `NOT_RUN`.
Contact persistence alone is not stable-grasp evidence.

## Screenshot Evidence

Capture raw and annotated true-top and side views for at least:

- dynamic support settle;
- open pregrasp;
- vertical descent;
- bilateral contact;
- release into fully dynamic motion;
- support clearance; and
- hold end.

Every annotated image must identify:

- `A` and `B`;
- the bottle axis;
- left and right supplier-CAD fingers;
- the gripper line;
- descent/lift direction;
- contact points and normals;
- table;
- key angles;
- frame/time;
- joint target/readback;
- bottle clearance/drop; and
- `PASS`, `FAIL`, or `PARTIAL`.

Every raw and annotated image must be reviewed individually with the vision
model. Retake views that hide the fingers, contact region, bottle/table
interface, or state change. Screenshots are supporting evidence; runtime
contact, pose, velocity, drop, and deterministic data remain authoritative.

## Evidence Classification And Blockers

Reports must distinguish:

- supplier/project CAD directly confirmed;
- project report reuse;
- local runtime readback;
- numerical calculation;
- engineering inference;
- `TEMPORARY_UNCALIBRATED`;
- `DIAGNOSTIC_CANONICAL_NOT_REAL_CALIBRATION`; and
- `HARD_BLOCKER`.

Current blockers are:

- Missing calibrated camera extrinsics block an episode-image-derived
  absolute world bottle pose.
- Missing validated table-to-base calibration blocks a sim-to-real workcell
  placement claim.
- Missing measured bottle fill state, mass, and material friction block a
  calibrated sim-to-real dynamics claim.

These blockers do not prevent an explicitly local, canonical, isolated
Isaac diagnostic. They do prevent promotion of its placement or dynamics as
measured real-world truth.

## Protection Boundaries

- Do not modify original CAD, source USD, imported robot USD, approved
  signal-correspondence Stage, final/default collider, or previous reports.
- Preserve legacy upright/shoulder-sweep evidence, but mark it inapplicable
  to the current default horizontal task.
- Do not access the real robot or `192.168.1.103`.
- Do not expand cameras, ROS, leaders, workcell construction, or
  bottle-mouth insertion.
- Task 8 remains `NOT_RUN`.
