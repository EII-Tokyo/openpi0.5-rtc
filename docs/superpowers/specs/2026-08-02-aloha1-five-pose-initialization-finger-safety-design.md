# ALOHA1 Five-Pose Initialization and Finger Safety Design

Date: 2026-08-02
Status: USER-APPROVED DESIGN, IMPLEMENTATION NOT STARTED

## 1. Purpose

Replace the ambiguous mixture of static USD captures and fully initialized
Isaac Sim grasp trials with one frozen experiment contract. The authoritative
runtime task remains the follower-left five-pose horizontal Bottle500 grasp:
dynamic tabletop settle, downward approach, bilateral contact, 0.20 m lift,
and 2.0 s hold.

The design also adds layered protection against two distinct failure modes:

1. rendering or validating supplier-CAD fingers at an illegal authored
   `q=(0, 0)` state because the articulation was never initialized; and
2. a finger being driven or pushed outside the official URDF interval during
   a live simulation, as occurred in `sample_02` when the right finger contacted
   the environment `angled_extrusion` collider.

The old five accepted videos remain immutable historical evidence that the
bottles were grasped, lifted, and held. They are not evidence that every frame
respected the official finger joint limits. A new, separately named batch will
become the formal Task 7 grasp baseline only after all gates in this design
pass.

## 2. Scope and Non-Goals

In scope:

- follower-left only;
- the already approved Z-up, meter-scale CAD-derived diagnostic workcell;
- supplier-CAD handed fingers already used by the five-pose runner;
- deterministic initialization, per-frame joint/contact safety, screenshots,
  videos, and fresh-process repeats;
- an isolated candidate for corrected finger limit or selective finger-pair
  collision behavior when official evidence supports it.

Out of scope:

- the real robot or `192.168.1.103`;
- leader arms, cameras, ROS, pipe insertion, full workcell expansion, or Task 8;
- changing friction, bottle mass, collider geometry, drives, mimic, timestep,
  solver iterations, gravity, grasp trajectory, or acceptance thresholds to
  make the experiment pass;
- changing final/default assets without a separate review and promotion step;
- using global self-collision without a controlled collision-filter audit.

Task 8 remains `NOT_RUN`.

## 3. Evidence Boundary

The new baseline must freeze and report the exact input Stage:

`assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_z_up_meters_diagnostic.usda`

The previously recorded SHA-256 is
`327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9`,
but every new run must recompute it before launch and fail closed if it differs
from the approved manifest.

The failed static screenshot candidate is not an equivalent experiment. It
loaded and rendered authored transforms without `World.reset()`, articulation
initialization, legal finger qpos, physics stepping, or joint readback. It is
retained only as a negative control for the initialization gate.

## 4. Frozen Initialization Contract

Every primary and repeat process must execute the same initialization state
machine. No screenshot, validator result, or runtime result may claim the
five-pose experiment identity unless all steps complete.

### 4.1 Pre-launch identity

The runner records and verifies before launching Isaac Sim:

- absolute Stage path and SHA-256;
- default prim, root prim, sublayers, references, and required prim paths;
- Isaac Sim `5.1.0.0`, Kit `107.3.3`, and PhysX `107.3.26`;
- config path and SHA-256;
- frozen sample identifier, random seed, arm start, bottle position, bottle
  axis, and bottle roll policy;
- expected articulation root and explicit DOF order;
- expected `upAxis=Z`, `metersPerUnit=1`, gravity direction, physics dt, and
  rendering dt;
- drive, mimic, material, bottle, collision, and solver manifest hashes.

Any mismatch produces `FAIL_INITIALIZATION_CONTRACT` before the new Isaac
process starts.

### 4.2 Isaac initialization sequence

Each fresh process performs, in order:

1. open the frozen Stage;
2. read back Stage identity and unit/up-axis metadata;
3. create `World` with the frozen physics and rendering dt;
4. set and read back
   `PhysicsContext.set_solve_articulation_contact_last(True)`;
5. create the expected `SingleArticulation` at the frozen prim path;
6. add it to the scene and call `World.reset()`;
7. verify the exact DOF name/order contract;
8. construct the full initial command, including both finger DOFs;
9. call `set_joints_default_state`, `post_reset`,
   `set_joint_positions`, and `set_joint_velocities`;
10. write the same command through the articulation controller;
11. read back all joint positions, velocities, limits, drives, and mimic data;
12. validate the finger initialization gates below;
13. initialize Bottle500 in setup-only kinematic state;
14. switch it to dynamic before settle and verify the transition readback;
15. begin the formal physics frame counter only after all setup gates pass.

No setup frame contributes to grasp success.

### 4.3 Initialization record

The machine report stores target and readback for every DOF, maximum readback
error, the first physics-frame jump, finger aperture, pair clearance, collider
paths, and a canonical initialization signature. Primary and repeat for the
same sample must have identical initialization signatures.

## 5. Finger Safety Gates

### 5.1 Source-of-truth limits

Finger limits must be derived from the pinned `aloha_vx300s` URDF/Xacro and
its exact mimic definition, then compared with the composed USD and live
articulation readback. The implementation must not assume that the current USD
range `[-0.0642, -0.0138]` is correct merely because PhysX reports it.

Before changing any USD or Isaac runtime behavior, the implementation must
query the directly connected NVIDIA official Isaac MCP and inspect the local
Isaac Sim 5.1 source/schema. Local 5.1 source and runtime readback remain the
version authority.

### 5.2 Pre-physics finger gate

Before the first formal physics frame, require all of the following:

- both targets and readbacks are finite;
- both are inside the source-derived legal intervals;
- the left/right sign and explicit order are correct;
- measured aperture agrees with the two q values under the audited mapping;
- supplier-CAD left/right finger volumes are separate and do not overlap;
- there is no forbidden overlap with the gripper bar or internal gripper
  structure;
- a static, unsolved `q=(0, 0)` state is rejected.

Failure classification is `FAIL_INITIALIZATION_CONTRACT` or
`FINGER_PAIR_OVERLAP`, with raw and annotated evidence saved before exit.

### 5.3 Per-frame runtime gate

Every formal physics frame records and evaluates:

- left/right target, readback, velocity, and target error;
- official-limit margin and composed-USD-limit margin;
- aperture and aperture monotonicity during open/close phases;
- finger-pair geometric clearance or overlap metric;
- finger-finger contact pairs, impulse, normal, and separation;
- finger-environment contact pairs, impulse, normal, separation, and duration;
- whether external contact drives either finger outside the official interval;
- bottle contact and task telemetry already required by the five-pose test.

The runner immediately terminates the affected sample with one or more exact
machine classifications:

- `FINGER_LIMIT_VIOLATION`;
- `FINGER_PAIR_OVERLAP`;
- `FINGER_PAIR_UNEXPECTED_CONTACT`;
- `ENVIRONMENT_CONTACT_FORCED_LIMIT_VIOLATION`;
- `INITIALIZATION_CONTRACT_MISMATCH`.

Finger contact with the table or workcell is not automatically a failure. It
is classified using contact part, impulse, separation/penetration, duration,
joint-limit effect, and task interference. A contact that pushes a finger
outside the official interval or changes the planned grasp is a failure.

## 6. Physical Finger-Pair Protection Candidate

Physical finger-finger collision is a secondary protection layer, not the
primary closing stop. At the currently audited legal closed state
`q=(+0.021, -0.021)`, the supplier-CAD finger colliders remain separated, so
the official joint limits must define the closing boundary.

The candidate investigation proceeds in this order:

1. prove the exact URDF-to-USD limit and mimic transform;
2. if the composed USD limit is a true defect, author the smallest possible
   correction in an isolated diagnostic physics layer;
3. inspect Isaac Sim 5.1 support for articulation self-collision and
   pair-specific collision filtering;
4. if supported, create an isolated candidate that enables self-collision but
   filters every audited internal/adjacent pair except the left-finger to
   right-finger pair;
5. verify that the pair produces contact only after an intentionally invalid
   command and does not change any legal open/close or bottle-grasp trajectory;
6. reject the candidate if it introduces new adjacent-link contacts, solver
   instability, first-frame movement, or deterministic-signature changes.

If pair-only behavior cannot be established with the local 5.1 APIs, the
project retains global self-collision disabled. The accepted protection then
consists of source-correct joint limits, pre-physics geometry validation,
per-frame monitoring, and fail-fast evidence capture. Collider geometry must
not be enlarged to manufacture a mechanical stop.

## 7. Test Strategy

Implementation follows test-driven development. Each gate first receives a
failing unit or report-fixture test before production code changes.

### 7.1 Non-Isaac tests

- reject an initialization record with no reset/readback evidence;
- reject `q=(0, 0)`;
- reject target or readback outside source-derived limits;
- reject a finger-pair overlap;
- classify a harmless contact separately from a contact-forced limit
  violation;
- ensure aggregation cannot report PASS when any frame violates a gate;
- ensure resume logic cannot reuse a historical sample that lacks the new
  initialization and per-frame safety signatures.

### 7.2 Fresh-process Isaac negative controls

- static USD load without articulation initialization must fail the contract;
- legal open, partial close, legal closed, and maximum aperture must pass;
- an intentionally illegal q command must be rejected before formal stepping;
- a controlled environment interference reproducing the `sample_02` failure
  must be detected and classified rather than hidden by later recovery;
- the optional physical pair candidate must detect intentional illegal overlap
  without affecting legal motion.

### 7.3 New formal five-pose baseline

After the negative controls pass, run all five configured bottle poses. Each
sample requires a fresh primary process and a fresh repeat process. The Stage,
physics, controller, bottle, trajectory, and acceptance parameters remain
unchanged from the approved five-pose configuration except for an isolated,
evidence-supported finger-limit candidate if separately approved.

Each sample must pass:

- initialization signature equality;
- full per-frame official-limit compliance;
- no finger-pair overlap or unexpected pair contact;
- no task-interfering environment contact;
- dynamic tabletop settle;
- downward approach and bilateral bottle contact;
- 0.20 m lift and 2.0 s hold within the existing drop gate;
- primary/repeat deterministic physics signature equality;
- complete telemetry, video, and screenshot evidence.

## 8. Visual Evidence

Every new sample records the entire arm, not only the end effector. Collision
display evidence must distinguish full-arm, left/right finger, bottle, table,
and environment colliders.

Required raw and annotated views include:

- post-initialization open state;
- partial close;
- legal closed state without a bottle;
- maximum legal aperture;
- bilateral bottle contact;
- height reached;
- hold end;
- any failure frame and the immediately preceding frame.

Failure annotations identify the two finger collider paths, q target/readback,
official-limit margin, relevant environment collider, contact normal/impulse,
and the specific failed gate. Screenshots and videos remain auxiliary evidence;
machine telemetry determines PASS/FAIL.

## 9. Outputs and Promotion Boundary

New reports and artifacts use a new attempt directory and never overwrite the
accepted historical MP4 files or prior reports. The implementation plan will
choose exact names consistent with the existing Task 7 reporting framework.

The final closure report separates:

- historical grasp-outcome evidence;
- new initialization-contract evidence;
- new per-frame finger safety evidence;
- optional isolated physical-collision candidate evidence;
- user visual review;
- final/default asset promotion status.

No isolated candidate becomes final/default automatically. Promotion requires
two consistent fresh-process validations, unchanged applicable grasp physics,
a written candidate diff, and explicit user approval. Until then Task 7 remains
`PARTIAL`, and Task 8 remains `NOT_RUN`.

## 10. Success Criteria

This design is complete only when:

1. all experiment-like paths share the frozen initialization contract or fail
   before producing admissible evidence;
2. illegal static `q=(0, 0)` can no longer be visually approved as a valid
   gripper state;
3. a `sample_02`-type mid-trajectory limit violation is detected on the first
   offending frame;
4. the new five-pose primary/repeat batch passes all grasp and finger-safety
   gates;
5. the reports distinguish grasp success from joint-limit and collision
   compliance;
6. no final/default asset, old evidence, or unrelated worktree content is
   modified; and
7. Task 8 remains `NOT_RUN`.
