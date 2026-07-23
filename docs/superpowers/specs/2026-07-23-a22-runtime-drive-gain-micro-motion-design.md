# A22 Runtime Drive-Gain And Gravity-Off Micro-Motion Design

## Goal

Advance the A19 clean ALOHA articulation from A21 target-buffer validation to
bounded physical motion in Isaac Sim. A22 must prove that every arm DOF can
follow a small signed position-target perturbation in the expected direction,
settle, and return to its baseline under gravity-off, collision-off conditions.

A22 does not validate gravity-on holding, finger motion, self-collision,
environment collision, object contact, HDF5 replay, policy inference, or
training.

## Approved Scope

A22 uses runtime-only PhysX drive gains. It does not author gains into USD and
does not save, flatten, export, or overwrite any stage.

The live gate:

- loads the A19 clean articulation in fresh headless Isaac processes;
- preserves the A20 path-resolved 16-DOF runtime order;
- applies reviewed gains to the complete articulation;
- disables gravity and requires collision to remain disabled;
- perturbs one arm DOF at a time by a limit-safe `0.25 degree`;
- tests all six left arm DOFs before starting a fresh right-arm process;
- tests all six right arm DOFs only after the complete left batch passes;
- restores the complete target vector and runtime gain buffers before exit;
- verifies that the A19 USD SHA-256 did not change.

The four finger DOFs receive reviewed holding gains so their targets remain
defined, but A22 never commands finger opening or closing. Finger motion and
mimic/contact behavior remain outside this gate.

No real robot, ROS bridge, 103 container, camera stream, HDF5 file, policy,
reward code, or training process is involved.

## Evidence Priority

The required evidence entry point is:

`docs/aloha1_isaac_adaptation/107_a22_real_aloha_drive_gain_evidence_chain_2026-07-23.md`

A22 uses the following priority:

1. Phase 97 same-lineage Isaac drive-target gains;
2. real ALOHA and ROBOTIS controller evidence as a control-intent anchor;
3. Interbotix Gazebo gains as relative per-joint-strength evidence;
4. Trossen Stationary AI only as a cross-robot physical sanity reference.

A19 `stiffness=0` must not be described as the real ALOHA hardware
configuration. DYNAMIXEL `Position_P_Gain=800` must not be copied numerically
to PhysX `stiffness=800`; the domains and units differ.

## Approaches Considered

### Selected: Same-Lineage Isaac Prior With One Fixed Candidate

Use the Phase 97 values as the only A22 live candidate:

```text
arm stiffness = 1600
arm damping = 100
finger stiffness = 200
finger damping = 50
```

These values already passed a 50 Hz `drive_target` run on the same
Menagerie-derived ALOHA asset family. A22 still treats them as a candidate
because A19 has a new single-root articulation composition.

A22 does not run an automatic gain sweep. If the fixed candidate fails, A22
fails and a separately reviewed design revision must explain the next
candidate.

### Rejected: Copy Real DYNAMIXEL Register Numbers

Tony/ROBOTIS `Position_P_Gain=800` is a servo-controller table value with
internal scaling and nested position, velocity, current, and PWM behavior. It
is not a joint-end `Nm/rad` stiffness. Direct copying would create a false
physical equivalence.

### Rejected: Copy Stationary AI Or Blindly Sweep

Stationary AI has different links, inertias, effort limits, and gripper
mechanics. Its values can bound plausibility but cannot define ALOHA gains.
Blind sweeps would make failures harder to attribute and could normalize unsafe
motion simply by searching until a permissive threshold passes.

## Runtime Architecture

### Pure Contract Module

A focused pure-Python module will define:

- the approved gain vector by canonical DOF path;
- the twelve arm micro-motion cases in canonical order;
- a limit-safe signed delta for each case;
- per-frame motion metrics;
- batch and aggregate pass/fail decisions;
- safety/readiness flags.

It must not import Isaac, USD, ROS, or robot libraries.

### Single-Batch Isaac Probe

One probe process owns one side:

- Batch L tests the six left arm paths;
- Batch R tests the six right arm paths.

The probe uses the A20 runtime articulation discovery and path resolver rather
than hard-coded raw indices. It may call only the reviewed runtime APIs needed
to:

- read DOF paths, names, types, limits, positions, velocities, targets,
  stiffnesses, and dampings;
- set the complete position-target, stiffness, and damping buffers;
- disable gravity;
- step a headless `World` with `render=False`;
- close the in-memory stage without saving.

It must not call state-teleport APIs such as `set_joint_positions`,
`set_dof_positions`, or `set_dof_velocities`; it must not apply an
`ArticulationAction`, effort, or velocity target.

### Coordinator

The coordinator:

1. validates exact A19/A20/A21 prerequisite evidence and artifact hashes;
2. performs a static A22 gain/motion preflight;
3. launches Batch L through the Isaac virtual-environment interpreter;
4. validates exactly one terminal JSON marker and a clean exit;
5. stops if Batch L fails;
6. launches Batch R in a fresh process;
7. aggregates the two markers;
8. verifies the pre/post A19 USD SHA-256;
9. writes bounded JSON and Markdown reports.

All live logs go through `codex-evidence`; raw Kit output remains in
`.codex/artifacts/`.

## Gain And Baseline Procedure

For each fresh process:

1. load A19 and initialize exactly one 16-DOF runtime articulation;
2. resolve all DOFs by full canonical path and verify the A20 order contract;
3. read and preserve the complete original target, stiffness, and damping
   arrays;
4. read finite positions and velocities and verify all positions are inside
   path-aligned limits;
5. set the complete target vector equal to current positions;
6. set the reviewed complete stiffness and damping vectors;
7. disable gravity for all articulation bodies;
8. step ten warmup frames at `physics_dt=0.02` with the baseline targets held;
9. define the post-warmup finite position vector as the batch baseline;
10. reject the batch if warmup produces a limit violation, non-finite value,
    more than `0.25 degree` motion on any arm DOF, or more than `0.0001 m`
    motion on any finger DOF.

Authored drive type and max-force values remain unchanged. A22 records them for
provenance but does not override them.

## Per-Joint Micro-Motion Procedure

Each arm case runs sequentially and starts only after the previous case has
returned to the batch baseline.

For a selected path:

1. choose `+0.25 degree` when the upper-limit room is at least the delta;
2. otherwise choose `-0.25 degree` when the lower-limit room is at least the
   delta;
3. fail static preflight if neither direction is limit-safe;
4. copy the complete baseline target vector and change only the selected index;
5. step at most 100 frames at `physics_dt=0.02`;
6. record full 16-DOF positions, velocities, and targets after every frame;
7. restore the complete baseline target vector;
8. step at most 100 recovery frames;
9. require successful baseline restoration before testing the next path.

Exactly one target element may differ during the outbound phase. Finger
targets and all non-selected arm targets remain at the baseline.

## Acceptance Metrics

Let:

```text
delta = 0.25 degree = 0.004363323129985824 radians
direction = sign(target - baseline)
```

Each arm case passes only when all conditions hold:

- all recorded positions, velocities, targets, gains, and limits are finite;
- no DOF violates its runtime limit;
- the selected joint's maximum signed displacement is at least
  `0.50 * delta`;
- the selected joint never moves more than `2.00 * delta` from baseline;
- the selected joint never moves opposite the approved direction by more than
  `0.10 * delta`;
- the final outbound target error is at most `0.20 * delta`;
- the maximum absolute velocity over the final ten outbound frames is at most
  `0.01 rad/s`;
- every non-selected arm DOF moves by at most `0.10 * delta`;
- every finger DOF moves by at most `0.0001 m`;
- outbound target readback changes exactly one intended raw runtime index;
- recovery returns the selected joint to within `0.20 * delta` of baseline;
- recovery returns every non-selected arm joint to within `0.10 * delta` of
  baseline;
- the maximum absolute velocity over the final ten recovery frames is at most
  `0.01 rad/s`.

Threshold equality passes. A metric missing the required number of frames
fails closed.

These thresholds define a small-signal runtime gate, not a claim of identified
physical equivalence to the real robot.

## Failure And Recovery Semantics

Every probe uses `try/finally` restoration after any target or gain write.

On an ordinary tracking or settling failure, the probe:

1. restores the complete baseline target vector;
2. performs only the bounded recovery frames;
3. writes the complete pre-probe original target vector;
4. restores the original complete stiffness and damping buffers;
5. verifies original target and gain readback without stepping again;
6. emits one terminal FAIL marker;
7. closes without saving.

On a hard numerical or safety failure, defined as a non-finite value, a limit
violation, or selected-joint excursion beyond `2.00 * delta`, the probe stops
physics stepping immediately. It still attempts in-memory target and gain
buffer restoration/readback, including the complete pre-probe original target
vector, but it does not step further merely to improve a report. Process
teardown is the final containment boundary.

The successful path performs the same final no-step teardown: write and verify
the complete pre-probe original target vector, restore and verify the original
stiffness and damping buffers, then close the process. The post-warmup batch
baseline is only the reference for motion metrics and between-case recovery;
it must never replace the original target buffer in final restoration claims.

Any restoration failure, probe exception, timeout, extra marker, nonzero child
exit, prerequisite mismatch, or USD hash change fails A22.

Batch L failure prevents Batch R from running.

## Output Contract

Each child emits exactly one JSON marker:

```text
A22_RUNTIME_DRIVE_GAIN_MICRO_MOTION_RESULT=<json>
```

The marker includes:

- schema version and side;
- config, mapping, stage, A20, and A21 artifact paths and hashes;
- raw runtime DOF paths, names, types, order, and limits;
- original and reviewed gain arrays;
- original, baseline, outbound, recovery, and restored target evidence;
- gravity-disabled and collision-disabled assertions;
- per-case frame counts and metrics;
- restoration attempts and results;
- prohibited-action flags;
- readiness flags;
- one terminal status.

The aggregate pass status is:

```text
PASS_A22_RUNTIME_DRIVE_GAIN_GRAVITY_OFF_ARM_MICRO_MOTION
```

Required readiness flags after a pass:

```text
gravity_off_arm_micro_motion_ready = true
finger_motion_ready = false
gravity_on_hold_ready = false
collision_ready = false
contact_ready = false
replay_ready = false
training_ready = false
```

A22 does not set the existing USD `aloha:controlReady` attribute and does not
claim overall robot readiness.

## Static And Runtime Prerequisites

Before either live batch:

- A19 static audit must pass against the configured stage;
- A20 Asset Validator, Layer 1, and three-run Layer 2 evidence must pass
  exactly;
- A21 policy target-limit preflight and two-batch target readback must pass
  exactly;
- config, mapping, stage, and prerequisite artifact hashes must match;
- A19 must expose one articulation, sixteen unique finite-limit DOFs, and the
  exact path-resolved A20 order;
- collision readiness must remain false and the live stage must not add or
  enable collision geometry;
- the official NVIDIA Isaac MCP prerequisite must have been satisfied for the
  implementation session.

## Testing Strategy

Implementation follows strict test-driven development.

Pure tests cover:

- Phase 97 gain-vector construction by canonical path;
- rejection of missing, duplicate, wrong-type, non-finite, negative, or
  unapproved gains;
- all twelve path-resolved single-arm cases;
- limit-safe `+delta` and `-delta`, including equality at the boundary;
- exact one-index target changes;
- direction, excursion, tracking, settling, cross-joint, finger-drift, and
  recovery threshold boundaries;
- insufficient-frame and non-finite traces;
- hard-failure classification;
- left-failure blocking right;
- marker normalization, exactly-one-marker parsing, hash binding, aggregation,
  and readiness flags.

Fake-view probe tests cover:

- full original target/gain capture;
- reviewed gain writes;
- baseline target write;
- sequential one-index perturbations;
- restoration after setter, step, readback, and evaluator failures;
- immediate no-further-step behavior after hard failures;
- rejection of state teleport, action, effort, velocity-target, save, and
  export calls.

Live execution runs only after:

- focused A22 tests pass;
- the complete A19/A20/A21 regression suite passes;
- source-policy inspection passes;
- static A22 preflight passes;
- the A19 USD pre-run hash is recorded.

After live Batch L and Batch R, the regression suite, A19 static audit, and USD
hash verification run again.

## Acceptance Criteria

A22 is complete only when:

1. the documented Phase 97 gain candidate is applied at runtime without
   changing authored max force or drive type;
2. gravity and collision remain disabled;
3. every left arm DOF and every right arm DOF passes the single-joint
   direction, tracking, settling, cross-motion, and recovery metrics;
4. fingers remain within the passive drift threshold and are never actively
   commanded;
5. the complete original targets and gain buffers are restored and verified;
6. no forbidden state/action/save API is used;
7. Batch L and Batch R run in fresh processes and aggregate deterministically;
8. all prerequisite and regression gates pass;
9. the A19 USD SHA-256 is unchanged;
10. the report keeps finger motion, gravity-on hold, collision, contact,
    replay, and training readiness false.

## Next Gate

After A22 passes, the next specification may define gravity-on hold and
collision validation. A22 alone does not authorize adding support-frame,
lower-camera-housing, or water-pipe colliders, pressing Play in the GUI, or
running contact/replay tests.
