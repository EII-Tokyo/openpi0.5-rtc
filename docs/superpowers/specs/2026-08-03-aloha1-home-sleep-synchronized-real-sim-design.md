# ALOHA1 Synchronized Real–Simulation Home/Sleep Design

**Date:** 2026-08-03

**Status:** USER_APPROVED_DESIGN

**Selected approach:** dual local playback with a shared manifest and sample-index alignment

**Hardware scope:** Stationary ALOHA 1 `follower_left`, Interbotix ViperX-300 6DOF / `aloha_vx300s`

## 1. Objective

Execute the same hash-frozen `Home -> Sleep -> Home` command stream on the real
`follower_left` and the Isaac Sim 5.1 digital `follower_left`, observe both at
the same time, and retain synchronized command, readback, timing, and video
evidence. The experiment repeats the trajectory three times without commanding
`follower_right` or either gripper.

This is the first real-versus-digital signal-correspondence experiment. It can
establish command identity, joint semantics, endpoint correspondence, and
measured dynamic differences. It does not by itself establish a calibrated
sim-to-real dynamics model.

## 2. Relationship To The Previous Design

This document supersedes Sections 3.2, 6, 7, and 8 of
`docs/superpowers/specs/2026-08-03-aloha1-home-sleep-digital-twin-design.md`
for the live experiment in two ways:

1. the user-selected official historical legal Sleep command replaces the
   newer current-Humble out-of-limit comparison command; and
2. the real and digital runners execute concurrently from one coordinated
   start rather than as unrelated sequential playbacks.

The earlier Task 8 closure, historical failed Humble run, completed digital
qualification, and immutable evidence remain valid historical records.

## 3. Frozen Command Authority

The sole command authority is
`reports/aloha1_mapping/aloha1_home_sleep_command_manifest.json`.

It contains:

- joint order: `waist`, `shoulder`, `elbow`, `forearm_roll`, `wrist_angle`,
  `wrist_rotate`;
- Home: `[0.0, -0.96, 1.16, 0.0, -0.3, 0.0] rad`;
- Sleep: `[0.0, -1.80, 1.55, 0.0, -1.57, 0.0] rad`;
- command rate: `50 Hz`;
- move duration: `5 s`;
- endpoint hold: `1 s`;
- cycle count: `3`;
- sample count: `1850`;
- command signature:
  `d481b71bc8d6160fae0bdc1b264715e782712565064bb18099f8a9a4883f499e`.

The selected Sleep source is the official Interbotix historical commit
`dbc6aefb53e956181fe97f60474f1ad292491f0c`, file
`interbotix_ros_xsarms/interbotix_xsarm_control/config/aloha_vx300s.yaml`,
BSD-3-Clause, source-blob SHA-256
`a5c809a5dd1cd6fb795a8f4f4cbf69de6e0133e1916cb8816061d29f4a8aa75e`.
This is an explicit cross-version command selection. The current-Humble
driver/URDF limits remain frozen and its newer Sleep vector remains comparison
evidence only.

Neither runner may regenerate interpolation, reorder joints, substitute a
convenience Sleep function, or clamp an illegal target. Both consume the stored
1850 samples and verify the manifest SHA-256 before becoming ready.

## 4. Architecture

The implementation has five isolated components.

### 4.1 Local coordinator on machine 101

The coordinator owns the run ID and immutable input manifest. It launches or
attaches only to the new experiment processes, waits for both workers to report
`READY`, schedules a future start, collects their completion records, and
builds the comparison report. It never publishes ROS motor commands itself.

### 4.2 Isaac worker on machine 101

The Isaac worker starts a fresh Isaac Sim 5.1.0.0 / Kit 107.3.3 / PhysX
107.3.26 process and loads the already qualified frozen Stage:

`/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_z_up_meters_diagnostic.usda`

The expected Stage SHA-256 is
`327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9`,
but the worker must recompute it and verify root prim, sublayers, references,
and required articulation prims before loading. It uses the existing 60 Hz
physics rate and rational 50 Hz command scheduler without changing physics,
drive, collider, material, mass, inertia, solver, or timestep.

The Isaac GUI opens on workspace 2 using the project-approved application
launcher. The worker records a fixed full-arm view and does not operate any
user-started Isaac process.

### 4.3 Real worker on machine 103

The real worker is a project-local ROS1 Noetic program under
`/home/eii/openpi0.5-rtc-reward-learning`. It verifies the exact live ROS
namespace, topic types, joint order, driver identity, and official Interbotix
message semantics before importing or instantiating a publisher.

Expected topics from existing project evidence are:

- command candidate: `/puppet_left/commands/joint_group`;
- readback candidate: `/puppet_left/joint_states`.

These names are discovery expectations, not permission to publish. The actual
message type, group name, field semantics, queue behavior, and command
acceptance path must be confirmed from the live read-only ROS graph and pinned
official Interbotix source. A mismatch blocks live execution.

The real worker loads the same stored manifest locally and schedules samples
from its own monotonic clock. It does not depend on one command packet crossing
the LAN every 20 ms. This keeps network jitter out of the motor command cadence.

### 4.4 `cam_high` recorder on machine 103

The user confirmed that `cam_high` contains the complete real
`follower_left`. The recorder subscribes to `/cam_high`, whose current project
adapter expects `aloha.msg/RGBGrayscaleImage`. Read-only preflight must confirm
the live type and image encoding.

For every source frame it preserves:

- ROS header stamp when present;
- nested image header stamp when present;
- local monotonic receive time;
- local wall-clock receive time;
- source sequence number when present;
- decoded frame index, dimensions, encoding, and SHA-256.

Recording begins before either worker is armed and continues through the final
Home hold. The raw frame/timestamp manifest is retained alongside the encoded
MP4 so video encoding cannot replace source timing evidence.

### 4.5 Offline alignment and report builder

Raw command and readback logs remain immutable. Alignment uses
`run_id + command_signature + cycle + segment + sample_index` as the primary
key. Monotonic and wall-clock timestamps quantify dispatch and observation
latency but are not used to reorder the command sequence.

## 5. Start Coordination

The coordinator uses a prepare/ready/start protocol:

1. generate a unique run ID and freeze all input hashes;
2. start the real recorder in read-only mode;
3. start the real worker with publishing disabled and wait for `REAL_READY`;
4. start the Isaac worker paused at Home and wait for `ISAAC_READY`;
5. measure machine-101/machine-103 wall-clock offset and round-trip jitter with
   a bounded read-only exchange;
6. schedule a future start far enough ahead for both workers to arm;
7. each worker converts that start into a local monotonic deadline and records
   the conversion inputs;
8. both workers apply sample zero at their local deadline and record the actual
   application timestamp.

The observed first-sample start skew is reported. A skew no greater than one
command period (`20 ms`, derived from 50 Hz) qualifies as
`SYNCHRONIZED_START_PASS`. A larger skew does not permit timestamps to be
silently rewritten; the run is classified `POST_ALIGNED_ONLY` and may still be
used for endpoint comparison, but not for simultaneous transient-response
claims.

Workers schedule each sample from `start_deadline + index * 20 ms`; they do not
accumulate repeated sleeps. A missed deadline is recorded. The real worker
must never burst multiple late motor commands to catch up.

## 6. Recorded Signals

### 6.1 Shared command record

Every sample records:

- schema version, run ID, manifest SHA-256, command signature;
- cycle, segment, sample index, nominal time;
- six target positions in explicit joint order;
- scheduled local deadline and actual application time;
- lateness and command acceptance result.

### 6.2 Real record

For each `/puppet_left/joint_states` message, retain the original message fields
and record:

- ROS timestamp and local receive timestamps;
- joint names and unmodified source order;
- six mapped arm positions;
- velocities only when present or when a separately labeled finite-difference
  value is calculated;
- efforts only when the live driver provides an officially defined value;
- message age, interval, duplicates, gaps, and non-finite values;
- current driver/hardware error state and abort state.

`Present_Current` is not a required gate. If the exact-model official driver
cannot reliably expose it, the report records `NOT_AVAILABLE`; no guessed
register or substitute value is used.

### 6.3 Isaac record

Every physics frame records:

- physics frame, simulation time, and exact `dt`;
- active command index and target;
- six-DOF readback, velocity, and target error;
- actual command-application frame/time;
- stationary `follower_right` and both-gripper readback;
- finite state, limit, endpoint, and contact gates.

## 7. Safety And Authorization

Implementation and offline tests do not authorize the real robot. Before any
connection to `192.168.1.103`, the user must explicitly authorize remote
read-only access. Before publishing, the same live session requires explicit
real-motion authorization and operator confirmation that:

- the workspace is clear;
- `cam_high` shows the complete arm;
- the arm is at or within the predeclared Home-entry tolerance;
- the correct `follower_left` namespace and six-joint order are verified;
- `/puppet_left/joint_states` is current and continuous;
- no second controller or DYNAMIXEL Wizard owns the bus;
- hardware error, voltage, and temperature checks pass when officially and
  safely observable;
- a verified stop/abort mechanism is immediately available.

The live adapter remains fail-closed until the deployed official control path
and stop behavior are confirmed. Software abort never guesses that torque-off
is safe. It invokes only the verified deployed stop/hold behavior, then stops
issuing trajectory samples.

Immediate abort conditions include stale/missing/non-finite readback, joint
order changes, rejected commands, opposite-direction motion, actual limit
violation, unintended right-arm or gripper motion, driver/hardware error,
unplanned resistance/collision, loss of `cam_high`, or operator stop.

The experiment is supervised and is never an unattended real-robot run.

## 8. Comparison Metrics And Classifications

For each joint, segment, and cycle, compute:

- command identity and direction;
- first-response latency;
- start skew and phase lag;
- rise and settling time;
- overshoot;
- endpoint and steady-state error;
- RMSE and maximum absolute position error;
- velocity difference when both velocity meanings are verified;
- Home/Sleep repeatability and residual drift.

Numeric acceptance tolerances for dynamic similarity are frozen before the
real result is visible. They must come from command period, timestamp quality,
encoder/controller resolution, and the completed digital baseline. A
successful endpoint run is not automatically a dynamic-calibration pass.

Literal final classifications are:

- `KINEMATIC_AND_SIGNAL_DIGITAL_TWIN_PASS_DYNAMIC_CALIBRATION_PENDING`;
- `SYNCHRONIZED_KINEMATIC_AND_DYNAMIC_CORRESPONDENCE_PASS`;
- `POST_ALIGNED_ONLY`;
- `SIGNAL_MAPPING_FAILURE`;
- `KINEMATIC_ENDPOINT_MISMATCH`;
- `DYNAMIC_RESPONSE_MISMATCH`;
- `REAL_EXECUTION_ABORTED`;
- `INCONCLUSIVE`.

## 9. Visual Evidence

Retained media include:

- raw `cam_high` frame/timestamp manifest and MP4;
- Isaac full-arm MP4 from workspace 2;
- a post-aligned side-by-side MP4;
- raw and annotated initial Home, first Sleep, first returned Home, third
  Sleep, and final Home frames from both sources.

Annotations identify source, run ID, cycle, segment, sample index, target,
readback, timing offset, and PASS/FAIL/PARTIAL. Every retained key image and all
three videos receive visual-model review. The review verifies complete-arm
visibility, visibly distinct Home/Sleep states, correct cycle order, readable
annotations, and absence of duplicated stages. Machine signals remain the
acceptance authority.

If the live run aborts or produces a reproducible mismatch, retain the last
valid frame, first anomalous frame, and final failure frame with the responsible
joint/link and matching telemetry marked. Do not rerun with changed physics or
hardware parameters to hide the failure.

## 10. Implementation Boundaries

Implementation proceeds in the following gates:

1. pure Python manifest, scheduler, alignment, and safety-state tests;
2. fake real transport plus fake camera tests;
3. independent Isaac 5.1 replay and recording regression;
4. local coordinator dry-run with no network;
5. separately authorized 103 read-only discovery and recorder preflight;
6. operator-confirmed live publishing and concurrent three-cycle run;
7. offline alignment, visual review, and report closure.

No gate may be skipped because a later component appears to work.

The implementation does not modify final/default USD, physics parameters,
joint limits, robot firmware, motor registers, real controller gains, ROS
driver configuration, `192.168.1.103` files outside the project, or Task 8
assets. It does not control `follower_right`, grippers, leaders, cameras other
than read-only `cam_high`, or any grasp/insertion task.

## 11. Deliverables

Planned deliverables are:

- a synchronized-experiment configuration bound to the frozen manifest;
- a transport-independent real worker core and fake transport;
- a project-local ROS1 real adapter and `cam_high` recorder;
- an Isaac 5.1 synchronized worker and full-arm recorder;
- a local prepare/ready/start coordinator;
- raw real, Isaac, command, and camera telemetry;
- synchronized comparison JSON/CSV/Markdown;
- three reviewed videos and paired key-stage screenshots;
- focused tests, ALOHA mapping regression, Ruff, and `py_compile` logs;
- README and `.codex/TASK_STATE.md` updates;
- logical commits without push.

Raw high-output logs and media remain under a dated `.codex/artifacts/`
directory and are referenced by absolute path and SHA-256 from machine-readable
reports.

## 12. Acceptance Boundary

The design is implemented when all offline/fake/Isaac components are verified
and the read-only real preflight can produce a complete machine report. The
live experiment is complete only after explicit live authorization, the
supervised three-cycle run, immutable telemetry capture, signal alignment, and
visual review.

If live authorization or a verified stop path is unavailable, implementation
ends at `READY_FOR_SUPERVISED_REAL_EXECUTION`; this is not a failure and must
not be reported as a real/digital correspondence pass.
