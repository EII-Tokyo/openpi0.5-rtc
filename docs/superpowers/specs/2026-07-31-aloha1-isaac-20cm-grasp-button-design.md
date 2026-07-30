# ALOHA1 Isaac 20 cm Bottle-Grasp Button Design

## Status

User-approved design, 2026-07-31.

This design is limited to a simulation-only diagnostic control inside
Isaac Sim 5.1.0.0. It does not promote an asset, change a final collider, or
authorize access to the real robot.

## Objective

Provide a dockable Isaac Sim window with a button that performs the complete
left-follower Bottle500 pickup sequence:

1. dynamically settle the horizontal bottle on the table;
2. open the gripper above the bottle;
3. approach vertically along world `-Z`;
4. establish physical left and right finger contact;
5. close with the existing diagnostic preload;
6. lift vertically along world `+Z`;
7. reach a measured bottle-to-table clearance of at least `0.200 m`;
8. hold for two seconds.

The 20 cm target is defined by geometry, not by end-effector displacement:

```text
bottle_clearance =
    minimum_world_z(Bottle500 collision geometry)
    - world_z(table collider top)
```

The controller reaches the height target only when
`bottle_clearance >= 0.200 m`. Moving the end effector by 0.20 m without
lifting the bottle is a failure.

## Selected Architecture

Use a project-local, standalone diagnostic launcher that opens the approved
Stage and creates an `omni.ui` window inside Isaac Sim. Do not modify the
native Grasp Editor extension and do not install a permanent Kit extension
until the diagnostic controller has passed.

This approach provides an Isaac-native button while keeping UI code,
trajectory control, reports, and session-only USD authoring isolated from the
source Stage and final assets.

The implementation will have three bounded components:

1. **Pure controller state and evaluation module**
   - Defines phases, transitions, height calculation, abort behavior, and
     PASS/FAIL evaluation.
   - Contains no Isaac imports so it can be tested with project Python.

2. **Isaac runtime adapter**
   - Validates and loads the frozen Stage.
   - Creates the session-only Bottle500 reference and contact reporting.
   - Reads articulation, bottle, table, and contact state.
   - Applies non-blocking arm and gripper targets on physics steps.
   - Runs IK for the vertical approach and lift waypoints.

3. **Dockable diagnostic window**
   - Owns `Run: Grasp + Lift 20 cm`, `Abort`, and `Reset` buttons.
   - Shows phase, target and measured clearance, contact state, IK state,
     finger target/readback, hold drop, and final outcome.
   - Never runs a blocking simulation loop in a button callback.

## Frozen Inputs And Stage Contract

The launcher must use the currently approved table-aligned diagnostic Stage:

```text
/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/table_support_alignment/1.0/aloha1_table_support_aligned_workcell.usda
```

Expected SHA-256:

```text
2b3f76365ed67532f478d995ae859a88b5639975ac07cb7ac8a53ac679e8205c
```

Before enabling the Run button, the launcher must verify:

- absolute Stage path;
- Stage SHA-256;
- default/root prim;
- sublayers and references;
- follower-left articulation prim;
- expected ALOHA six-joint order;
- both handed supplier-CAD finger prims;
- table support prim;
- Bottle500 source path, reference prim, and SHA-256;
- Grasp Editor Variant B report and raw-YAML hashes.

The Run button stays disabled and the UI displays a bounded failure reason if
the contract does not match. No filename guess or remembered Stage is
acceptable.

The source Stage hash must be captured before and after every run and must be
unchanged.

## Grasp Pose And Coordinate Transforms

The controller uses the validated Grasp Editor Variant B object-to-gripper
pose `T_O_G`. At runtime it computes:

```text
T_W_G = T_W_O * T_O_G
```

where:

- `W` is the table-centered world frame already approved for this diagnostic;
- `O` is the Bottle500 object frame, with its origin at the bottle-bottom axis
  center;
- `G` is `follower_left_ee_gripper_link`.

The bottle is horizontal and dynamically supported by the table before the
formal grasp begins. Its directed CAD axis `AB`, table-normal angle,
gripper-line angle, and lowest collision point are recomputed from runtime
world transforms.

Each approach and lift waypoint is solved for the ALOHA six-joint order:

```text
waist
shoulder
elbow
forearm_roll
wrist_angle
wrist_rotate
```

The implementation may reuse the existing verified IK and Variant B helper
logic, but it must not silently reuse a fixed baseline joint vector after the
bottle pose changes.

## Runtime State Machine

The state machine advances only from physics-step callbacks or an Isaac
extension-safe asynchronous update path:

```text
IDLE
  -> VALIDATE
  -> SETUP_KINEMATIC
  -> RELEASE_DYNAMIC
  -> SETTLE
  -> OPEN_PREGRASP
  -> VERTICAL_DESCENT
  -> BILATERAL_CONTACT
  -> CLOSE_PRELOAD
  -> VERTICAL_LIFT
  -> HEIGHT_REACHED
  -> HOLD
  -> PASS | FAIL
```

`ABORTED` is reachable from every active phase.

The setup phase may use a kinematic bottle only to establish the initial
horizontal pose. The bottle must be dynamic with gravity enabled during
settle, contact, lift, and hold. Setup frames cannot contribute to PASS.

The vertical lift uses multiple world-`+Z` IK waypoints. The controller
continues until the measured bottle collision clearance reaches `0.200 m`.
It must stop with FAIL if any of these occurs:

- IK becomes unreachable or exceeds the existing residual gates;
- either finger contact is lost and the bottle begins free fall;
- the bottle remains on the table while the gripper moves up;
- the maximum configured safe run duration is exceeded;
- state becomes non-finite;
- forbidden attachment or constraint evidence appears.

No runtime bottle teleport is permitted after the dynamic release.

## UI Behavior

Window title:

```text
ALOHA1 Bottle Grasp 20 cm — DIAGNOSTIC
```

Controls:

- **Run: Grasp + Lift 20 cm**
  - Enabled only in `IDLE` after the Stage contract passes.
  - Starts one fresh run.
  - Disabled until the run finishes or is aborted.

- **Abort**
  - Stops generating new motion targets.
  - Does not teleport, attach, or freeze the bottle.
  - Leaves the current state visible for inspection.

- **Reset**
  - Stops the timeline.
  - Removes and recreates only session-owned diagnostic prims and controller
    state.
  - Does not edit or save the source Stage.

Displayed readbacks:

- state-machine phase and elapsed time;
- current and maximum Bottle500 clearance;
- table-top world `Z`;
- end-effector world position;
- IK success and residuals;
- left and right finger target/readback;
- left and right contact state;
- bottle vertical and angular velocity;
- hold drop;
- `IDLE`, `RUNNING`, `PASS`, `FAIL`, or `ABORTED`;
- the label `DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING`.

## Physical Acceptance

PASS requires all of the following:

- the bottle dynamically settled on the table;
- the bottle remained dynamic with gravity enabled;
- approach direction passed the existing world-`-Z` gate;
- left and right physical contacts existed before lift;
- no SurfaceGripper, fixed joint, or parent attachment was used;
- the bottle collision clearance reached at least `0.200 m`;
- bilateral contact persisted through the lift and hold;
- the bottle was held for two seconds;
- hold drop was no greater than `0.010 m`;
- state, impulse, velocity, and orientation values remained finite;
- no persistent numerical penetration or ejection occurred;
- source Stage and protected input hashes were unchanged.

Reaching 20 cm does not by itself prove calibrated sim-to-real dynamics.
Existing friction, force-drive, mimic adapter, mass, timestep, solver, and
collider classifications remain unchanged and explicitly diagnostic.

## Evidence

Every button run produces:

- a machine-readable JSON report;
- per-frame JSONL or CSV telemetry;
- full runtime log and exit/error summary;
- source Stage and input hashes before and after;
- a deterministic run signature;
- a raw full-arm video;
- an annotated full-arm video.

The video must show the complete left arm from base through fingers and a
synchronized gripper/bottle close-up. It must cover settle, open, descent,
bilateral contact, close, lift, reaching 20 cm, and hold end.

Accepted evidence must additionally include collision-display screenshots at:

- dynamic release;
- open pregrasp;
- bilateral contact;
- initial support clearance;
- 20 cm height reached;
- hold end.

Each accepted screenshot and the complete video must receive visual-model
self-review. The user's video confirmation remains a separate final gate.
Screenshots and video are supporting evidence; runtime collision, pose,
velocity, and clearance data determine physical PASS.

## Error Handling

- A Stage-contract mismatch prevents Run and does not switch Stage.
- An unavailable NVIDIA or runtime API produces `HARD_BLOCKER` or
  `NOT_SUPPORTED_IN_LOCAL_5_1`, not a guessed fallback.
- An IK failure records the target pose and residuals, stops the trajectory,
  and leaves the scene visible.
- A contact loss, bottle drop, timeout, or non-finite state transitions to
  FAIL and records the first failing frame.
- Abort is never reported as PASS.
- UI callback exceptions are captured in the report and cannot be hidden by
  a zero Kit process exit code.

## Testing

Project-Python tests:

- height is computed from bottle collision minimum and table top;
- EE motion without bottle motion cannot pass;
- all state-machine transitions and illegal transitions;
- Abort and Reset semantics;
- bilateral-contact gate before lift;
- 20 cm gate and two-second hold gate;
- source-hash mutation failure;
- report schema and deterministic signature.

Isaac Sim 5.1 tests:

- exact local UI and timeline API readback;
- frozen Stage contract;
- button responsiveness while physics is running;
- fresh dynamic settle and vertical approach;
- six-joint IK residuals at every lift waypoint;
- physical bilateral contact;
- measured 20 cm clearance;
- two-second hold;
- source Stage unchanged;
- repeatability across fresh resets;
- raw and annotated evidence completeness.

Focused pytest, Ruff, `py_compile`, and the applicable ALOHA mapping
regression suite must run before the launcher is presented as ready.

## Non-Goals

- no real-robot control or access to `192.168.1.103`;
- no follower-right motion;
- no leader, ROS, cameras, full workcell expansion, or pipe insertion;
- no SurfaceGripper or synthetic attachment;
- no physics-parameter calibration;
- no collider promotion or final/default collider change;
- no Task 8 optimization;
- no permanent Kit extension installation in this phase.

Task 8 remains `NOT_RUN`.
