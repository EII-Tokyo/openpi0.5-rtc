# ALOHA1 Minimal Visual Tutor–Grasp Editor Bridge Design

## Status

Scope selected by the user on 2026-07-30. Awaiting written-spec review before
implementation.

This design covers only:

1. a minimal project-local Visual Tutor to Isaac GUI bridge;
2. actual Isaac Sim 5.1 Grasp Editor configuration of Variant B;
3. actual `SIMULATE` and native raw YAML export; and
4. exported pose, left/right finger runtime readback, and transform
   validation.

It does not modify the canonical grasp loader, perform IK, run the dynamic
pickup, record a new grasp video, or start Task 8.

## Simplified MCPJungle Boundary

MCPJungle is required only for NVIDIA official Isaac documentation/API
verification. That minimum gate already passes:

- the only configured Codex MCP connection is `mcpjungle_lab`;
- the Gateway endpoint is reachable;
- NVIDIA Isaac documentation tools are discoverable and callable.

This design does not:

- add a Visual Tutor server to MCPJungle;
- create a new MCPJungle group;
- change MCPJungle networking;
- add a Streamable HTTP MCP bridge;
- modify `/home/eii/mcpjungle-lab`.

The live Visual Tutor bridge is a local application-native project component,
not another MCP connection. NVIDIA API decisions are still checked through
the existing MCPJungle NVIDIA tools before Isaac implementation changes.

## Stop Rule

The local Visual Tutor live round trip is attempted at most twice after its
tests pass.

It must prove:

- Isaac Sim and the project extension are live;
- the exact approved Stage is open;
- timeline and extension state can be read back;
- one bounded `capture state` command reaches the Kit main thread and returns
  a fresh acknowledgement.

If the same round trip fails twice, stop and publish the failure report. Do
not run Grasp Editor configuration, `SIMULATE`, export, IK, or video. Do not
substitute direct MCP, Chrome control, arbitrary shell clicks, or screen
coordinates.

## Confirmed Inputs

- Isaac Sim: `5.1.0.0`
- Kit: `107.3.3`
- PhysX: `107.3.26`
- Grasp Editor: `isaacsim.robot_setup.grasp_editor 2.0.20`
- approved Stage:
  `/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/1.0/aloha1_signal_correspondence_workcell.usda`
- approved Stage SHA-256:
  `d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf`
- articulation:
  `/World/follower_left/vx300s_left/root_joint`
- gripper frame:
  `/World/follower_left/vx300s_left/follower_left_gripper_link`
- object:
  `/World/ALOHA1GraspEditorSession/Bottle500`
- Bottle500 USD:
  `/home/eii/project/openpi0.5-rtc-reward-learning/assets/bottle_500ml/isaac/bottle_500ml_sim.usd`
- Bottle500 SHA-256:
  `16427135f152ec951de2321fd689366d745a2dd389cbe260976631783952533e`
- CAD body grasp coordinate: `s = 69 mm`
- approved control: Variant B

All values are re-read before action. A changed Stage/hash or missing prim
stops without switching Stage.

## Minimal Local Bridge

The existing Visual Tutor Isaac extension gains a small fixed command queue.
Requests are written only by a project-local controller and consumed on the
Kit main thread.

Each request contains:

- generated run ID;
- fixed action enum;
- expected frozen-manifest SHA;
- timestamp.

The extension returns:

- request/run ID;
- action status;
- extension heartbeat;
- Isaac/Kit/PhysX/Grasp Editor versions;
- Stage path and root-layer identifier;
- edit target;
- timeline state;
- selected prims;
- action-specific readback;
- completion timestamp.

Allowed actions are fixed in code:

1. `capture_state`
2. `open_grasp_editor`
3. `prepare_approved_session`
4. `configure_approved_variant_b`
5. `simulate_approved_variant_b`
6. `export_approved_raw_grasp`
7. `capture_evidence`
8. `cleanup_approved_session`

There is no caller-supplied Python, shell, screen coordinate, arbitrary prim
path, arbitrary Stage path, joint name, joint value, export path, ROS action,
or real-robot action.

This is not a separate security-isolation project. The fixed action enum and
frozen manifest exist only to keep the ALOHA experiment reproducible and to
prevent accidental operations on the wrong Stage.

## Real Grasp Editor Operation

The bridge must operate the actual local Grasp Editor window:

`Tools → Robotics → Grasp Editor`

It uses the local Isaac Sim 5.1 extension's semantic UI/model callbacks. It
does not use viewport coordinates.

Required selections:

- Select Gripper:
  `/World/follower_left/vx300s_left/root_joint`;
- Select Rigid Body:
  `/World/ALOHA1GraspEditorSession/Bottle500`;
- Gripper Frame:
  `/World/follower_left/vx300s_left/follower_left_gripper_link`;
- Rigid Body Frame:
  `/World/ALOHA1GraspEditorSession/Bottle500`.

The Bottle and any Grasp Editor side effects are authored only into the
existing anonymous/session diagnostic layer. The frozen Stage is not saved.

After `READY`, verify:

- edit target remains diagnostic;
- root file SHA is unchanged;
- approved Bottle collision inventory is unchanged;
- material, drives, mimic disposition, solver, and collider settings are
  unchanged.

Unexpected persistent authoring stops and runs cleanup.

## Variant B

First verify the runtime DOF order:

```text
0 waist
1 shoulder
2 elbow
3 forearm_roll
4 wrist_angle
5 wrist_rotate
6 gripper
7 left_finger
8 right_finger
```

Configure:

- active `left_finger`;
- open `0.057 m`;
- closed target `0.021 m`;
- speed `0.02 m/s`;
- composed maximum effort readback `5.0 N`;
- fixed/observer `right_finger`, setup `-0.057 m`;
- all six arm DOFs fixed at approved readback;
- auxiliary `gripper` fixed;
- `Include All DOFs` disabled.

`right_finger` is an observer in the project report, not a native Grasp
Editor field. Its position and velocity are recorded every simulation step.

## SIMULATE

Execute the real UI path:

`Author Grasp → Simulate Grasp → SIMULATE`

`SKIP SIM` is forbidden.

Pass requires:

- UI result `Passed Grasp Testing`;
- machine result `GraspTestResults.success == true`;
- exactly one terminal callback;
- finite telemetry;
- expected left active and right observer behavior;
- arm DOFs remain fixed within recorded drift;
- only approved Bottle/finger contacts contribute to the test.

This result is only:

`GRASP_TESTER_PASS_ONLY_NOT_TASK_PASS`

It does not prove dynamic table pickup or static hold.

## Native Raw Export

Export to a new file under:

`.codex/artifacts/20260730-aloha1-grasp-editor-ik-evidence/grasp_editor_gui_raw/`

Do not overwrite:

`configs/aloha1_grasps/bottle500_horizontal_body_grasp.isaac_grasp.yaml`

Independently reopen and verify:

- `format == isaac_grasp`;
- `format_version == 1.0`;
- exact object and gripper frames;
- exactly one grasp named `grasp_0`;
- c-space keys exactly `left_finger`;
- pregrasp keys exactly `left_finger`;
- right finger absent from both maps;
- pregrasp left value `0.057`;
- finite position/quaternion/confidence;
- normalized quaternion;
- non-empty new file;
- recorded SHA-256 and size.

The raw file remains unchanged evidence. Canonical promotion remains
`BLOCKED_SCHEMA_MISMATCH`.

## Pose, Finger, And Transform Validation

In a fresh Isaac process, reload the same frozen Stage and native raw YAML
without IK.

Verify:

- imported active joint is `left_finger`;
- `right_finger` remains fixed/observer;
- right-finger trajectory is preserved as readback evidence;
- arm and auxiliary gripper DOFs remain fixed;
- object and gripper frame paths are unchanged;
- exported pose equals runtime
  `T_O_G = inverse(T_W_O) * T_W_G`;
- finite homogeneous transforms;
- rotation orthogonality and determinant `+1`;
- no reflection or scale;
- forward/inverse closure;
- CAD body coordinate remains `s = 69 mm`;
- correct supplier-CAD fingers remain installed;
- frozen Stage and protected assets retain their hashes.

Right mimic accuracy remains:

`INCONCLUSIVE_NO_APPROVED_MIMIC_TOLERANCE`

No right-finger value is invented in the raw YAML.

## Evidence

Save raw and annotated screenshots for:

- live extension/Stage probe;
- Grasp Editor selection;
- reference frames;
- Variant B joint table;
- pre-simulation state;
- `Passed Grasp Testing`;
- native export and independent readback;
- true-top and side final geometry.

Each screenshot records versions, Stage path/hash, active/observer joints,
target/readback, frame/time, and result. Screenshots are vision-reviewed but
do not replace machine data.

## Cleanup

Always:

1. pause/stop timeline;
2. close/reset Grasp Editor;
3. remove callbacks and queued requests;
4. restore edit target;
5. remove the anonymous diagnostic layer;
6. restore root runtime metadata;
7. verify root content, dirty state, sublayers, references, and frozen hashes;
8. never save the Stage.

Cleanup mismatch is `FAIL_CLEANUP`.

## Testing

Implementation follows TDD:

- command enum and unknown-action rejection;
- run-ID freshness;
- queue timeout and two-failure stop;
- extension heartbeat and stale-response rejection;
- wrong Stage/hash rejection;
- Kit main-thread request/ack;
- exact Variant B fields;
- actual Grasp Editor callback result;
- raw YAML validation and no-overwrite;
- transform closure;
- cleanup/hash restoration.

Run:

- project Visual Tutor tests;
- focused ALOHA Grasp Editor tests;
- actual Isaac live probe;
- actual capture-state round trip;
- `SIMULATE` and raw export;
- fresh-process import;
- Ruff and `py_compile`.

Exit code zero alone is not sufficient.

## Status Ceiling

Successful completion may report:

```text
MCPJUNGLE_NVIDIA_OFFICIAL = PASS
LOCAL_VISUAL_TUTOR_LIVE_BRIDGE = PASS
ACTUAL_GRASP_EDITOR_GUI = PASS
VARIANT_B_STRUCTURE = PASS
NATIVE_RAW_EXPORT = PASS
GRASP_EDITOR_TRANSFORM_VALIDATION = PASS
RIGHT_MIMIC_ACCURACY = INCONCLUSIVE_NO_APPROVED_MIMIC_TOLERANCE
CANONICAL_PROMOTION = BLOCKED_SCHEMA_MISMATCH
IK = NOT_RUN
DYNAMIC_GRASP_VIDEO = NOT_RUN
TASK_PASS = NOT_ESTABLISHED
TASK8 = NOT_RUN
```
