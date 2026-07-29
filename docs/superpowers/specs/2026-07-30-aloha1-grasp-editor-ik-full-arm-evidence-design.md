# ALOHA1 Grasp Editor, IK, And Full-Arm Evidence Design

## Status

User-approved design, 2026-07-30.

This design corrects the dependency order used by the first horizontal
Bottle500 pickup smoke test. The old run remains immutable failure evidence,
but its directly constructed end-effector pose and its local-only videos do
not contribute to acceptance of the corrected grasp.

Task 8 remains `NOT_RUN`.

## Goal

Establish a traceable grasp configuration for the supplier-CAD ALOHA follower
gripper and the project Bottle500, verify that the configuration maps
correctly through the ALOHA six-arm-DOF kinematic chain, and then rerun the
horizontal pickup with machine-readable and full-arm visual evidence.

The dependency order is:

1. freeze the approved Stage and source inputs;
2. define and verify coordinate frames;
3. configure and export the grasp with the local Isaac Sim 5.1 Grasp Editor;
4. validate the exported grasp geometrically without IK;
5. verify the ALOHA/Interbotix/Lula kinematic correspondence;
6. solve and validate IK;
7. run the dynamic pickup; and
8. record synchronized evidence that always includes the complete articulated
   arm.

An IK solver reporting success is not evidence that the grasp pose is correct.
A Grasp Editor viewport preview is not evidence that the ALOHA kinematic
mapping is correct. Dynamic pickup acceptance requires both chains to pass.

## Runtime Boundary

All Isaac work is restricted to the local installation:

- Isaac Sim `5.1.0.0`;
- Kit `107.3.3`;
- PhysX `107.3.26`; and
- `isaacsim.robot_setup.grasp_editor 2.0.20`.

Before changing Isaac code, USD, Stage contents, physics behavior, GUI
automation, or runtime settings, use NVIDIA's official Isaac capability
through the MCPJungle Gateway. Do not use latest or 6.0 APIs.

The local Grasp Editor source is:

`.venv_issac/lib/python3.11/site-packages/isaacsim/exts/isaacsim.robot_setup.grasp_editor`

Its locally read format contract is:

- `format: isaac_grasp`;
- `format_version: 1.0`;
- `object_frame`;
- `gripper_frame`;
- grasp `position` and `orientation`;
- `pregrasp_cspace_position`; and
- `cspace_position`.

## Frozen Inputs

### Approved Stage

Absolute path:

`/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/1.0/aloha1_signal_correspondence_workcell.usda`

Current recorded SHA-256:

`d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf`

Before every mutating or runtime process, recompute and freeze:

- absolute path;
- SHA-256;
- root/default prim;
- sublayers;
- references;
- follower-left articulation;
- follower-left base and end-effector prims;
- supplier-CAD finger prims; and
- user-confirmed table prim.

The current hash is evidence, not permission to load a changed file. All new
frames, Bottle500 composition, grasp visualization, and diagnostics must be
authored in isolated session or diagnostic layers.

### Project Bottle500

- CAD:
  `/home/eii/project/openpi0.5-rtc-reward-learning/assets/bottle_500ml/cad/bottle_500ml.FCStd`
- CAD SHA-256:
  `3594f60200e54181bc8480a229484293a0d386c146d3f235b32e31a0c16bbf8a`
- Isaac USD:
  `/home/eii/project/openpi0.5-rtc-reward-learning/assets/bottle_500ml/isaac/bottle_500ml_sim.usd`
- USD SHA-256:
  `16427135f152ec951de2321fd689366d745a2dd389cbe260976631783952533e`
- CAD-authored bottle axis length: `206 mm`.
- CAD-derived body interval: `s = 18-120 mm`.
- Default body grasp coordinate: `s = 69 mm`.

The bottle remains horizontal, dynamically settles on the table, and is
approached from above along task-world `-Z`.

### ALOHA Sources

- Generated follower-left URDF:
  `generated/urdf/follower_left.urdf`
- Lula descriptor:
  `configs/aloha1_lula_follower_left.yaml`
- Explicit joint map:
  `configs/aloha1_joint_map.yaml`
- User-confirmed real-data window:
  `/home/eii/project/bottles_data/episode_18.hdf5`, frames `208-244`
  inclusive.

The implementation must re-freeze their hashes. The existing hashes in prior
reports remain historical evidence.

## Protection Boundary

Do not modify:

- source STEP or FCStd;
- original URDF or Xacro;
- imported source USD;
- the approved signal-correspondence Stage;
- default/final collider;
- current convex-hull or convex-decomposition baselines;
- existing failure reports or videos; or
- the current physics parameter baseline.

Do not connect to or control the real robot, and do not access
`192.168.1.103`.

Do not add:

- SurfaceGripper;
- fixed joints;
- parent attachment;
- runtime bottle teleport after the dynamic phase begins; or
- artificial friction or force changes.

No workcell expansion, camera expansion, ROS work, leader work, bottle-mouth
insertion, or Task 8 optimization is in scope.

## Coordinate-Frame Contract

The implementation must keep six frames distinct:

- `W_U`: immutable USD `/World` frame;
- `W_T`: tabletop task-world frame;
- `B`: follower base frame used by the kinematic solver;
- `O`: Bottle500 object/reference frame;
- `G`: Grasp Editor gripper frame; and
- `E`: Lula end-effector frame.

No pair of frames may be treated as identical unless the composed runtime
transform is read back and verified as identity.

### USD World Readback

The current frozen Stage readback establishes:

- `metersPerUnit = 1.0`;
- `upAxis = Z`;
- `/World` has an identity world transform;
- follower-left base:
  `(-0.4695, -0.0190, +0.0200) m`, identity rotation;
- follower-right base:
  `(+0.4695, -0.0190, +0.0200) m`, rotation `Rz(180 degrees)`;
- table cube center:
  `(0, 0, -0.0984000015258789) m`;
- table dimensions:
  `1.100 x 0.600 x 0.015 m`; and
- table top:
  `z = -0.0909000015258789 m`.

These are digital Stage readbacks, not real-hardware calibration data.

### Tabletop Task World

Define `W_T` at the geometric center of the table's top surface:

```text
origin = (0, 0, -0.0909000015258789) in W_U
+Z_T   = table normal, upward
+X_T   = follower_left base toward follower_right base
+Y_T   = Z_T cross X_T
```

The current Stage axes already satisfy this orientation, so the digital
transform is a pure translation:

```text
T_WU_WT =
[ 1  0  0   0                  ]
[ 0  1  0   0                  ]
[ 0  0  1  -0.0909000015258789 ]
[ 0  0  0   1                  ]
```

Do not rebase or rewrite `/World`. Author `W_T` only in an isolated diagnostic
layer and use it as the task/reporting frame.

In `W_T`, the current digital base origins are:

```text
B_left  = (-0.4695, -0.0190, +0.1109000015258789) m
B_right = (+0.4695, -0.0190, +0.1109000015258789) m
```

### Calibration Boundary

The table-centered digital frame provides a calibration method but is not a
completed real calibration.

A later real measurement must:

1. identify the physical tabletop center;
2. measure the left and right base mounting centers;
3. fit the tabletop plane from at least three non-collinear measured points;
4. choose upward `+Z_T`;
5. define `+X_T` from left base to right base;
6. compute `+Y_T = Z_T cross X_T`;
7. orthonormalize the axes;
8. solve both base transforms; and
9. record point residuals, orthogonality error, determinant, date, and source.

Until then, reports must label the frame
`DIGITAL_STAGE_READBACK_NOT_REAL_CALIBRATION`.

## Transform Chain

The Grasp Editor exports the gripper frame relative to the object frame:

```text
T_O_G
```

At runtime:

```text
T_WT_G(t) = T_WT_O(t) * T_O_G
```

The IK target in the robot-base frame is:

```text
T_B_G(t) = inverse(T_WT_B) * T_WT_G(t)
```

If the Lula end-effector frame `E` differs from Grasp Editor frame `G`, the
fixed transform must be read and applied explicitly:

```text
T_B_E(t) = T_B_G(t) * inverse(T_E_G)
```

The implementation must state its matrix convention and multiplication order.
It must save all input and output matrices and verify:

- finite entries;
- homogeneous last row/column as appropriate to the convention;
- rotation orthogonality;
- rotation determinant `+1`;
- no reflection or scale;
- forward and inverse closure;
- `W_T -> O -> G` closure against the actual previewed gripper frame; and
- `W_T -> B -> E -> G` closure against the same frame.

Any transform applied both to the solver base pose and manually to the target
is a duplicate-transform failure.

## Grasp Editor Compatibility Probe

Before authoring the final diagnostic grasp, verify the local extension
against the ALOHA follower articulation.

The probe must determine:

- whether the full follower articulation can be selected;
- whether the embedded finger DOFs can be selected independently of the six
  arm DOFs;
- which prim is used as the articulation/gripper reference;
- whether the selected `gripper_frame` may be an end-effector subframe;
- the exact active-joint names and order;
- whether mimic joints appear as active, read-only, or derived;
- whether open and closed values are read back in radians or linear units;
- whether test execution changes arm joints;
- whether the exported YAML imports without loss; and
- whether an import/export round trip is deterministic.

If the local extension cannot isolate the embedded ALOHA gripper, create a
diagnostic-only standalone gripper articulation using the same supplier-CAD
finger geometry, joint definitions, limits, mimic semantics, and frame
relationship. Record it as
`DIAGNOSTIC_GRIPPER_ONLY_NOT_FINAL_ROBOT_CONFIGURATION`.

The diagnostic copy may not replace the follower asset. Mapping back to the
full robot requires the verified `T_E_G`.

## Grasp Authoring

Use the actual Isaac Sim 5.1 Grasp Editor GUI to configure one initial grasp:

`horizontal_body_grasp`.

Export to:

`configs/aloha1_grasps/bottle500_horizontal_body_grasp.isaac_grasp.yaml`

The GUI evidence must show:

- frozen Stage path and hash;
- Bottle500 object/reference frame;
- ALOHA gripper frame;
- selected active gripper joints;
- pregrasp/open values;
- grasp/closed values;
- grasp position;
- grasp orientation; and
- export path.

The exported YAML is authoritative. A screenshot is supporting evidence.

The grasp position must use the CAD-derived body coordinate `s = 69 mm`.
Neither bottle roll nor gripper roll may be adjusted by visual trial and
error. If their relationship cannot be derived from CAD, the runtime Stage,
episode 18, or the verified gripper geometry, record a `HARD_BLOCKER`.

## Pre-IK Grasp Validation

Reload the exported YAML in a fresh Isaac process. Do not use IK during this
gate.

The actual composed Stage must demonstrate:

- Bottle500 horizontal and supported on the table;
- both supplier-CAD finger inward surfaces face the bottle;
- the two inward contact-region centers lie on opposite sides of the bottle
  axis;
- their signed radial distances have opposite signs;
- they lie at the same intended bottle-body section within tolerance;
- the contact-center line projected into table `XY` is perpendicular to
  bottle axis `AB`;
- pregrasp aperture exceeds the CAD section diameter plus the validated
  contact-envelope allowance;
- the close trajectory moves both inward surfaces toward the bottle;
- neither finger starts inside the bottle;
- no finger is mirrored, swapped, or arbitrarily rotated; and
- no collision, material, drive, or solver parameter changes during this
  gate.

Failure prevents IK from running. It is classified as
`GRASP_EDITOR_GEOMETRY_GATE_FAIL`, not as an IK or physics failure.

## ALOHA Kinematic Correspondence

The ALOHA follower uses six arm DOFs plus the gripper mechanism. Lula may be
used only after proving that its model matches the ALOHA model used by this
project.

Extract and explicitly compare:

1. generated URDF arm joint names and order;
2. composed USD articulation DOF names and order;
3. `configs/aloha1_joint_map.yaml`;
4. local ALOHA/Interbotix control and dataset order;
5. local official Interbotix kinematic descriptions or model parameters;
6. joint axes, origins, zero offsets, and limits;
7. Lula descriptor c-space order;
8. Lula base frame; and
9. Lula end-effector frame.

Do not sort joint names alphabetically.

### Required Numerical Tests

- One-joint-at-a-time FK for all six arm joints.
- Positive and negative perturbations from an interior reference pose.
- URDF/USD observed-motion direction agreement.
- Lula FK versus composed USD runtime frame pose.
- Lula FK versus the locally available Interbotix/ALOHA kinematic reference.
- IK solution within every joint limit.
- `IK -> Lula FK` target-position and target-orientation residual.
- `IK -> USD runtime readback` target-position and target-orientation
  residual.
- Previous-waypoint seeding and continuity.
- Detection of alternate IK branches and discontinuities.
- Deterministic repeat from a fresh process.

The report must distinguish:

- solver convergence;
- model correspondence;
- target reachability;
- achieved runtime pose; and
- grasp suitability.

`IK success=True` alone is never a passing result.

If no independent official Interbotix/ALOHA FK reference is available
locally, the cross-implementation item is `HARD_BLOCKER`, while URDF/USD/Lula
closure continues.

## IK And Motion Sequence

Only after the pre-IK grasp and kinematic-correspondence gates pass:

1. read dynamic Bottle500 `T_WT_O`;
2. compute `T_WT_G` from the exported grasp;
3. convert to the verified Lula end-effector target;
4. solve an open pregrasp vertically above the grasp;
5. solve a primarily `-Z_T` descent while preserving orientation;
6. close using the exported gripper c-space state;
7. preserve the closed state;
8. solve a `+Z_T` lift; and
9. hold for two seconds.

Each waypoint is verified by FK and runtime readback. An unreachable or
discontinuous waypoint fails without moving the bottle or changing the task.

## Actual Visual Evidence

Abstract diagrams and perspective views cannot establish geometric
acceptance.

### Orthographic Screenshots

Capture actual Isaac Stage views:

- true top `XY`, camera forward `-Z_T`;
- front `XZ`, camera forward along the selected `Y_T` direction; and
- side `YZ`, camera forward along the selected `X_T` direction.

Every view must record:

- orthographic camera mode and scale;
- camera world matrix;
- `W_T`, `B`, `O`, `E`, and `G` axes;
- complete follower articulation;
- table;
- Bottle500;
- `A`, `B`, and the bottle axis;
- supplier-CAD left and right fingers;
- inward contact-region centers;
- grasp section;
- relevant angles and distances; and
- raw and annotated absolute paths.

Axes and labels may not obscure the contact geometry. The visual model must
review every raw and annotated image. Runtime matrices and geometry remain
authoritative.

### Full-Arm Video

Every primary evidence frame must include:

- follower base;
- shoulder;
- elbow;
- forearm;
- wrist;
- gripper;
- Bottle500; and
- table.

Use a synchronized split layout:

- main region: complete articulated arm and task;
- inset: gripper, inward surfaces, Bottle500, and contact data.

The required phases are:

- home/reference;
- gripper open;
- above-bottle pregrasp;
- vertical descent;
- bilateral contact;
- close/preload;
- lift;
- support clear; and
- hold end.

Every frame stream must contain a shared frame index and simulation time.
Supplemental close-up video may exist, but it cannot replace the full-arm
primary video.

The vision model must review the complete video or a documented dense frame
sample that covers every phase transition. Retake when:

- any joint link leaves the frame;
- shoulder, elbow, wrist, finger, bottle, or table is occluded;
- the inset is not synchronized;
- open and closed states are visually indistinguishable;
- descent or lift direction is ambiguous; or
- annotations obscure the motion.

Visual `PASS` means the evidence is readable. It does not mean the physical
grasp passed.

## Dynamic Acceptance

Retain the existing physical baseline:

- mass `0.020 kg`, labelled `TEMPORARY_UNCALIBRATED`;
- friction `0.7`;
- restitution `0`;
- physics `60 Hz`;
- existing collider;
- existing drive;
- existing mimic/control disposition;
- existing solver iterations; and
- hold interval `2 s`;
- maximum hold drop `0.010 m`.

Do not change any of those while correcting the grasp pose or IK chain.

A dynamic trial passes only when:

- the Grasp Editor geometry gate passed;
- ALOHA kinematic correspondence passed;
- Bottle500 settled dynamically on the table;
- approach was primarily `-Z_T`;
- both fingers established physical contact before lift;
- contact paths, positions, normals, impulses, and separations were finite;
- the bottle lost table support;
- the bottle remained supported only through finger contact;
- hold drop remained within the gate;
- bottle pose, velocity, and angular velocity remained finite;
- there was no forbidden attachment;
- there was no persistent excessive penetration or numerical ejection; and
- the deterministic signature repeated.

The result must distinguish:

- grasp configuration failure;
- IK/model correspondence failure;
- contact not established;
- contact lost then free fall;
- bilateral contact with continuous slip;
- rotation-induced escape;
- normal-force decay;
- penetration/ejection; and
- stable hold.

## Deliverables

### Configuration

- `configs/aloha1_grasps/bottle500_horizontal_body_grasp.isaac_grasp.yaml`
- `configs/aloha1_table_task_frame.yaml`

### Tools

- `tools/probe_aloha1_grasp_editor_compatibility.py`
- `tools/validate_aloha1_grasp_transform_chain.py`
- `tools/validate_aloha1_aloha_ik_correspondence.py`
- `tools/capture_aloha1_grasp_editor_evidence.py`
- updates to the horizontal kinematics, dynamic grasp, annotation, and video
  tools so that they consume the exported grasp and record the full arm.

### Tests

- `tests/aloha1_mapping/test_grasp_editor_compatibility.py`
- `tests/aloha1_mapping/test_grasp_transform_chain.py`
- `tests/aloha1_mapping/test_aloha_ik_correspondence.py`
- updates to the horizontal grasp screenshot/video tests.

### Reports

- `reports/aloha1_mapping/aloha1_table_task_frame.json`
- `reports/aloha1_mapping/aloha1_grasp_editor_compatibility.json`
- `reports/aloha1_mapping/aloha1_grasp_transform_validation.json`
- `reports/aloha1_mapping/aloha1_ik_correspondence_v2.json`
- `reports/aloha1_mapping/aloha1_grasp_editor_screenshot_review.json`
- `reports/aloha1_mapping/aloha1_full_arm_video_review.json`
- updated horizontal kinematics and dynamic grasp reports without overwriting
  the original failed-run evidence.

High-output logs, screenshots, videos, and temporary diagnostic layers go
under:

`.codex/artifacts/20260730-aloha1-grasp-editor-ik-evidence/`

## Status Rules

Reports use only:

- `PASS`;
- `FAIL`;
- `PARTIAL`; or
- `NOT_RUN`.

Important substatus values include:

- `GRASP_EDITOR_COMPATIBILITY`;
- `GRASP_EDITOR_GEOMETRY`;
- `TRANSFORM_CLOSURE`;
- `ALOHA_KINEMATIC_CORRESPONDENCE`;
- `IK_REACHABILITY`;
- `IK_RUNTIME_READBACK`;
- `FULL_ARM_VISUAL_EVIDENCE`; and
- `DYNAMIC_PICKUP`.

The first corrected smoke trial remains `PARTIAL` even if it physically
passes. Repeated trials run only after the smoke gate passes.

Task 8 remains `NOT_RUN` until the applicable Task 5 and Task 7 gates pass.

## Implementation Order

1. Freeze inputs and capture the current failure boundary.
2. Author and validate the isolated tabletop task frame.
3. Probe local Grasp Editor compatibility with the embedded ALOHA gripper.
4. Configure and export the grasp in the actual GUI.
5. Fresh-process import and deterministic round-trip check.
6. Run the pre-IK geometry and transform-closure gates.
7. Audit independent ALOHA/Interbotix kinematic sources.
8. Run the six-DOF correspondence and IK/FK/runtime tests.
9. Generate actual orthographic Stage evidence and complete visual review.
10. Run one corrected dynamic smoke trial.
11. Record and review the synchronized full-arm/inset video.
12. Run repeated trials only when the smoke trial passes.
13. Rerun applicable Task 7 validation, focused pytest, Ruff, and
    `py_compile`.
14. Update README and `.codex/TASK_STATE.md`.
15. Keep Task 8 `NOT_RUN`.
