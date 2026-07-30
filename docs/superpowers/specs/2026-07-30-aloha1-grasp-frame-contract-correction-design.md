# ALOHA1 Grasp Frame Contract Correction Design

## Status

User-approved in conversation on 2026-07-30. This document supersedes only
the frame-selection portions of the earlier Grasp Editor and full-gripper CAD
clearance designs. Their source protection, evidence, dynamic pickup, video,
and Task 8 boundaries remain active.

## Goal

Make the project Bottle500 grasp portable between the local Isaac Sim 5.1
Grasp Editor, ALOHA/Interbotix kinematics, Lula IK, and the composed follower
USD by assigning one unambiguous meaning to every coordinate frame.

The correction must not change source CAD, the approved Stage, collider,
friction, drive, mimic, bottle mass, timestep, solver settings, or Task 8.

## Considered Approaches

### A. Canonical `ee_gripper_link` everywhere — selected

Use `follower_left_ee_gripper_link` as both the Grasp Editor gripper frame and
the Lula/IK end-effector frame. Keep the supplier-CAD pad/contact frame as a
geometry-only helper and convert it to `ee_gripper_link` with an audited fixed
transform.

This matches the generated URDF, Interbotix Modern Robotics description,
MoveIt SRDF, and the Isaac Sim 5.1 Grasp Editor requirement that a robot
gripper frame be meaningful to the corresponding URDF/motion-generation
model.

### B. Keep Lula on `gripper_link` and convert every target

This is mathematically valid if every consumer applies the fixed
`gripper_link -> ee_gripper_link` transform exactly once. It is rejected as
the default because the current project already demonstrated how easy it is
to mix that frame with the CAD pad center, and because it differs from the
published ALOHA/Interbotix end-effector semantics by 107.2 mm.

### C. Use the CAD pad center directly as the Grasp Editor and IK frame

Rejected. The helper is session-only and is not a frame in the generated
ALOHA URDF or the current Lula robot model. It remains useful for geometric
clearance and contact construction, but an exported grasp using it cannot be
consumed directly by the motion-generation pipeline.

## Coordinate Contract

The matrix convention is:

```text
T_A_B maps homogeneous column vectors from frame B into frame A.
```

The frames are:

- `W_U`: immutable USD `/World`.
- `W_T`: table-task world with origin at the tabletop geometric center and
  axes aligned with the current USD world; `+Z` is the table normal.
- `B`: `follower_left_base_link`.
- `O`: Bottle500 bottom-center frame; local `+Z` runs from bottom to mouth.
- `G`: `follower_left_ee_gripper_link`.
- `C`: supplier-CAD effective pad/contact-center helper.

`C` is not an IK frame and is not the exported Grasp Editor frame.

The required transforms are:

```text
T_O_G = inverse(T_W_O) * T_W_G
T_W_G = T_W_O * T_O_G
T_B_G = inverse(T_W_B) * T_W_G
T_O_G = T_O_C * inverse(T_G_C)
```

The project must never reuse `T_O_C` numeric values while merely changing the
frame label to `G`.

## Frozen ALOHA Frame Evidence

The generated URDF fixed chain is:

```text
gripper_link
  --42.825 mm--> ee_arm_link
  -- 0.000 mm--> gripper_bar_link
  --25.875 mm--> fingers_link
  --38.500 mm--> ee_gripper_link
```

Therefore:

```text
translation(gripper_link -> ee_gripper_link) = 0.1072 m along local +X
rotation(gripper_link -> ee_gripper_link) = identity
```

Two helper frames exist in the historical evidence:

- the older whole-pad-face center is 0.11127188479610935 m from
  `gripper_link`; and
- the later user-approved complete-gripper clearance frame is
  0.13552080444282988 m from `gripper_link`.

Only the user-approved clearance frame is the current `C`, so it is
0.02832080444282989 m ahead of `ee_gripper_link` along the same local axis.
The older helper remains immutable superseded evidence. These numbers must be
re-derived from frozen evidence at runtime; this document does not authorize
hard-coded use without hash and closure checks.

## Bottle And Grasp Semantics

The Bottle500 object origin remains at the bottle-bottom center. The default
body grasp coordinate remains 0.069 m along the Bottle500 local axis. It is a
grasp offset, not an object-origin replacement.

The user-approved complete-gripper clearance candidate expresses `T_O_C`.
Before native Grasp Editor use it must be converted to `T_O_G`. For the
frozen current candidate, the expected translation is approximately:

```text
T_O_C translation = [0.0033365257, 0.0004430772, 0.069] m
T_O_G translation = [-0.0247378183, -0.0032850828, 0.069] m
```

The orientation is unchanged because the frozen `G -> C` transform has
identity rotation.

## Gripper DOF Contract

Only `left_finger` is an active Grasp Editor DOF:

```text
right_finger = -left_finger
```

`right_finger` is a mimic observer and must not appear in native exported
`cspace_position` or `pregrasp_cspace_position`.

Import-time UI selections are authoritative. The local Grasp Editor 2.0.20
ignores YAML object/gripper frame fields while importing and makes every DOF
listed in `cspace_position` active. Runtime evidence must therefore record
the selected `O`, selected `G`, active DOF set, and mimic readback.

## Data Flow

1. Freeze the approved Stage and source hashes.
2. Read `T_G_C` from the audited CAD/URDF mapping.
3. Construct the desired CAD contact pose `T_O_C`.
4. Convert it to `T_O_G`.
5. Configure native Grasp Editor with `O`, `G`, and only `left_finger`.
6. Simulate/export an original `isaac_grasp` YAML.
7. Reload it in a fresh process and compute `T_W_G` with local
   `GraspSpec.compute_gripper_pose_from_rigid_body_pose`.
8. Solve IK for the same `G`.
9. Verify FK and composed USD runtime readback for `G`.
10. Only after those gates pass, run dynamic horizontal pickup and the five
    random-position video suite.

## Acceptance Gates

The frame gate passes only if:

- Stage, URDF, CAD mapping, bottle USD, and config hashes are frozen;
- all transforms are finite rigid transforms with determinant `+1`;
- `gripper_link -> ee_gripper_link` is re-derived as 0.1072 m;
- `T_O_C = T_O_G * T_G_C` closes within 1e-9 m and 1e-9 rad;
- object/world/gripper forward and inverse transforms close;
- YAML records `G = ee_gripper_link`;
- YAML records only active `left_finger`;
- Lula reports `ee_gripper_link` as a valid frame;
- Interbotix POE, Lula FK, and composed USD `G` agree within the existing
  documented tolerances;
- native Grasp Editor import/export reproduces the pose and active DOF
  semantics in a fresh process; and
- screenshots show `W_T`, `B`, `O`, `G`, `C`, bottle axis, and approach axis
  without using viewport appearance as the numeric verdict.

Failure of this gate prevents IK or dynamic pickup from being described as
validated.

## Evidence And Status

Old `pad_center`/`gripper_link` results remain immutable historical evidence
and are marked `SUPERSEDED_WRONG_GRIPPER_FRAME_SEMANTICS`. They are not
deleted or silently rewritten.

Task 8 remains `NOT_RUN`.
