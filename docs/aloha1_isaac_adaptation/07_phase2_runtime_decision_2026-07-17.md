# Phase 2 Runtime Decision - 2026-07-17

## Decision

Use Trossen `stationary_ai` as the Isaac Sim runtime standard for the next scaffold.

Do not continue treating the current generated ALOHA1 USD as the main simulation asset. It can remain a kinematic naming/reference input, but it is not acceptable as a training, contact, camera, or RL asset.

## Evidence

Headless Isaac Sim runtime inspection:

- Markdown report: `reports/aloha1_isaac_adaptation/phase2_runtime_inspection_20260717/phase2_runtime_inspection.md`
- JSON report: `reports/aloha1_isaac_adaptation/phase2_runtime_inspection_20260717/phase2_runtime_inspection.json`
- Final bounded log artifact: `.codex/artifacts/20260717-231255_phase2-isaac-runtime-inspection-exit0`

The script did not touch the real robot and did not save the Isaac stage.

## Runtime Result

### Current Generated ALOHA1 Assets

The side assets and wrapper initialize as Isaac articulations:

- left side: 9 runtime DOFs;
- right side: 9 runtime DOFs;
- wrapper: two 9-DOF articulations.

The DOF names are:

```text
waist
shoulder
elbow
forearm_roll
wrist_angle
wrist_rotate
gripper
left_finger
right_finger
```

However, the same assets report:

```text
mesh_count = 0
collider_count = 0
camera_count = 0
```

The Isaac log also reports unresolved visual references under the generated ALOHA1 asset. Therefore these assets are not complete simulation assets even though the articulation can initialize.

### Trossen Stationary AI

Trossen `stationary_ai.usd` initializes as one bimanual articulation:

```text
num_dof = 16
joint_count = 32
mesh_count = 60
collider_count = 32
camera_count = 4
```

Runtime DOF order:

```text
follower_left_joint_0
follower_right_joint_0
follower_left_joint_1
follower_right_joint_1
follower_left_joint_2
follower_right_joint_2
follower_left_joint_3
follower_right_joint_3
follower_left_joint_4
follower_right_joint_4
follower_left_joint_5
follower_right_joint_5
follower_left_left_carriage_joint
follower_left_right_carriage_joint
follower_right_left_carriage_joint
follower_right_right_carriage_joint
```

This makes Trossen the better Isaac Sim scaffold: it has a complete articulation, meshes, colliders, materials, and cameras.

## What This Means

The current generated ALOHA1 USD is useful for:

- arm DOF naming reference;
- rough kinematic comparison;
- historical explanation of what has already been tried.

It is not useful as the primary basis for:

- contact simulation;
- bottle grasping;
- camera validation;
- Isaac Lab RL;
- replay with visual/contact acceptance.

## What Can Be Copied From Trossen

The following can be copied or mirrored from Trossen as Isaac infrastructure:

- USD asset layering pattern;
- single bimanual articulation organization;
- articulation root strategy;
- link, joint, mesh, collider, material, and camera completeness gates;
- headless runtime inspection style;
- controller and IK scaffolding as Isaac plumbing.

These are not automatically ALOHA1 truth. They are only the working Isaac Sim structure.

## What Must Be Replaced With ALOHA1 Facts

The following cannot be guessed:

- real ALOHA1 joint signs;
- joint offsets;
- joint limits;
- velocity and effort limits;
- left/right base poses;
- home and sleep poses;
- normalized gripper command mapping;
- gripper physical opening;
- camera names, intrinsics, extrinsics, and optical frame convention;
- any DYNAMIXEL or ROS operating-mode assumption.

If any of these are uncertain, mark the field as `REQUIRES_REAL_DATA_VERIFICATION`.

When file evidence is not enough, verify from the real robot stack on `192.168.1.103` using read-only diagnostics. Do not infer physical or electrical truth from Trossen or from visual similarity.

## Next Scaffold Contract

The next implementation target is a Trossen-backed ALOHA1 scaffold report.

It passes only if:

- Isaac starts headless;
- no real robot is touched;
- no stage is saved;
- the scaffold loads from Trossen structure, not from the broken generated ALOHA1 USD;
- one bimanual articulation initializes;
- mesh, collider, and camera counts are nonzero;
- unresolved robot visual/collision references are absent;
- a proposed ALOHA1 adapter table is emitted;
- every adapter field is marked as `CONFIRMED`, `UNKNOWN`, or `REQUIRES_REAL_DATA_VERIFICATION`.

## Blocked Gates

The following remain blocked:

- controller reuse;
- gripper mapping;
- camera projection;
- contact validation;
- bottle grasp;
- bottle-pipe insertion task;
- Isaac Lab RL.

They stay blocked until the scaffold contract and one-joint validation gates pass.

