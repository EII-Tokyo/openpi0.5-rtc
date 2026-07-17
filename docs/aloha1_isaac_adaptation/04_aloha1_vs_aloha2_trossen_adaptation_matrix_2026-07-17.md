# ALOHA1 vs ALOHA2 / Trossen AI Isaac Adaptation Matrix - 2026-07-17

## Purpose

This matrix prevents accidental assumptions when adapting ALOHA2 or Trossen AI Isaac assets to the user's ALOHA1.

## Difference Matrix

| Dimension | ALOHA1 / Current Project Evidence | ALOHA2 / Trossen AI Isaac Evidence | Adaptation Rule |
|---|---|---|---|
| Robot identity | Real setup uses dual follower `vx300s` and dual leader `wx250s`. | Trossen AI Isaac supports WXAI, Stationary AI, Mobile AI. | Do not treat WXAI or Stationary AI as ALOHA1 without mapping. |
| Asset source | Project has local ALOHA-style USD/Menagerie-derived assets and generated workcell layers. | Trossen USDs are generated from `trossen_arm_description`. | Mesh source must be recorded per stage. |
| Unit system | Project user-measured stages use meters and Z-up. | Trossen assets also use SI meters in Isaac. | Unit system is compatible, but pose frames still require validation. |
| Base layout | User-measured ALOHA1 base placement is still approximate in several generated stages. | Trossen Stationary AI has its own mount layout and yaw directions. | Trossen base layout is a reference only. |
| Joint names | Project canonical order uses names like `left_waist`, `left_shoulder`, etc. | Trossen uses names like `joint_0..5` or `follower_left_joint_0..5`. | Always use an explicit mapping table. |
| Joint order | Project canonical action/qpos is 14D. | Trossen Stationary AI demo uses interleaved dual-arm indices. | Never align by index without proof. |
| Joint limits | Real ALOHA1 limits come from Interbotix/ROS/DYNAMIXEL configuration. | Trossen limits come from WXAI URDF. | Read ALOHA1 limits from the real stack or verified ALOHA1 URDF. |
| Continuous joints | Project handles `forearm_roll` and `wrist_rotate` as continuous/extended positions in some paths. | Trossen WXAI public URDF uses finite bounds for corresponding joints. | Wrap/unwrap behavior must be tested separately. |
| Gripper mechanism | ALOHA1 data exposes one gripper scalar per arm, with normalized dataset semantics and real command conversion. | Trossen WXAI uses prismatic carriage joints and mimic. | Gripper cannot be mapped by name or value range alone. |
| Gripper values | ALOHA1 constants include puppet joint open/close and position open/close values. | Trossen examples use carriage values like `0.044` open and `0.022` or `0.0` close. | Calibrate opening distance, command direction, and clamp range. |
| Controller interface | ALOHA1 real control uses ROS topics and Interbotix APIs. | Trossen Isaac controller uses Isaac articulation and IK APIs. | Build a controller adapter; do not reuse commands directly. |
| Home/sleep | ALOHA1 real code defines reset and sleep poses. | Trossen examples often default arms to zero. | Use ALOHA1 home/sleep for ALOHA1 simulation. |
| Camera names | ALOHA1 runtime expects `cam_high`, `cam_low`, `cam_left_wrist`, `cam_right_wrist`. | Trossen has high/low and follower wrist camera concepts. | Same names do not prove same extrinsics or optical frames. |
| Camera role | Bottle insertion depends heavily on `cam_low` and right wrist in this project. | Trossen camera layout is generic Stationary AI/WXAI layout. | Validate projection against real recorded frames. |
| Leader/follower | ALOHA1 leader/follower are physical `wx250s`/`vx300s` roles. | Trossen leader uses Ethernet Trossen arm semantics and external effort/gravity compensation. | Electrical and teleop semantics are not interchangeable. |
| Collision/contact | Project has passed arm replay but not bottle/pipe dynamic contact. | Trossen import enables collision settings, but task-specific bottle/pipe contact is not validated. | Physics validation is a separate gate. |
| RL readiness | Current reports block dynamic grasp and IK configuration. | Trossen Isaac Lab tasks are mostly WXAI examples. | ALOHA1 bottle insertion RL env is not ready until gates pass. |

## Must-Measure Fields

These fields are not optional:

- real ALOHA1 joint names, order, limits, velocity limits, effort limits, operating modes;
- real left/right base poses relative to the table and pipe;
- real gripper command-to-opening mapping;
- real camera intrinsics and extrinsics;
- real bottle and pipe geometry;
- Isaac articulation DOF names, order, limits, and drive parameters;
- collider and contact material configuration;
- end-effector frame definition;
- replay mapping between HDF5 qpos/action and Isaac DOFs.

## Non-Negotiable Validation Gates

1. Dump real and Isaac DOF names/order/limits.
2. Build static mapping from ALOHA1 14D to the selected Isaac articulation.
3. Validate one joint at a time.
4. Validate gripper alone.
5. Validate FK/end-effector pose under matched qpos.
6. Validate camera projection.
7. Validate contact and collisions.
8. Only then validate replay, controller, grasp, and RL.

