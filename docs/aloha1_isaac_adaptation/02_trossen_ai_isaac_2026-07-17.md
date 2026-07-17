# Trossen AI Isaac - 2026-07-17

## Scope

This note records what can be reused from Trossen AI Isaac and where it differs from the user's ALOHA1.

Primary sources:

- Local: `external/trossen_ai_isaac/README.md`
- Local: `external/trossen_ai_isaac/assets/robots/asset_generation.md`
- Local: `external/trossen_ai_isaac/scripts/controller.py`
- Local: `external/trossen_ai_isaac/scripts/stationary_ai_pick_place.py`
- GitHub: `https://github.com/TrossenRobotics/trossen_ai_isaac`
- Docs: `https://docs.trossenrobotics.com/trossen_arm/main/tutorials/trossen_ai_isaac.html`
- Description source: `https://github.com/TrossenRobotics/trossen_arm_description`

## Verified Facts

Trossen AI Isaac is an official Trossen Isaac Sim / Isaac Lab integration. It supports:

- WidowX AI single-arm variants;
- Stationary AI;
- Mobile AI;
- USD robot assets;
- differential IK examples;
- pick-and-place/follow-target examples;
- Isaac Lab tasks for WXAI reach/lift/open-drawer style experiments.

The documented environment is:

- Ubuntu 22.04;
- Isaac Sim 5.1.0;
- Isaac Lab 2.3.0;
- Python 3.11.

The USD assets are generated from `TrossenRobotics/trossen_arm_description`, using Isaac Sim's URDF importer and later post-import refinement. The asset-generation note states that the USDs use parameters matching real-world hardware specifications as of 2025-11-25.

## Controller Facts

The standalone controller implements damped least-squares differential IK and exposes high-level methods such as:

- `set_end_effector_pose`;
- `open_gripper`;
- `close_gripper`.

For Stationary AI, the demo hard-codes interleaved dual-arm indices:

- left arm: `[0, 2, 4, 6, 8, 10]`;
- right arm: `[1, 3, 5, 7, 9, 11]`;
- left gripper main DOF: `12`;
- right gripper main DOF: `14`;
- default arm positions: twelve zeros;
- gripper open value: `0.044` meters.

The end-effector link names used by the Stationary AI example are:

- `/follower_left_link_6`;
- `/follower_right_link_6`.

## What Can Be Reused

Trossen AI Isaac is a good starting point for:

- Isaac Sim 5.1-compatible USD asset organization;
- robot bringup scripts;
- differential IK controller structure;
- Stationary AI dual-arm pick-and-place script shape;
- Isaac Lab task layout;
- gripper mimic-joint treatment;
- USD generation workflow from Trossen arm descriptions.

## What Must Not Be Assumed

Do not assume:

- Stationary AI equals the user's ALOHA1;
- `joint_0..joint_5` or `follower_left_joint_0..5` have the same signs, zero positions, or limits as ALOHA1 `waist/shoulder/elbow/forearm_roll/wrist_angle/wrist_rotate`;
- Trossen gripper carriage meters equal ALOHA1 normalized gripper action;
- Trossen default home pose equals ALOHA1 home or sleep;
- Trossen camera names or nominal extrinsics equal the user's real cameras;
- Trossen leader/follower electrical and communication semantics equal the user's ALOHA1 ROS/DYNAMIXEL setup.

## Main Risk For ALOHA1 Adaptation

The largest risks are not the visible robot mesh. They are:

1. DOF identity and order.
2. Gripper command/opening semantics.
3. End-effector frame identity.
4. Camera extrinsics.
5. Contact and collision geometry.
6. Controller and electrical semantics.

Therefore, Trossen AI Isaac should be used as the scaffold, not as the final ALOHA1 truth.

