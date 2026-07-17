# Google DeepMind MuJoCo Menagerie ALOHA Assets - 2026-07-17

## Scope

This note records what the Google DeepMind MuJoCo Menagerie ALOHA assets can and cannot justify for the user's ALOHA1 Isaac Sim work.

Primary sources:

- Local: `external/mujoco_menagerie/aloha/README.md`
- Local: `external/mujoco_menagerie/aloha/aloha.xml`
- Local: `external/mujoco_menagerie/trossen_vx300s/README.md`
- GitHub: `https://github.com/google-deepmind/mujoco_menagerie`
- GitHub README: `https://github.com/google-deepmind/mujoco_menagerie/blob/main/aloha/README.md`
- ALOHA2 project page: `https://aloha-2.github.io/`

## Verified Facts

The Menagerie `aloha/` model is explicitly a simplified MJCF model of bimanual ALOHA 2, not ALOHA1.

The model is derived from the ViperX 300 6DOF MJCF tree and then modified for ALOHA2:

- the default ViperX gripper is replaced by the updated ALOHA2 gripper design;
- gripper actuation is modeled as a position-controlled linear actuator;
- an equality constraint links the two fingers;
- system identification was performed using 11 real trajectories;
- the scene includes an aluminum extrusion frame, wooden table, and four cameras;
- RealSense D405 camera intrinsics are represented.

The local `trossen_vx300s/` model is a simplified single-arm ViperX 300 6DOF MJCF derived from a public URDF. It is not a full ALOHA1 or ALOHA2 bimanual workcell.

## What Can Be Reused

Use Menagerie as a reference for:

- ViperX arm kinematic tree shape;
- ALOHA2-style MJCF organization;
- gripper modeling patterns;
- actuator parameterization and system-identification workflow;
- camera and table/frame scene organization ideas;
- a sanity check for expected bimanual ALOHA-like model structure.

## What Must Not Be Assumed

Do not assume:

- ALOHA2 gripper geometry equals the user's ALOHA1 gripper geometry;
- ALOHA2 gripper rail displacement equals ALOHA1 normalized gripper command;
- Menagerie table, frame, and camera layout equals the user's real workcell;
- Menagerie actuator gains, damping, friction, or armature values match ALOHA1;
- the single-arm `trossen_vx300s` model defines the user's full bimanual ALOHA1;
- filtered Cartesian actuators in Menagerie are system-identified for the user's use case.

## Consequence For ALOHA1 Isaac Work

Menagerie is a valuable comparison source, but it is not a direct drop-in model for the user's ALOHA1 workcell.

For Isaac Sim, its strongest use is as a reference asset and validation checklist. Any ALOHA1 Isaac asset derived from it must separately prove:

- DOF order;
- gripper semantics;
- base placement;
- camera placement;
- collision behavior;
- controller behavior;
- replay behavior against real ALOHA1 data.

