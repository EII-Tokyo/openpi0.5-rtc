# ALOHA1 Isaac Adaptation Execution Plan - 2026-07-17

## Goal

Use Trossen AI Isaac as a scaffold while preserving ALOHA1 truth. The target is a simulation stack that can replay real ALOHA1 data, execute bottle grasps, interact with the pipe, and eventually support reward/RL experiments.

## Principle

Do not jump directly from a visible mesh to RL. The work must advance through gates.

Each gate has:

- source evidence;
- a script or report;
- explicit pass/fail criteria;
- a stop condition.

## Phase 0 - Source Lock

Record exact sources:

- chosen ALOHA1 USD or URDF;
- chosen Trossen AI Isaac USD or scripts;
- chosen Menagerie/MuJoCo source;
- current Isaac Sim version;
- current project adapter code.

Pass condition:

- every source has a path, commit, or URL.

Stop condition:

- source identity is ambiguous.

## Phase 1 - Read-Only Asset Comparison

Compare, without editing:

- ALOHA1 project asset;
- Trossen Stationary AI asset;
- Menagerie ALOHA/Aloha2 source.

Collect:

- prim tree;
- articulation root;
- DOF names;
- DOF limits;
- link names;
- end-effector candidates;
- gripper joints;
- collider count;
- visual mesh count;
- material assignments;
- unit system and up axis.

Pass condition:

- machine-readable comparison report exists.

Stop condition:

- selected asset cannot expose DOF names/order through Isaac or USD tooling.

## Phase 2 - ALOHA1 Canonical Mapping

Build a mapping table:

```text
ALOHA1 canonical 14D field
-> real ROS/Interbotix joint
-> Isaac DOF
-> sign
-> offset
-> unit
-> source
```

Pass condition:

- every dimension has a source and a confidence level.

Stop condition:

- any arm DOF lacks a reliable identity.

## Phase 3 - One-Joint Validation

Move one simulated joint at a time in Isaac. Compare:

- expected direction;
- readback direction;
- limit behavior;
- visual motion;
- FK effect.

Do not include gripper in this phase.

Pass condition:

- all arm joints pass direction and readback checks.

Stop condition:

- sign/offset mismatch cannot be explained.

## Phase 4 - Gripper Validation

Validate gripper separately:

- ALOHA1 normalized qpos;
- real gripper joint command;
- Isaac gripper DOF;
- finger opening distance;
- open/close direction;
- mimic behavior.

Pass condition:

- open, close, and midpoint are numerically and visually consistent.

Stop condition:

- gripper command value cannot be mapped to physical opening.

## Phase 5 - FK and EE Frame Validation

Under the same canonical qpos, compare:

- left end-effector pose;
- right end-effector pose;
- finger opening;
- bottle grasp frame relation.

Pass condition:

- frame definitions are stable and documented.

Stop condition:

- gripper frame is ambiguous or changes across assets.

## Phase 6 - Camera Validation

Validate cameras:

- `cam_high`;
- `cam_low`;
- `cam_left_wrist`;
- `cam_right_wrist`.

Use recorded HDF5 frames, known workcell geometry, or calibration targets.

Pass condition:

- camera projection is good enough to explain bottle and pipe visibility.

Stop condition:

- `cam_low` or right wrist camera cannot be aligned to real data.

## Phase 7 - Contact and Physics Validation

Validate:

- robot self-collision;
- gripper/bottle contact;
- bottle/table contact;
- bottle/pipe contact;
- joint drive stability;
- friction and mass plausibility.

Pass condition:

- bottle can be grasped and lifted in simulation without obvious nonphysical behavior.

Stop condition:

- collision meshes or drive gains are not usable.

## Phase 8 - HDF5 Replay

Replay real HDF5:

- qpos-only;
- action-only;
- corrected action;
- gripper included;
- bottle pose if available or estimated.

Pass condition:

- replay produces the expected arm trajectory, gripper behavior, and camera consistency.

Stop condition:

- replay only works by ignoring gripper or contact.

## Phase 9 - Bottle-Pipe Task Environment

Only after the above gates:

- add pipe geometry;
- add bottle variants;
- define success/failure geometry;
- define reward candidates;
- evaluate real successful and failed HDF5 examples.

Pass condition:

- simulation can reproduce qualitative differences between success and failure trajectories.

Stop condition:

- simulation cannot distinguish insertion, near miss, and far miss.

## Phase 10 - RL / Isaac Lab

Build Isaac Lab task only after replay and contact validation.

Candidate scaffolds:

- Trossen WXAI reach/lift task structure;
- Trossen standalone Stationary AI controller;
- custom ALOHA1 canonical action adapter.

Pass condition:

- task reset, observation, action, reward, termination, and logging all use ALOHA1-validated semantics.

Stop condition:

- task relies on Trossen/WXAI assumptions that were not validated for ALOHA1.

## Immediate Next Step After Phase 2

Phase 1 and Phase 2 are complete enough to change direction.

The next actionable step is no longer to repair the current generated ALOHA1 USD. It is to build a Trossen-backed ALOHA1 scaffold contract:

- use Trossen `stationary_ai` as the Isaac runtime structure standard;
- keep the current generated ALOHA1 USD only as a kinematic naming/reference artifact;
- emit an ALOHA1 adapter table whose uncertain fields are explicitly marked `REQUIRES_REAL_DATA_VERIFICATION`;
- block controller, contact, camera, gripper, replay, and RL work until the scaffold and one-joint validation gates pass.

See [07 Phase 2 runtime decision](07_phase2_runtime_decision_2026-07-17.md).
