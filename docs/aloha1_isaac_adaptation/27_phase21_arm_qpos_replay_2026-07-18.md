# Phase 21: Native Wrapper Arm-Only Qpos Replay

## Goal

Phase 19 proved that the ALOHA1 native physics wrapper can initialize as an Isaac runtime articulation. Phase 20 proved that the arm DOF limits, efforts, velocities, and mimic-joint handling are structurally usable.

This phase checks the next gate:

```text
real ALOHA1 HDF5 observations/qpos
-> mapping into the native Isaac wrapper arm DOFs
-> Isaac set_joint_positions()
-> Isaac get_joint_positions()
```

The goal is only command/readback consistency for arm joints. This is not yet a dynamic controller, IK, contact, grasp, or full visual replay test.

## Official Isaac Basis

The official NVIDIA Isaac MCP robot setup, physics, and USD guidance was used before changing the replay script.

Relevant principles:

- use the actual runtime articulation prim path, not a guessed visual hierarchy path;
- validate joints and drives before controller work;
- reference USD assets without flattening the original source asset;
- treat mimic and gripper semantics separately from active arm joints.

## Script Change

Script:

```text
aloha_isaac_replay/scripts/replay_aloha_qpos_arm_only.py
```

The replay script previously hard-coded old asset articulation paths:

```text
/World/left/root_joint/root_joint
/World/right/root_joint/root_joint
```

The Phase 19 and Phase 20 native wrapper runtime roots are:

```text
/World/left/root_joint
/World/right/root_joint
```

The script now exposes these as arguments:

```text
--left-prim-path
--right-prim-path
```

The defaults are the validated native wrapper paths.

## Input Data

Episode:

```text
local_rlt_data/raw_from_103/rollouts/key_regions/twist_off_the_bottle_cap/2026-06-17/rl/key_region_0e2f4956f0a64d55acb3fa7363b2fdc4/episode.hdf5
```

Mapping:

```text
configs/aloha/original_stationary_aloha_mapping.yaml
```

Native wrapper assets:

```text
assets/isaac/aloha1_native_physics_wrapper/aloha1_left.usda
assets/isaac/aloha1_native_physics_wrapper/aloha1_right.usda
```

Frame count:

```text
40
```

Gripper:

```text
disabled
```

Reason: Phase 20 still flags the `gripper` DOF as semantically unbounded. Arm replay should be validated before gripper calibration.

## Command

```bash
codex-evidence --name phase21-arm-qpos-replay-native-wrapper-fixed-prim -- \
  .venv_issac/bin/python aloha_isaac_replay/scripts/replay_aloha_qpos_arm_only.py \
  --episode local_rlt_data/raw_from_103/rollouts/key_regions/twist_off_the_bottle_cap/2026-06-17/rl/key_region_0e2f4956f0a64d55acb3fa7363b2fdc4/episode.hdf5 \
  --mapping configs/aloha/original_stationary_aloha_mapping.yaml \
  --output-dir reports/aloha1_isaac_adaptation/phase21_arm_qpos_replay_native_20260718 \
  --left-usd assets/isaac/aloha1_native_physics_wrapper/aloha1_left.usda \
  --right-usd assets/isaac/aloha1_native_physics_wrapper/aloha1_right.usda \
  --max-frames 40
```

Evidence artifact:

```text
.codex/artifacts/20260718-005010_phase21-arm-qpos-replay-native-wrapper-fixed-prim
```

Report files:

```text
reports/aloha1_isaac_adaptation/phase21_arm_qpos_replay_native_20260718/replay_metrics.json
reports/aloha1_isaac_adaptation/phase21_arm_qpos_replay_native_20260718/joint_error.png
reports/aloha1_isaac_adaptation/phase21_arm_qpos_replay_native_20260718/expected_qpos.csv
reports/aloha1_isaac_adaptation/phase21_arm_qpos_replay_native_20260718/readback_qpos.csv
reports/aloha1_isaac_adaptation/phase21_arm_qpos_replay_native_20260718/joint_error.csv
```

## Result

The arm-only replay/readback gate passed:

```text
status = PASS
frames = 40
max_abs_readback_error = 0.0
mean_abs_readback_error = 0.0
gate_max_abs_error_rad = 1e-5
```

Runtime DOF names for each side:

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

Used arm indices:

```text
left_indices = [0, 1, 2, 3, 4, 5]
right_indices = [0, 1, 2, 3, 4, 5]
```

Ignored for this gate:

```text
left/gripper
right/gripper
left/left_finger
left/right_finger
right/left_finger
right/right_finger
```

## Interpretation

This is a stronger gate than only checking that an articulation initializes.

It proves:

1. The native wrapper can be referenced into a fresh Isaac stage.
2. The correct runtime articulation prim paths are `/World/left/root_joint` and `/World/right/root_joint`.
3. The first six runtime DOFs on both sides correspond to the expected arm joints.
4. Real ALOHA1 recorded `observations/qpos` can be written into those arm joints and read back exactly for a short sequence.

It does not prove:

1. Dynamic tracking under a controller.
2. Stable stiffness/damping values.
3. Contact realism.
4. Gripper semantics.
5. Bottle grasp, lift, insertion, or release.
6. Visual mesh completeness.

## Known Warning

The Isaac logs still include unresolved visual reference warnings for some imported visual paths. The arm articulation readback result is valid, but the visual asset quality is not yet final.

The replay script currently records:

```text
video_status = BLOCKED_VISUAL_MESH_IMPORT_HAS_ZERO_MESHES
```

That field is stale for this native-wrapper gate and should not be interpreted as a failure of arm qpos readback. A later phase should split visual replay status from articulation readback status.

## Next Gates

1. Split stale visual status from `replay_aloha_qpos_arm_only.py` metrics.
2. Run the same arm-only readback gate on more episodes and longer frame windows.
3. Add a dynamic tracking gate with simulation stepping and controller targets.
4. Calibrate gripper and finger semantics separately.
5. Only after gripper semantics pass, attempt bottle grasp/release simulation.

