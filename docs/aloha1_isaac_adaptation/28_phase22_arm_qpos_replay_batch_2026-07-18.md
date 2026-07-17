# Phase 22: Native Wrapper Arm-Only Qpos Replay Batch

## Goal

Phase 21 validated one short real ALOHA1 HDF5 segment against the native ALOHA1 Isaac wrapper. This phase expands that gate to multiple episodes in one Isaac runtime session.

This still validates only deterministic arm-joint command/readback consistency:

```text
HDF5 observations/qpos
-> arm-only mapping
-> Isaac set_joint_positions()
-> Isaac get_joint_positions()
```

It does not validate dynamic tracking, controller gains, contact, gripper semantics, or bottle manipulation.

## Script

```text
aloha_isaac_replay/scripts/validate_aloha1_native_arm_qpos_replay_batch.py
```

The script:

1. discovers valid `episode.hdf5` files under a bounded HDF5 root;
2. loads `observations/qpos` with shape `(T, 14)`;
3. starts one Isaac SimulationApp;
4. references the Phase 19 native wrapper left and right assets;
5. writes the first six arm DOFs for both arms;
6. reads the same DOFs back from Isaac;
7. reports per-episode and aggregate readback error.

## Inputs

HDF5 root:

```text
local_rlt_data/raw_from_103/rollouts/key_regions/twist_off_the_bottle_cap/2026-06-17/rl
```

Mapping:

```text
configs/aloha/original_stationary_aloha_mapping.yaml
```

Assets:

```text
assets/isaac/aloha1_native_physics_wrapper/aloha1_left.usda
assets/isaac/aloha1_native_physics_wrapper/aloha1_right.usda
```

Articulation roots:

```text
/World/left/root_joint
/World/right/root_joint
```

Batch size:

```text
6 episodes
80 frames per episode
480 total frames
```

## Command

```bash
codex-evidence --name phase22-arm-qpos-replay-batch-native-wrapper -- \
  .venv_issac/bin/python aloha_isaac_replay/scripts/validate_aloha1_native_arm_qpos_replay_batch.py \
  --episode-limit 6 \
  --max-frames-per-episode 80
```

Evidence artifact:

```text
.codex/artifacts/20260718-005247_phase22-arm-qpos-replay-batch-native-wrapper
```

Report:

```text
reports/aloha1_isaac_adaptation/phase22_arm_qpos_replay_batch_20260718/batch_replay_metrics.json
reports/aloha1_isaac_adaptation/phase22_arm_qpos_replay_batch_20260718/batch_replay_metrics.md
```

## Result

```text
status = PASS
overall_pass = true
episodes_tested = 6
total_frames = 480
max_abs_readback_error = 0.0
mean_abs_readback_error = 0.0
gate_max_abs_error_rad = 1e-5
```

Used arm indices:

```text
left_indices = [0, 1, 2, 3, 4, 5]
right_indices = [0, 1, 2, 3, 4, 5]
```

Per-episode result:

| idx | frames | status | max abs error |
| ---: | ---: | --- | ---: |
| 1 | 80 | PASS | 0.0 |
| 2 | 80 | PASS | 0.0 |
| 3 | 80 | PASS | 0.0 |
| 4 | 80 | PASS | 0.0 |
| 5 | 80 | PASS | 0.0 |
| 6 | 80 | PASS | 0.0 |

## Interpretation

This makes the native wrapper more credible than the earlier Trossen-backed scaffold for ALOHA1 arm-state replay.

It proves:

1. the native wrapper can be loaded once and reused for multiple real HDF5 episodes;
2. the runtime arm DOF names and indices remain stable across the batch;
3. the central ALOHA mapping can drive both left and right arm DOFs without readback error;
4. the result is not a single-episode accident.

It still does not prove:

1. physical tracking under simulation stepping;
2. stable drive tuning;
3. gripper/finger semantics;
4. visual mesh completeness;
5. collision or contact correctness;
6. bottle grasp or insertion behavior.

## Next Gates

1. Add a dynamic tracking gate that steps simulation after setting controller targets.
2. Measure tracking error instead of direct set/readback error.
3. Keep gripper disabled until the unbounded `gripper` DOF is calibrated.
4. Fix unresolved visual references as a separate asset-quality task.

