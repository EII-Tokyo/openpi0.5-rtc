# Phase 11 - Orientation Consistency Check - 2026-07-18

## Purpose

Phase 9 and Phase 10 made the left-arm mapping candidate look plausible for position trajectory shape and joint limits. Phase 11 checks a stricter condition:

If the Trossen candidate link and the trusted ALOHA1 end-effector frame represent the same physical rigid body, their orientation difference should be approximately a fixed rotation over time.

## Evidence

- Script: `aloha_isaac_replay/scripts/validate_trossen_mapping_orientation_consistency.py`
- Full bounded run artifact: `.codex/artifacts/20260718-000141_phase11-orientation-consistency`
- JSON report: `reports/aloha1_isaac_adaptation/phase11_orientation_consistency_20260718/orientation_consistency.json`
- Markdown report: `reports/aloha1_isaac_adaptation/phase11_orientation_consistency_20260718/orientation_consistency.md`

## Scope

This phase is offline only:

- no real robot command;
- no stage save;
- no controller execution;
- no gripper/contact validation.

## Method

The test compares:

- trusted ALOHA1 FK orientation from the generated VX300S URDF;
- Trossen scaffold body orientation from `follower_left_link_6`.

The first frame is used to estimate a fixed orientation offset. Then the script checks how much the relative orientation changes over sampled qpos frames.

If the mapping and link frame are correct, the residual should stay small.

## Dataset

```text
valid episodes: 16
sampled frames: 128
candidate Trossen body: follower_left_link_6
```

## Gates

```text
real_robot_touched: PASS_FALSE
stage_saved: PASS_FALSE
isaac_runtime_started: PASS
qpos_loaded: PASS
orientation_consistency: FAIL_ORIENTATION_INCONSISTENT
controller: BLOCKED_NOT_ATTEMPTED
```

## Orientation Residuals

```text
mean residual: 14.509027 deg
p95 residual:  41.103683 deg
max residual:  42.633401 deg
```

Threshold used by the diagnostic:

```text
p95 <= 5 deg
max <= 10 deg
```

## Interpretation

This is a hard warning.

The Phase 9 mapping candidate passes position-shape and full-dataset limit checks, but it fails orientation consistency. That means the current candidate should not be used for controller work.

Possible causes:

1. `follower_left_link_6` is not the correct Trossen frame to compare against the ALOHA1 `ee_gripper_link`;
2. one or more wrist-related joint signs are still wrong;
3. one or more wrist offsets are still wrong;
4. the Trossen frame and ALOHA1 frame differ by more than a fixed terminal transform because the link definitions are not semantically equivalent.

## Decision

Do not proceed to controller execution.

The next gate should search or inspect the correct Trossen terminal frame:

- compare `follower_left_link_6`;
- compare `follower_left_camera_mount_d405`;
- compare gripper carriage/finger bodies only if their rigid-body semantics are known;
- inspect Trossen USD joint body relationships before selecting a frame.

## Status

```text
BLOCKED_BY_ORIENTATION
```
