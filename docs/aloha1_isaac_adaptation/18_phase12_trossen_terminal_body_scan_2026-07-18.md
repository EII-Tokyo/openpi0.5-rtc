# Phase 12 - Trossen Terminal Body Scan - 2026-07-18

## Purpose

Phase 11 failed orientation consistency when comparing ALOHA1 `ee_gripper_link` against Trossen `follower_left_link_6`.

Phase 12 checks whether the failure is simply caused by choosing the wrong Trossen terminal body.

## Evidence

- Script: `aloha_isaac_replay/scripts/scan_trossen_left_terminal_body_candidates.py`
- Full bounded run artifact: `.codex/artifacts/20260718-000347_phase12-terminal-body-scan`
- JSON report: `reports/aloha1_isaac_adaptation/phase12_trossen_terminal_body_scan_20260718/terminal_body_scan.json`
- Markdown report: `reports/aloha1_isaac_adaptation/phase12_trossen_terminal_body_scan_20260718/terminal_body_scan.md`

## Scope

This is offline only:

- no real robot command;
- no stage save;
- no controller execution;
- no gripper/contact validation.

## Dataset

```text
valid episodes: 16
sampled frames: 128
scanned bodies: 8
```

## Gates

```text
real_robot_touched: PASS_FALSE
stage_saved: PASS_FALSE
isaac_runtime_started: PASS
body_scan_executed: PASS
best_orientation_consistency: FAIL_NO_BODY_WITH_STABLE_ORIENTATION
controller: BLOCKED_NOT_ATTEMPTED
```

## Body Scan Result

| rank | body | position RMSE m | orientation p95 deg | orientation max deg |
|---:|---|---:|---:|---:|
| 1 | `follower_left_link_5` | 0.024403 | 40.412576 | 46.065286 |
| 2 | `follower_left_link_6` | 0.023296 | 41.103683 | 42.633401 |
| 3 | `follower_left_gripper_right` | 0.036121 | 41.103689 | 42.633416 |
| 4 | `follower_left_camera_mount_d405` | 0.024443 | 41.103690 | 42.633418 |
| 5 | `follower_left_gripper_left` | 0.033579 | 41.103693 | 42.633418 |
| 6 | `follower_left_carriage_right` | 0.036121 | 41.103693 | 42.633407 |
| 7 | `follower_left_carriage_left` | 0.033579 | 41.103696 | 42.633408 |
| 8 | `follower_left_camera_link` | 0.033442 | 44.588526 | 45.734174 |

## Interpretation

This rules out the simplest explanation that Phase 11 failed only because `follower_left_link_6` was the wrong terminal body.

Every scanned body has orientation residual around 40 degrees p95. Therefore, the remaining likely causes are:

1. wrist or forearm joint sign is still wrong;
2. wrist or forearm joint zero offset is still wrong;
3. Trossen and ALOHA1 joint axes are not semantically equivalent for the terminal chain;
4. ALOHA1 `ee_gripper_link` and Trossen terminal frames differ in a way that is not captured by a fixed rigid transform after the current mapping.

## Decision

Do not proceed to controller work.

The next gate should inspect joint axis semantics directly:

- compare ALOHA1 URDF joint axes and origins against Trossen USD joint axes and body relationships;
- prioritize `forearm_roll`, `wrist_angle`, and `wrist_rotate`;
- only after this should another FK candidate be tested.

## Status

```text
BLOCKED_BY_JOINT_AXIS_OR_OFFSET_SEMANTICS
```
