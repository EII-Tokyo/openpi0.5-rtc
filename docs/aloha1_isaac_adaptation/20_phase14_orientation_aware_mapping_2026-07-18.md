# Phase 14 - Orientation-Aware Mapping Search - 2026-07-18

## Purpose

Phase 13 showed that the Trossen scaffold left-arm joint schema is not a simple one-to-one replacement for ALOHA1, especially around the elbow, forearm roll, and wrist angle.

Phase 14 tests whether a broader sign/offset search can repair the orientation problem while still matching the end-effector path.

## Evidence

- Script: `aloha_isaac_replay/scripts/search_trossen_orientation_aware_mapping.py`
- Full bounded run artifact: `.codex/artifacts/20260718-001403_phase14-orientation-aware-mapping`
- JSON report: `reports/aloha1_isaac_adaptation/phase14_orientation_aware_mapping_20260718/orientation_aware_mapping.json`
- Markdown report: `reports/aloha1_isaac_adaptation/phase14_orientation_aware_mapping_20260718/orientation_aware_mapping.md`

## Scope

This is an offline diagnostic:

- no real robot command;
- no stage save;
- no controller execution;
- no gripper/contact validation.

## Search Setup

```text
search episodes: 8
holdout episodes: 4
search frames: 64
holdout frames: 32
candidate combinations: 256
orientation weight: 0.002 m per degree
```

The search score is:

```text
composite = position_rmse_m + 0.002 * orientation_p95_deg
```

This makes a 5 degree orientation error count roughly like 1 cm of position error.

## Gates

```text
real_robot_touched: PASS_FALSE
stage_saved: PASS_FALSE
isaac_runtime_started: PASS
search_executed: PASS
holdout_executed: PASS
holdout_position: PASS_DIAGNOSTIC
holdout_orientation: FAIL_ORIENTATION
controller: BLOCKED_NOT_ATTEMPTED
```

## Best Candidate

| split | composite | position RMSE m | position max m | orientation p95 deg | orientation max deg |
|---|---:|---:|---:|---:|---:|
| search | 0.075194 | 0.022327 | 0.053011 | 26.433765 | 28.619549 |
| holdout | 0.088355 | 0.022924 | 0.058872 | 32.715272 | 43.596137 |

Best candidate mapping:

```text
left_waist        sign -1  offset  0.000000
left_shoulder     sign  1  offset  1.850000
left_elbow        sign  1  offset -1.550000
left_forearm_roll sign -1  offset  2.246259
left_wrist_angle  sign  1  offset -0.800000
left_wrist_rotate sign -1  offset  0.000000
```

## Full-Dataset Limit Check For The Best Candidate

The best orientation-aware candidate is not controller-usable because it violates Trossen runtime limits on the full local HDF5 dataset:

```text
episodes: 248
frames: 42756
```

| joint | mapped range | Trossen limit | outside frames | inside fraction |
|---|---|---|---:|---:|
| `left_waist` | [-0.793068, 0.535359] | [-3.054326, 3.054326] | 0 | 1.000000 |
| `left_shoulder` | [0.885126, 2.314796] | [0.000000, 3.141593] | 0 | 1.000000 |
| `left_elbow` | [-1.451825, -0.368835] | [0.000000, 2.356194] | 42756 | 0.000000 |
| `left_forearm_roll` | [0.068007, 1.962473] | [-1.570796, 1.570796] | 281 | 0.993428 |
| `left_wrist_angle` | [-1.178893, 0.557573] | [-1.570796, 1.570796] | 0 | 1.000000 |
| `left_wrist_rotate` | [-0.536893, 1.581534] | [-3.141593, 3.141593] | 0 | 1.000000 |

## Interpretation

The result is useful because it rules out a tempting shortcut.

The search can reduce orientation p95 from about 40 degrees to about 33 degrees on holdout, but:

1. the orientation error is still far too high for controller work;
2. the best orientation candidate makes `left_elbow` invalid for every full-dataset frame;
3. the remaining forearm/wrist mismatch is likely structural, not just a scalar sign/offset issue.

## Decision

Do not continue with the current Trossen-backed scaffold as a controller target.

The next valid direction is to rebuild the ALOHA1 Isaac asset around the trusted ALOHA1 kinematic semantics, while borrowing only the Trossen framework pieces that are actually reusable:

- asset organization;
- articulation setup pattern;
- physics/drive configuration strategy;
- validation workflow;
- controller scaffolding.

Do not force the ALOHA1 joint chain to fit Trossen `stationary_ai` joint axes.

## Status

```text
BLOCKED_BY_STRUCTURAL_LEFT_ARM_KINEMATIC_MISMATCH
```
