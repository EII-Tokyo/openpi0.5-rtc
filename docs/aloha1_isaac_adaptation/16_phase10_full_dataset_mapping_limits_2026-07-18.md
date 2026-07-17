# Phase 10 - Full-Dataset Mapping Limit Check - 2026-07-18

## Purpose

Phase 9 produced a left-arm FK mapping candidate that generalized on a small holdout split. Phase 10 checks whether that candidate stays inside the Trossen runtime joint limits over all currently local ALOHA1 HDF5 key-region qpos.

This is a necessary safety filter before any controller work.

## Evidence

- Script: `aloha_isaac_replay/scripts/validate_trossen_mapping_limits_full_dataset.py`
- JSON report: `reports/aloha1_isaac_adaptation/phase10_mapping_full_dataset_limits_20260718/mapping_full_dataset_limits.json`
- Markdown report: `reports/aloha1_isaac_adaptation/phase10_mapping_full_dataset_limits_20260718/mapping_full_dataset_limits.md`
- CSV report: `reports/aloha1_isaac_adaptation/phase10_mapping_full_dataset_limits_20260718/mapping_full_dataset_limits.csv`

## Scope

This phase is pure offline data validation:

- no Isaac Sim runtime started;
- no real robot command;
- no stage save;
- no controller validation.

## Dataset

```text
HDF5 root: local_rlt_data/raw_from_103/rollouts/key_regions
valid episodes: 248
frames: 42756
```

## Gates

```text
real_robot_touched: PASS_FALSE
isaac_runtime_started: PASS_FALSE
qpos_loaded: PASS
all_mapped_values_inside_trossen_limits: PASS
controller: BLOCKED_NOT_ATTEMPTED
```

## Candidate Checked

```text
left_waist:        sign = -1, offset = 0.000000
left_shoulder:     sign =  1, offset = 1.850000
left_elbow:        sign = -1, offset = 1.550000
left_forearm_roll: sign =  1, offset = -1.498699
left_wrist_angle:  sign =  1, offset = -0.800000
left_wrist_rotate: sign =  1, offset = 0.000000
```

## Full-Dataset Limit Results

| joint | mapped range | Trossen limit | inside fraction | outside count | min margin |
|---|---|---|---:|---:|---:|
| `left_waist` | [-0.793068, 0.535359] | [-3.054326, 3.054326] | 1.000000 | 0 | 2.261258 |
| `left_shoulder` | [0.885126, 2.314796] | [0.000000, 3.141593] | 1.000000 | 0 | 0.826797 |
| `left_elbow` | [0.368835, 1.451825] | [0.000000, 2.356194] | 1.000000 | 0 | 0.368835 |
| `left_forearm_roll` | [-1.214913, 0.679553] | [-1.570796, 1.570796] | 1.000000 | 0 | 0.355884 |
| `left_wrist_angle` | [-1.178893, 0.557573] | [-1.570796, 1.570796] | 1.000000 | 0 | 0.391903 |
| `left_wrist_rotate` | [-1.581534, 0.536893] | [-3.141593, 3.141593] | 1.000000 | 0 | 1.560058 |

## Interpretation

The Phase 9 left-arm candidate passes this necessary limit check across all currently local HDF5 qpos.

The tightest joint is `left_forearm_roll`, with minimum margin:

```text
0.355884 rad
```

This supports continuing validation of this candidate.

It still does not validate:

- end-effector orientation;
- right-arm mapping;
- gripper carriage and physical opening;
- controller stability;
- contact dynamics;
- real positive-direction semantics on 103.

## Status

```text
PASS_LIMIT_FILTER_NOT_CONTROLLER_READY
```
