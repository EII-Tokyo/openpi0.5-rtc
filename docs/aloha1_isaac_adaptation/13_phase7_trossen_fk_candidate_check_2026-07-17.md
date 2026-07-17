# Phase 7 - Trossen Scaffold FK Candidate Check - 2026-07-17

## Purpose

This phase checks whether the Phase 6 affine candidates can make the Trossen `stationary_ai` scaffold reproduce the trusted ALOHA1 end-effector trajectory.

It is still an offline validation:

- no real robot command was sent;
- no runtime actor was started;
- no USD stage was saved;
- no controller, gripper, contact, or RL behavior was tested.

## Evidence

- Script: `aloha_isaac_replay/scripts/check_trossen_scaffold_fk_against_aloha1.py`
- Full bounded run artifact: `.codex/artifacts/20260717-235112_phase7-trossen-fk-candidate-check`
- JSON report: `reports/aloha1_isaac_adaptation/phase7_trossen_fk_candidate_check_20260717/trossen_fk_candidate_check.json`
- Markdown report: `reports/aloha1_isaac_adaptation/phase7_trossen_fk_candidate_check_20260717/trossen_fk_candidate_check.md`
- CSV points: `reports/aloha1_isaac_adaptation/phase7_trossen_fk_candidate_check_20260717/trossen_fk_candidate_points.csv`

## Method

The trusted reference side is the original ALOHA1 FK:

- Pinocchio model from `assets/isaac/original_stationary_aloha/generated/puppet_left_vx300s_resolved.urdf`
- Pinocchio model from `assets/isaac/original_stationary_aloha/generated/puppet_right_vx300s_resolved.urdf`
- these URDFs were previously resolved from archived 103 `vx300s` robot descriptions.

The candidate side is the Trossen-backed scaffold:

- USD: `local_eval_assets/aloha1_trossen_backed_scaffold_20260717/aloha1_trossen_backed_scaffold.usda`
- candidate mapping: `reports/aloha1_isaac_adaptation/phase6_affine_candidate_inference_20260717/affine_candidates.json`
- left candidate body: `follower_left_link_6`
- right candidate body: `follower_right_link_6`

The script reports:

1. raw position error;
2. rigid-aligned trajectory-shape error.

Rigid alignment removes an unknown fixed base-frame transform between assets. It does not hide joint sign or offset errors, because wrong joint mapping changes the trajectory shape.

## Gates

```text
real_robot_touched: PASS_FALSE
stage_saved: PASS_FALSE
isaac_runtime_started: PASS
trusted_aloha1_fk_loaded: PASS
trossen_scaffold_fk_loaded: PASS
candidate_mapping_complete: BLOCKED_1_FAIL_7_AMBIGUOUS
fk_position_shape: BLOCKED_MAPPING_CANDIDATES_INCOMPLETE
orientation: BLOCKED_FRAME_ALIGNMENT_NOT_ESTABLISHED
controller: BLOCKED_NOT_ATTEMPTED
gripper: BLOCKED_NOT_ATTEMPTED
```

The important result is that FK code can run on both sides, but the mapping is not allowed to pass because Phase 6 is still incomplete.

## Candidate Completeness

```text
unique: 4
ambiguous: 7
failed: 1
```

Failed candidate:

```text
left_forearm_roll
```

Ambiguous candidates:

```text
left_waist
left_wrist_angle
left_wrist_rotate
right_waist
right_forearm_roll
right_wrist_angle
right_wrist_rotate
```

## FK Diagnostic Numbers

| side | raw RMSE m | raw max m | rigid-aligned RMSE m | rigid-aligned max m |
|---|---:|---:|---:|---:|
| left | 0.282599 | 0.312110 | 0.032487 | 0.067740 |
| right | 0.225826 | 0.228700 | 0.001709 | 0.003506 |

Interpretation:

- Raw RMSE is large on both sides because the asset base frames are not yet calibrated to each other.
- Rigid-aligned right-arm trajectory shape is close under the current candidate mapping.
- Rigid-aligned left-arm trajectory shape is still poor, around 3.2 cm RMSE and 6.8 cm max error.
- This matches the Phase 6 warning that `left_forearm_roll` cannot be explained by the current limit-fit candidate.

## Decision

Do not proceed to controller, gripper, grasp, or RL with this mapping.

The next implementation step should focus on resolving left-arm mapping, especially `left_forearm_roll`, using stronger evidence than limit fitting:

1. matched real reference poses if available;
2. real one-joint positive-direction evidence from 103, read-only planned first;
3. wrap-aware handling for roll joints;
4. FK shape optimization over sign/offset candidates, with a strict holdout trajectory.

## Status

```text
BLOCKED_FOR_CONTROL
```

The Trossen scaffold remains a good Isaac runtime base, but the ALOHA1 adapter mapping is not yet verified.
