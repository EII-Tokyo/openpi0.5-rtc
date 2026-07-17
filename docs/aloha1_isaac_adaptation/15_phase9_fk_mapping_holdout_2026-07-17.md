# Phase 9 - FK Mapping Holdout Validation - 2026-07-17

## Purpose

Phase 8 found a better left-arm FK candidate, but it used one sampled trajectory. Phase 9 adds a simple search/holdout split so the selected mapping must generalize to trajectories that were not used during search.

## Evidence

- Script: `aloha_isaac_replay/scripts/validate_trossen_fk_mapping_holdout.py`
- Full bounded run artifact: `.codex/artifacts/20260717-235544_phase9-fk-mapping-holdout`
- JSON report: `reports/aloha1_isaac_adaptation/phase9_fk_mapping_holdout_20260717/fk_mapping_holdout.json`
- Markdown report: `reports/aloha1_isaac_adaptation/phase9_fk_mapping_holdout_20260717/fk_mapping_holdout.md`

## Scope

This phase is still offline only:

- no real robot command;
- no stage save;
- no controller execution;
- no gripper/contact validation.

## Dataset

```text
search episodes: 8
holdout episodes: 4
search frames: 64
holdout frames: 32
combinations tested: 48
```

## Gates

```text
real_robot_touched: PASS_FALSE
stage_saved: PASS_FALSE
isaac_runtime_started: PASS
search_executed: PASS
holdout_executed: PASS
holdout_fk_shape: PASS_DIAGNOSTIC
controller: BLOCKED_NOT_ATTEMPTED
```

## Best Candidate

```text
left_waist:        sign = -1, offset = 0.000000
left_shoulder:     sign =  1, offset = 1.850000
left_elbow:        sign = -1, offset = 1.550000
left_forearm_roll: sign =  1, offset = -1.498699
left_wrist_angle:  sign =  1, offset = -0.800000
left_wrist_rotate: sign =  1, offset = 0.000000
```

The new `left_forearm_roll` candidate is approximately:

```text
q_isaac = q_aloha - 1.499
```

This is close to subtracting half pi, rather than mapping the ALOHA1 sleep pose to zero.

## FK Shape Scores

| set | rigid-aligned RMSE m | rigid-aligned max m | raw RMSE m |
|---|---:|---:|---:|
| search | 0.019271 | 0.039098 | 0.531197 |
| holdout | 0.019781 | 0.047104 | 0.504502 |

## Interpretation

The holdout result is stronger than Phase 8 because the best candidate keeps a similar FK trajectory-shape error on unseen sampled episodes.

The result suggests:

1. the Trossen scaffold can be used as the Isaac runtime base;
2. the left-arm mapping is likely not the sleep-to-zero mapping;
3. `left_forearm_roll` probably needs a roughly half-pi offset;
4. base-frame translation between assets remains uncalibrated, which explains the large raw RMSE.

This still does not mean the mapping is ready for control:

- orientation is not validated;
- gripper carriage mapping is not validated;
- real one-joint positive direction is not validated;
- the search used sparse sampled frames;
- the candidate should be checked on more diverse trajectories before a controller is attempted.

## Next Step

Use this candidate as a hypothesis, not as a final adapter.

The next gate should validate:

1. orientation trajectory;
2. full-dataset joint-limit validity;
3. right-arm equivalent holdout;
4. real one-joint positive-direction plan on 103, with no motion until the plan is reviewed.

## Status

```text
DIAGNOSTIC_CANDIDATE_FOUND_NOT_CONTROLLER_READY
```
