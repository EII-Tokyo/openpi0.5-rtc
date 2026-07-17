# Phase 8 - FK Mapping Candidate Search - 2026-07-17

## Purpose

Phase 7 showed that the current Phase 6 mapping is not sufficient for the left arm:

```text
left rigid-aligned RMSE = 0.032487 m
left rigid-aligned max  = 0.067740 m
```

The largest known issue is `left_forearm_roll`. The Trossen scaffold runtime limit for the corresponding joint is only:

```text
[-1.570796, 1.570796]
```

But the local ALOHA1 HDF5 qpos range for `left_forearm_roll` reaches beyond that range under the Phase 6 sleep-to-zero assumption.

This phase searches a small discrete set of left-arm sign and offset candidates using FK trajectory shape.

## Evidence

- Script: `aloha_isaac_replay/scripts/search_trossen_fk_mapping_candidates.py`
- Full bounded run artifact: `.codex/artifacts/20260717-235347_phase8-fk-mapping-search`
- JSON report: `reports/aloha1_isaac_adaptation/phase8_fk_mapping_search_20260717/fk_mapping_search.json`
- Markdown report: `reports/aloha1_isaac_adaptation/phase8_fk_mapping_search_20260717/fk_mapping_search.md`

## Scope

This is still diagnostic only:

- no real robot command;
- no stage save;
- no controller validation;
- no gripper/contact validation;
- no final mapping approval.

## Search Space

Fixed from Phase 6:

```text
left_shoulder
left_elbow
```

Searched:

```text
left_waist
left_forearm_roll
left_wrist_angle
left_wrist_rotate
```

Total combinations:

```text
48
```

## Best Result

Best left-arm diagnostic candidate:

```text
left_waist:        sign = -1, offset = 0.000
left_shoulder:     sign =  1, offset = 1.850
left_elbow:        sign = -1, offset = 1.550
left_forearm_roll: sign = -1, offset = 0.126
left_wrist_angle:  sign = -1, offset = 0.800
left_wrist_rotate: sign = -1, offset = 0.000
```

FK shape score:

```text
rigid-aligned RMSE = 0.024616 m
rigid-aligned max  = 0.051196 m
```

This improves the Phase 7 left-arm RMSE from about 3.25 cm to about 2.46 cm.

## Interpretation

The new evidence suggests the Phase 6 `left_forearm_roll` assumption is likely wrong.

The best candidate found here uses:

```text
q_isaac = -q_aloha + 0.126
```

for `left_forearm_roll` on the sampled trajectory.

However, this is not final:

1. The best RMSE is still centimeters, not millimeters.
2. Several different sign combinations produce nearly identical scores.
3. This was tested on one sampled trajectory, not a holdout set.
4. Orientation was not validated.
5. Physical positive-direction evidence is still missing.

## Decision

Do not move to controller execution.

The next valid step is a holdout FK search across multiple trajectories and sides:

1. split local HDF5 trajectories into search and holdout;
2. search sign and offset candidates on the search set;
3. score the selected candidate on holdout;
4. only then decide whether a real one-joint positive-direction test plan is needed on 103.

## Status

```text
DIAGNOSTIC_PROGRESS_NOT_CONTROLLER_READY
```
