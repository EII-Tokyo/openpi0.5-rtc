# Phase 6 Affine Candidate Inference - 2026-07-17

## Result

Phase 6 generated conservative offline `sign/offset` candidates from local HDF5
`observations/qpos`.

This did **not** solve the ALOHA1 to Trossen mapping. It only established a
limit-fit baseline and showed that limits alone are insufficient.

The most important result:

```text
valid episodes = 248
frames used = 42756
unique limit-fit candidates = 4
ambiguous candidates = 7
failed candidates = 1
```

The failed row is important:

```text
left_forearm_roll = FAIL_NO_SIGN_FITS_LIMITS
```

This means the simple reference assumption is not reliable enough:

```text
ALOHA1 puppet sleep pose maps to Trossen scaffold zero pose
```

## Evidence

- Script: `aloha_isaac_replay/scripts/infer_trossen_aloha1_affine_candidates.py`
- Report: `reports/aloha1_isaac_adaptation/phase6_affine_candidate_inference_20260717/affine_candidates.md`
- JSON: `reports/aloha1_isaac_adaptation/phase6_affine_candidate_inference_20260717/affine_candidates.json`
- CSV: `reports/aloha1_isaac_adaptation/phase6_affine_candidate_inference_20260717/affine_candidates.csv`

The analysis was offline only:

```text
real_robot_touched = false
isaac_runtime_started = false
```

## Candidate Model

The candidate form is:

```text
q_isaac = sign * q_aloha + offset
sign in {+1, -1}
```

The script sets `offset` by assuming ALOHA1 puppet sleep maps to Trossen
scaffold zero, then checks whether all sampled HDF5 qpos values stay inside the
Trossen runtime joint limits.

This is only a filter. It is not a proof.

## Gates

```text
real_robot_touched = PASS_FALSE
isaac_runtime_started = PASS_FALSE
hdf5_qpos_loaded = PASS
limit_fit_candidates_generated = PASS
mapping_candidates_complete = BLOCKED_1_FAIL_7_AMBIGUOUS
sign = BLOCKED_LIMIT_FIT_IS_NOT_POSITIVE_DIRECTION_EVIDENCE
offset = BLOCKED_REFERENCE_ASSUMPTION_NOT_GEOMETRIC_PROOF
fk = BLOCKED_REQUIRES_TRUSTED_FK_OR_REFERENCE_POSES
```

## Candidate Table

| joint | selected sign | selected offset | inside fraction | min margin | status |
|---|---:|---:|---:|---:|---|
| `left_waist` | 1 | 0.000000 | 1.0000 | 2.261258 | `AMBIGUOUS_BOTH_SIGNS_FIT_LIMITS` |
| `left_shoulder` | 1 | 1.850000 | 1.0000 | 0.826797 | `PASS_LIMIT_FIT_UNIQUE_CANDIDATE` |
| `left_elbow` | -1 | 1.550000 | 1.0000 | 0.368835 | `PASS_LIMIT_FIT_UNIQUE_CANDIDATE` |
| `left_forearm_roll` | 1 | 0.000000 | 0.2386 | -0.607456 | `FAIL_NO_SIGN_FITS_LIMITS` |
| `left_wrist_angle` | 1 | -0.800000 | 1.0000 | 0.391903 | `AMBIGUOUS_BOTH_SIGNS_FIT_LIMITS` |
| `left_wrist_rotate` | 1 | 0.000000 | 1.0000 | 1.560058 | `AMBIGUOUS_BOTH_SIGNS_FIT_LIMITS` |
| `right_waist` | 1 | 0.000000 | 1.0000 | 2.081782 | `AMBIGUOUS_BOTH_SIGNS_FIT_LIMITS` |
| `right_shoulder` | 1 | 1.850000 | 1.0000 | 0.886660 | `PASS_LIMIT_FIT_UNIQUE_CANDIDATE` |
| `right_elbow` | -1 | 1.550000 | 1.0000 | 0.374971 | `PASS_LIMIT_FIT_UNIQUE_CANDIDATE` |
| `right_forearm_roll` | 1 | 0.000000 | 1.0000 | 0.949534 | `AMBIGUOUS_BOTH_SIGNS_FIT_LIMITS` |
| `right_wrist_angle` | 1 | -0.800000 | 1.0000 | 0.749321 | `AMBIGUOUS_BOTH_SIGNS_FIT_LIMITS` |
| `right_wrist_rotate` | 1 | 0.000000 | 1.0000 | 2.391476 | `AMBIGUOUS_BOTH_SIGNS_FIT_LIMITS` |

## Interpretation

The four unique rows are useful, but weak:

```text
left_shoulder
left_elbow
right_shoulder
right_elbow
```

They only say that one sign fits the observed HDF5 range under the reference
assumption. They do not prove that the Isaac positive direction matches the real
robot positive direction.

The seven ambiguous rows are expected for joints whose observed HDF5 motion
range is small compared with the Trossen runtime limit. Limits alone cannot
choose their sign.

The failed `left_forearm_roll` row means the reference-pose assumption is
probably wrong for at least one joint, or the Trossen scaffold joint axis/zero
semantics differ from ALOHA1 enough that a sleep-to-zero mapping is invalid.

## Decision

Do not use this table directly as a controller mapping.

It is only a candidate filter. The adapter remains blocked until geometric
evidence is added.

## Next Gate

The next gate must add at least one of:

1. trusted ALOHA1 FK using the real URDF/MJCF chain and the same qpos samples;
2. matched real reference poses with measured or visually inferred end-effector
   poses;
3. a separately reviewed real one-joint positive-direction test plan.

The preferred next action is still offline:

```text
build an arm-only FK comparison harness that compares candidate mappings against a trusted ALOHA1 FK chain
```

Real hardware one-joint motion remains a separate safety-reviewed plan, not the
default next step.
