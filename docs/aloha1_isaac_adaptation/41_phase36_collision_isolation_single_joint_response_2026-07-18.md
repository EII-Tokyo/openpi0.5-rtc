# Phase 36: Collision isolation for clean-stage single-joint response

## Question

Phase 35 showed that the clean runtime stage can be opened and initialized, but dynamic one-joint tracking fails badly when collision is enabled.

This phase asks:

Is the failure caused by drive control itself, or by the collision composition in the clean stage?

## Code change

Updated:

```text
aloha_isaac_replay/scripts/validate_aloha1_native_single_joint_response.py
```

The existing `--disable-robot-collisions` path only disabled collision prims under the old `/World/left` and `/World/right` reference-loading paths. That meant it disabled `0` prims for the new clean runtime stage.

The validator now uses whole-stage collision traversal when `--stage-usd` is provided.

## Static comparison

Command:

```bash
codex-evidence --name phase36-clean-side-physics-comparison -- \
  .venv_issac/bin/python \
  aloha_isaac_replay/scripts/compare_aloha1_trossen_physics_properties.py \
  --aloha-left-usd local_eval_assets/aloha1_clean_runtime_20260718/aloha1_left_clean_runtime.usda \
  --aloha-right-usd local_eval_assets/aloha1_clean_runtime_20260718/aloha1_right_clean_runtime.usda \
  --output-dir reports/aloha1_isaac_adaptation/phase36_clean_side_physics_comparison_20260718
```

Evidence:

```text
.codex/artifacts/20260718-020312_phase36-clean-side-physics-comparison
```

The clean side wrappers still have the same important static differences from the known-working Trossen Stationary AI asset:

- ALOHA1 clean wrappers use `meters_per_unit = 0.01`; Trossen uses `1.0`.
- ALOHA1 clean wrappers use `up_axis = Y`; Trossen uses `Z`.
- ALOHA1 arm drives have zero authored damping in static USD inspection.
- ALOHA1 clean wrappers contain collision APIs, but those collisions must still be validated for filtering and shape correctness.

## Dynamic collision isolation

First, a left-waist-only test with collision correctly disabled:

```bash
codex-evidence --name phase36-left-waist-units1-disable-collisions-v2 -- \
  .venv_issac/bin/python \
  aloha_isaac_replay/scripts/validate_aloha1_native_single_joint_response.py \
  --stage-usd local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_clean_runtime.usda \
  --stage-units-in-meters 1.0 \
  --joint left:waist \
  --phase-offset 0 \
  --phase-offset 0.02 \
  --phase-offset 0 \
  --phase-steps 80 \
  --settle-steps 20 \
  --arm-kp 200 \
  --arm-kd 200 \
  --disable-robot-collisions \
  --final-error-tolerance 0.1 \
  --output-dir reports/aloha1_isaac_adaptation/phase36_left_waist_units1_no_collision_v2_20260718
```

Evidence:

```text
.codex/artifacts/20260718-020535_phase36-left-waist-units1-disable-collisions-v2
```

Result:

| Check | Result |
| --- | --- |
| Validator status | PASS |
| Disabled collision prims | 22 |
| Tested joint | left `waist` |
| Max final absolute error | 0.0040 |
| Max absolute error | 0.0197 |
| Limit violations | 0 |

Then, both left and right smoke joints:

```bash
codex-evidence --name phase36-clean-stage-single-joint-no-collision-both -- \
  .venv_issac/bin/python \
  aloha_isaac_replay/scripts/validate_aloha1_native_single_joint_response.py \
  --stage-usd local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_clean_runtime.usda \
  --stage-units-in-meters 1.0 \
  --joint left:waist \
  --joint right:shoulder \
  --phase-offset 0 \
  --phase-offset 0.02 \
  --phase-offset 0 \
  --phase-steps 80 \
  --settle-steps 20 \
  --arm-kp 200 \
  --arm-kd 200 \
  --disable-robot-collisions \
  --final-error-tolerance 0.1 \
  --output-dir reports/aloha1_isaac_adaptation/phase36_clean_stage_single_joint_no_collision_both_20260718
```

Evidence:

```text
.codex/artifacts/20260718-020558_phase36-clean-stage-single-joint-no-collision-both
```

Result:

| Joint | Result | Max final absolute error | Max absolute error | Limit violations |
| --- | --- | ---: | ---: | ---: |
| left `waist` | PASS | 0.0040 | 0.0197 | 0 |
| right `shoulder` | PASS | 0.0040 | 0.0197 | 0 |

## Interpretation

This is the first strong dynamic-control pass on the clean runtime stage.

It proves:

1. `SingleArticulation` target control can work on the clean stage.
2. Runtime gain override can stabilize at least these smoke joints.
3. The Phase 35 failure was not caused by a completely broken Articulation API.
4. The currently composed collision layer is unsafe for free-space joint-control validation.

The practical conclusion is:

Do not use the current clean-stage collision geometry for grasp/contact simulation yet. First isolate and repair the collision layer.

## Next required gate

The next phase should inspect the 22 collision prims and identify why enabling them destabilizes a free-space one-joint movement.

Minimum next checks:

1. list all collision prim paths and bounding boxes;
2. identify self-overlapping or oversized collision geometry;
3. disable only suspected collision groups instead of all collisions;
4. validate the same single-joint gate after each group is removed;
5. keep only collision geometry that does not destabilize free-space control;
6. only then move toward bottle/table/pipe contact.

