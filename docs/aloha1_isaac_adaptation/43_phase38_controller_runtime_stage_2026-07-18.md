# Phase 38: Collision-disabled controller runtime stage

## Question

Phase 36 showed that clean-stage dynamic joint control passes only when the current root-level collision prims are disabled.

Phase 37 showed why: the 22 collision prims are not attached to the ALOHA1 articulation links and currently behave like suspicious static robot-shaped collision objects.

This phase asks:

Can the clean runtime asset builder generate a separate controller-only stage that disables those collision prims by construction, so controller and replay gates no longer need an extra runtime flag?

## Code change

Updated:

```text
aloha_isaac_replay/scripts/build_aloha1_clean_runtime_asset.py
```

The builder now emits two stages:

```text
local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_clean_runtime.usda
local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_controller_runtime.usda
```

The first stage keeps the clean runtime composition.

The second stage sublayers the same left and right clean wrappers, then authors an overlay for every current collision prim path with:

```text
physics:collisionEnabled = false
```

The builder report records the disabled collision paths so this is auditable instead of implicit.

## Build validation

Command:

```bash
codex-evidence --name phase38-build-controller-runtime-stage -- \
  .venv_issac/bin/python \
  aloha_isaac_replay/scripts/build_aloha1_clean_runtime_asset.py \
  --output-dir local_eval_assets/aloha1_clean_runtime_20260718 \
  --overwrite
```

Evidence:

```text
.codex/artifacts/20260718-021127_phase38-build-controller-runtime-stage
```

Generated report:

```text
local_eval_assets/aloha1_clean_runtime_20260718/clean_runtime_asset_report.json
local_eval_assets/aloha1_clean_runtime_20260718/clean_runtime_asset_report.md
```

Result:

| Check | Result |
| --- | ---: |
| Build status | PASS |
| Controller stage generated | PASS |
| Controller-disabled collision prims | 22 |
| Missing local reference targets | 0 |
| Runtime articulations initialized | 2 |

The generated controller stage contains 22 authored collision-disable opinions:

```text
physics:collisionEnabled = 0
```

## Dynamic controller validation

Command:

```bash
codex-evidence --name phase38-controller-runtime-single-joint-smoke -- \
  .venv_issac/bin/python \
  aloha_isaac_replay/scripts/validate_aloha1_native_single_joint_response.py \
  --stage-usd local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_controller_runtime.usda \
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
  --final-error-tolerance 0.1 \
  --output-dir reports/aloha1_isaac_adaptation/phase38_controller_runtime_single_joint_smoke_20260718
```

Evidence:

```text
.codex/artifacts/20260718-021152_phase38-controller-runtime-single-joint-smoke
```

Report:

```text
reports/aloha1_isaac_adaptation/phase38_controller_runtime_single_joint_smoke_20260718/single_joint_response_metrics.json
```

Result:

| Joint | Result | Max final absolute error | Max absolute error | Limit violations | Direction OK |
| --- | --- | ---: | ---: | ---: | --- |
| left `waist` | PASS | 0.0040 | 0.0197 | 0 | true |
| right `shoulder` | PASS | 0.0040 | 0.0197 | 0 | true |

This command did not pass `--disable-robot-collisions`; the controller stage itself carried the collision-disabled opinions.

## Interpretation

The controller runtime stage is now the stable target for:

- single-joint controller gates;
- qpos replay gates;
- future arm-only replay/controller experiments;
- debugging actuator gains and joint mapping without collision instability.

It is not a contact simulation target.

The root-level imported collision prims are disabled because they are currently unsafe, not because collision is unimportant. Bottle, table, pipe, and gripper contact remain blocked until collision geometry is repaired or replaced with link-owned simplified proxies.

## Decision

Use:

```text
aloha1_dual_controller_runtime.usda
```

for controller and replay validation.

Use:

```text
aloha1_dual_clean_runtime.usda
```

as the source asset for collision repair experiments.

Do not use either stage to claim grasp/contact success until a later phase validates repaired collision geometry with free-space joint control and contact tests.

## Next step

Develop a collision repair layer:

1. ignore the current root-level `/colliders` layer for contact;
2. create small simplified collision proxies under the actual link hierarchy;
3. enable one link group at a time;
4. run the Phase 38 single-joint gate after each group;
5. add table, pipe, and bottle contact only after free-space control remains stable.
