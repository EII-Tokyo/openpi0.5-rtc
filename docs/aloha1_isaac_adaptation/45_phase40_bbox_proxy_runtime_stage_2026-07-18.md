# Phase 40: Bbox proxy runtime stage

## Question

Phase 39 showed that the clean ALOHA1 stage has no direct Mesh descendants under rigid bodies, but 22 rigid bodies do have valid composed bounding boxes.

This phase asks:

Can bbox-only collision proxies be added under the actual ALOHA1 rigid-body links without breaking the free-space controller gate?

## Code change

Added:

```text
aloha_isaac_replay/scripts/build_aloha1_bbox_proxy_runtime_stage.py
```

The builder creates an experimental stage:

```text
local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_bbox_proxy_runtime.usda
```

It:

1. sublayers the clean runtime stage;
2. disables the current root-level `/colliders` prims;
3. adds `UsdGeom.Cube` collision proxies under selected rigid-body links;
4. supports `--include-regex` and `--exclude-regex` so collision proxies can be enabled by group.

## Failed all-link proxy attempt

Command:

```bash
codex-evidence --name phase40-build-bbox-proxy-runtime-stage -- \
  .venv_issac/bin/python \
  aloha_isaac_replay/scripts/build_aloha1_bbox_proxy_runtime_stage.py \
  --stage-usd local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_clean_runtime.usda \
  --output-usd local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_bbox_proxy_runtime.usda \
  --output-dir reports/aloha1_isaac_adaptation/phase40_bbox_proxy_runtime_build_20260718 \
  --bbox-scale 0.6
```

Evidence:

```text
.codex/artifacts/20260718-022200_phase40-build-bbox-proxy-runtime-stage
```

Result:

| Check | Result |
| --- | ---: |
| Selected proxies | 22 |
| Disabled root collision prims | 22 |

Controller gate:

```bash
codex-evidence --name phase40-bbox-proxy-single-joint-smoke -- \
  .venv_issac/bin/python \
  aloha_isaac_replay/scripts/validate_aloha1_native_single_joint_response.py \
  --stage-usd local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_bbox_proxy_runtime.usda \
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
  --output-dir reports/aloha1_isaac_adaptation/phase40_bbox_proxy_single_joint_smoke_20260718
```

Evidence:

```text
.codex/artifacts/20260718-022214_phase40-bbox-proxy-single-joint-smoke
```

Result:

| Joint | Result | Max final absolute error | Direction OK |
| --- | --- | ---: | --- |
| left `waist` | PASS | 0.0133 | true |
| right `shoulder` | FAIL | 0.1732 | true |

Reducing all-link bbox scale to `0.3` and `0.45` still failed at least one joint gate, so the next viable path is grouped proxies rather than global shrinking.

## Passing gripper-only proxy attempt

Command:

```bash
codex-evidence --name phase40-build-bbox-proxy-runtime-stage-gripper-only -- \
  .venv_issac/bin/python \
  aloha_isaac_replay/scripts/build_aloha1_bbox_proxy_runtime_stage.py \
  --stage-usd local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_clean_runtime.usda \
  --output-usd local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_bbox_proxy_runtime.usda \
  --output-dir reports/aloha1_isaac_adaptation/phase40_bbox_proxy_runtime_build_gripper_only_20260718 \
  --bbox-scale 0.6 \
  --include-regex 'gripper|finger'
```

Evidence:

```text
.codex/artifacts/20260718-022452_phase40-bbox-proxy-single-joint-smoke-gripper-only
```

Result:

| Check | Result |
| --- | ---: |
| Selected proxies | 10 |
| Skipped rigid bodies | 18 |
| Disabled root collision prims | 22 |

Selected proxy links:

```text
gripper_link
gripper_prop_link
gripper_bar_link
left_finger_link
right_finger_link
```

on both left and right.

Controller gate:

| Joint | Result | Max final absolute error | Max absolute error | Limit violations | Direction OK |
| --- | --- | ---: | ---: | ---: | --- |
| left `waist` | PASS | 0.0107 | 0.0224 | 0 | true |
| right `shoulder` | PASS | 0.0039 | 0.0188 | 0 | true |

## Interpretation

All-link bbox proxies are too aggressive for this stage. They likely introduce self-contact or bad proxy placement around shoulder/arm links.

Gripper-only proxies are currently the first collision repair subset that passes the free-space dynamic gate. That makes them a reasonable next test target for bottle grasp/contact experiments.

This does not prove final grasp realism. It proves only:

- root-level imported `/colliders` remain disabled;
- link-owned gripper/finger bbox proxies can coexist with basic arm control;
- the repair strategy should proceed by small proxy groups, not by enabling all arm proxies at once.

## Decision

Treat:

```text
aloha1_dual_bbox_proxy_runtime.usda
```

as the current gripper-only proxy runtime stage after the gripper-only build command.

Do not use the all-link proxy configuration for downstream tests.

## Next step

Use the gripper-only proxy stage for the next minimum contact gate:

1. add a simple passive bottle proxy;
2. keep table and pipe simple and static;
3. close/open gripper fingers without arm motion;
4. verify gripper proxies produce stable contact and no explosions;
5. only then replay a short real qpos segment.
