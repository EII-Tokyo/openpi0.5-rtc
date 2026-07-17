# Phase 41: Gripper DOF smoke on gripper-only proxy stage

## Question

Phase 40 produced the first collision proxy subset that passes the arm free-space controller gate: gripper-only bbox proxies.

Before adding a bottle, this phase asks:

Can the left and right `gripper` DOFs move stably on that gripper-only proxy stage?

## Stage under test

The current stage is:

```text
local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_bbox_proxy_runtime.usda
```

It was generated with:

```text
--include-regex 'gripper|finger'
--bbox-scale 0.6
```

This means it contains 10 gripper/finger bbox collision proxies and disables the 22 root-level imported `/colliders`.

## Validation

Command:

```bash
codex-evidence --name phase41-gripper-dof-single-joint-smoke -- \
  .venv_issac/bin/python \
  aloha_isaac_replay/scripts/validate_aloha1_native_single_joint_response.py \
  --stage-usd local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_bbox_proxy_runtime.usda \
  --stage-units-in-meters 1.0 \
  --joint left:gripper \
  --joint right:gripper \
  --phase-offset 0 \
  --phase-offset 0.005 \
  --phase-offset 0 \
  --phase-offset -0.005 \
  --phase-offset 0 \
  --phase-steps 80 \
  --settle-steps 20 \
  --arm-kp 200 \
  --arm-kd 200 \
  --final-error-tolerance 0.02 \
  --limit-margin 0.001 \
  --output-dir reports/aloha1_isaac_adaptation/phase41_gripper_dof_single_joint_smoke_20260718
```

Evidence:

```text
.codex/artifacts/20260718-022634_phase41-gripper-dof-single-joint-smoke
```

Report:

```text
reports/aloha1_isaac_adaptation/phase41_gripper_dof_single_joint_smoke_20260718/single_joint_response_metrics.json
reports/aloha1_isaac_adaptation/phase41_gripper_dof_single_joint_smoke_20260718/single_joint_response_metrics.md
```

## Result

| Joint | Result | Max final absolute error | Max absolute error | Limit violations | Direction OK |
| --- | --- | ---: | ---: | ---: | --- |
| left `gripper` | PASS | 0.0009 | 0.0044 | 0 | true |
| right `gripper` | PASS | 0.0000 | 0.0033 | 0 | true |

## Interpretation

The gripper-only proxy stage now passes two minimum dynamic gates:

1. arm smoke gate: left `waist`, right `shoulder`;
2. gripper smoke gate: left `gripper`, right `gripper`.

This still does not validate:

- finger mimic correctness;
- full gripper geometry accuracy;
- bottle contact;
- grasp stability;
- lift or insertion task dynamics.

It does show that the next contact gate can reasonably start with a simple passive bottle proxy and gripper-only collisions.

## Decision

Proceed to a minimal contact gate, not a full task replay:

1. add a simple passive cylinder bottle;
2. place it between the left gripper proxy fingers;
3. close/open the left gripper only;
4. verify contact stability, no explosion, and bounded bottle motion.
