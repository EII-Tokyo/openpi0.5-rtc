# Phase 47 Closure and Object-Size Isolation

## Question

Phase 46 proved that USD-authored fingertip proxy offsets can be loaded without PhysX offset validation errors, but passive object contact still failed.

Phase 47 tested whether the failure was caused mainly by abrupt finger target changes or by object/proxy geometry.

## Inputs

- Validator:
  `aloha_isaac_replay/scripts/validate_aloha1_gripper_passive_contact.py`
- Stage:
  `local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_fingertip_pad_proxy_offset_runtime.usda`
- Linear closure artifact:
  `.codex/artifacts/20260718-043936_phase47-linear-closure-offset-only-contact`
- Small-object linear closure artifact:
  `.codex/artifacts/20260718-044011_phase47-small-object-linear-contact`

The validator now supports:

- `--closure-profile abrupt`
- `--closure-profile linear`

The default remains `abrupt`, so older tests keep their original behavior.

## Results

| run | closure profile | object fill fraction | status | object side length | object displacement | stderr offset errors |
| --- | --- | ---: | --- | ---: | ---: | --- |
| Phase46 offset-only baseline | abrupt | 0.6 | `FAILED_GATE` | 0.040309 | 3.990530 | no |
| Phase47 linear closure | linear | 0.6 | `FAILED_GATE` | 0.040309 | 3.990530 | no |
| Phase47 small object | linear | 0.2 | `FAILED_GATE` | 0.013436 | 1.203650 | no |

## Interpretation

Linear closure did not improve the result for the 0.6-fill object. The displacement was exactly the same as the abrupt baseline. That means this failure is not explained by a simple one-step target jump.

Shrinking the object reduced displacement from about 3.99 to about 1.20 stage units, but it still failed the stability gate. That means geometry scale matters, but smaller geometry alone is still not enough.

The current evidence points to a more local contact-model issue:

1. the proxy pads may still be poorly aligned with the true fingertip contact surface;
2. the object may be placed in a marginal or unstable contact region;
3. the test may need a one-finger-against-fixed-object gate before opposed-finger closure;
4. the cube may be a poor shape for this next gate compared with a cylinder or rounded bottle proxy;
5. drive gains/contact solver settings may still be too aggressive for small free passive objects.

## Engineering Decision

Keep the linear closure option because it is useful for future diagnostics, but do not assume closure smoothing solves the contact problem.

The next repair should isolate contact geometry before tuning controller parameters:

1. draw or log fingertip pad centers and object center at the first contact frame;
2. test a cylinder proxy;
3. test one moving finger against a fixed or kinematic object;
4. only then return to two-finger free passive grasp.

