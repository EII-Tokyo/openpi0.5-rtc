# Phase 46 USD-Authored Fingertip Proxy Offsets

## Question

Phase 45 showed that fingertip-pad geometry is the right direction, but the passing passive-contact runs still emitted PhysX validation errors when proxy contact/rest offsets were authored at runtime inside the test script.

Phase 46 moved proxy contact/rest offsets and optional material parameters into the generated USD layer.

The goal was not to tune the final grasp yet. The goal was narrower:

1. prove the USD-authored parameter path is clean;
2. avoid runtime authoring errors;
3. measure whether that alone stabilizes passive object contact.

## Inputs

- Builder:
  `aloha_isaac_replay/scripts/build_aloha1_bbox_proxy_runtime_stage.py`
- Validator:
  `aloha_isaac_replay/scripts/validate_aloha1_gripper_passive_contact.py`
- Material stage:
  `local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_fingertip_pad_proxy_contact_runtime.usda`
- Offset-only stage:
  `local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_fingertip_pad_proxy_offset_runtime.usda`
- Material build artifact:
  `.codex/artifacts/20260718-043715_phase46-usd-authored-fingertip-stage-build`
- Material contact artifact:
  `.codex/artifacts/20260718-043730_phase46-usd-authored-fingertip-contact`
- Offset-only build artifact:
  `.codex/artifacts/20260718-043757_phase46-usd-authored-offset-only-stage-build`
- Offset-only contact artifact:
  `.codex/artifacts/20260718-043812_phase46-usd-authored-offset-only-contact`

## Stage Parameters

Both stages used:

- selected bodies: `finger_link$`
- `axis-scale = [0.18, 0.6, 0.18]`
- `min-extent = 0.003`
- proxy contact offset: `0.01`
- proxy rest offset: `-0.001`

The material stage additionally used:

- static friction: `1.0`
- dynamic friction: `0.8`
- restitution: `0.0`

The generated USD contains the expected authored proxy attributes:

- `physxCollision:contactOffset = 0.01`
- `physxCollision:restOffset = -0.001`

## Results

| run | stderr offset errors | status | object displacement | interpretation |
| --- | --- | --- | ---: | --- |
| material stage build | no | `PASS` | n/a | USD generation path is clean. |
| material stage contact | no | `FAILED_GATE` | 1.547390 | Contact still ejects object. |
| offset-only stage build | no | `PASS` | n/a | USD generation path is clean without material. |
| offset-only stage contact | no | `FAILED_GATE` | 3.990530 | Material was not the only failure cause. |

## Interpretation

Phase 46 separates two facts that were previously mixed together.

First, USD-authored proxy offsets are clean. They avoid the PhysX validation errors that appeared when the test script authored proxy offsets at runtime.

Second, clean authoring is not enough to make the current proxy contact stable. The passive object still gets ejected during closure.

This means the remaining problem is likely not “the offset API is invalid.” It is more likely one or more of:

1. proxy shape is still not aligned with the true fingertip contact surface;
2. closure is too abrupt for a free passive object;
3. the object starts in a marginal penetration or near-contact state;
4. the solver/contact parameters need a stable local-contact setup;
5. a cube is the wrong next object; a cylinder/bottle-mouth proxy may expose different contact behavior.

## Engineering Decision

Keep the USD-authored offset path.

Do not use runtime proxy offset authoring as the final solution.

Do not declare passive contact solved yet. The current gate is:

- clean USD authoring: `PASS`;
- stable passive contact: `FAIL`.

## Next Gate

Phase 47 should isolate geometry and closure:

1. visualize or log the fingertip-pad centers and object initial center before closure;
2. test a slower closure ramp;
3. test a smaller object and a cylinder proxy;
4. test one finger moving against a fixed passive object before opposed-finger closure;
5. only proceed to bottle grasp when passive contact is stable without PhysX errors.

