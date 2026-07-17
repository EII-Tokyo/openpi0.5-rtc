# Phase 44 Gripper Contact Runtime Inspection

## Question

Phase 43 showed that a passive object either did not move or was ejected during gripper closure. Phase 44 separated three possible causes:

1. The finger proxy colliders might not be owned by moving rigid-body links.
2. The passive object might not be a dynamic rigid body after `world.reset()`.
3. The object might already be unstable before deliberate finger closure.

## Inputs

- Stage:
  `local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_bbox_proxy_runtime.usda`
- Inspector:
  `aloha_isaac_replay/scripts/inspect_aloha1_gripper_contact_runtime.py`
- DynamicCuboid open-finger artifact:
  `.codex/artifacts/20260718-042701_phase44-contact-runtime-dynamic-open`
- Raw USD open-finger artifact:
  `.codex/artifacts/20260718-042703_phase44-contact-runtime-raw-usd-open`

## Results

| object creation | status | object dynamic after reset | finger proxies have rigid-body ancestor | warmup displacement | no warmup ejection |
| --- | --- | --- | --- | ---: | --- |
| Isaac `DynamicCuboid` | `PASS` | `True` | `True` | 0.012566 stage units | `True` |
| raw USD physics schemas | `PASS` | `True` | `True` | 0.0 stage units | `True` |

Important physics rows from the DynamicCuboid run:

| prim | collision | rigid body | rigid ancestor | mass | contact offset | rest offset |
| --- | --- | --- | --- | ---: | ---: | ---: |
| left finger bbox proxy | `True` | `False` | left finger link | None | None | None |
| right finger bbox proxy | `True` | `False` | right finger link | None | None | None |
| test object | `True` | `True` | none | 0.01 | 0.1 | 0.0 |

## Interpretation

Phase 44 rules out two earlier broad explanations:

1. The gripper bbox proxies are not detached static colliders. They have rigid-body ancestors under the moving finger links.
2. The test object is a dynamic rigid body after reset.

The failure is therefore more specific: the contact pair exists in principle, but the current contact geometry and contact parameters are not stable enough during finger closure.

The object does not explode during open-finger warmup. It explodes only during closure. That points to contact impulse generation between the object and coarse finger bbox proxies, not to a general stage startup failure.

## Engineering Conclusion

The gripper bbox proxies are good enough for a kinematic gap gate, but not good enough for passive-object contact.

The next repair should replace the finger bbox proxy with contact-surface proxies:

- small fingertip pad boxes;
- capsules aligned with the actual contact surface;
- or a pair of manually positioned convex primitives.

Do not expand back to full-arm or full-chain bbox collision. The research and Phase 43/44 evidence both point away from that route.

## Next Gate

Create a fingertip-pad proxy stage and repeat:

1. open-finger warmup inspection;
2. passive object closure with bounded displacement;
3. no PhysX offset validation errors;
4. no object ejection over the configured threshold.
