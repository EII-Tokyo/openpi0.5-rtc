# Phase 43 Gripper Passive Contact Smoke

## Question

After Phase 42 showed that direct finger DOF control moves the gripper-only bbox proxies, the next gate tested whether a small passive object between the finger proxies participates in contact.

This is still not a grasp test. It only checks the local contact plumbing.

## Inputs

- Stage:
  `local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_bbox_proxy_runtime.usda`
- Validator:
  `aloha_isaac_replay/scripts/validate_aloha1_gripper_passive_contact.py`
- Default run artifact:
  `.codex/artifacts/20260718-041829_phase43-gripper-passive-contact`
- Oversize object run artifact:
  `.codex/artifacts/20260718-041906_phase43-gripper-passive-contact-oversize`

## Results

| run | status | open surface gap | object side length | object displacement | interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| default object | `FAILED_GATE` | 0.071133 stage units | 0.042680 stage units | 0.0 | Object stayed finite but did not move. |
| oversize object | `FAILED_GATE` | 0.071133 stage units | 0.092472 stage units | 0.0 | Even an oversized object did not move. |

## Interpretation

This is a useful failure. It is not a numerical explosion, and it is not just a too-small object. Both runs produced exactly zero object displacement.

The likely issue is that the passive cube or the gripper proxy colliders are not participating in dynamic contact as expected. Candidate causes:

1. The runtime-added cube is not being registered into the active PhysX scene as a dynamic rigid body.
2. The bbox proxy colliders exist in USD but are not active colliders in the articulation runtime.
3. A collision filter or collision group prevents the cube from contacting the finger proxies.
4. The proxy geometry is visually/kinematically attached but not in the solver's contact pair set.

## Next Gate

Do not proceed to bottle contact yet.

The next diagnostic should inspect PhysX runtime state for:

- the passive cube rigid-body status;
- the cube collision enabled status;
- the finger proxy collision enabled status;
- contact pair generation between cube and finger proxies;
- whether adding the cube before the first `world.reset()` changes the result;
- whether using Isaac core `DynamicCuboid` instead of raw USD schemas changes the result.

