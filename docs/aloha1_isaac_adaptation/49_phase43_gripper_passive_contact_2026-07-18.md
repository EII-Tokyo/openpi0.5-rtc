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
- DynamicCuboid run artifact:
  `.codex/artifacts/20260718-042059_phase43-gripper-passive-contact-dynamic-cuboid`
- Smaller/softer DynamicCuboid run artifact:
  `.codex/artifacts/20260718-042135_phase43-gripper-passive-contact-soft`

## Results

| run | object creation | status | open surface gap | object side length | object displacement | interpretation |
| --- | --- | --- | ---: | ---: | ---: | --- |
| default object | raw USD physics schemas | `FAILED_GATE` | 0.071133 stage units | 0.042680 stage units | 0.0 | Object stayed finite but did not move. |
| oversize object | raw USD physics schemas | `FAILED_GATE` | 0.071133 stage units | 0.092472 stage units | 0.0 | Even an oversized object did not move. |
| default object | Isaac `DynamicCuboid` | `FAILED_GATE` | 0.071133 stage units | 0.042680 stage units | 1.049855 | Dynamic contact happened, but the object was ejected too far. |
| smaller/softer object | Isaac `DynamicCuboid` | `FAILED_GATE` | 0.071133 stage units | 0.017783 stage units | 0.581294 | Reducing object size and closure amount improved but did not stabilize contact. |

## Interpretation

This is a useful failure, and it has two different modes.

The raw USD-created cube produced exactly zero displacement, even when oversized. That suggests the raw schema path did not enter active runtime contact in the way needed for this gate.

The Isaac `DynamicCuboid` path did enter dynamic contact, but the object was ejected too far. That suggests the gripper proxy contact is active but not yet stable enough for grasp or bottle contact.

Candidate causes:

1. The raw USD-created cube is not being registered into the active PhysX scene as a dynamic rigid body.
2. The bbox proxy colliders are too coarse, creating a strong penetration correction when the object touches the gripper.
3. Contact offset/rest offset/material settings are too aggressive for this tiny gripper-scale contact.
4. Finger closure is still too abrupt for the current contact geometry.
5. Stage-unit scaling may make apparently small numbers physically harsher than expected.

## Next Gate

Do not proceed to bottle contact yet.

The next diagnostic should inspect PhysX runtime state for:

- the passive cube rigid-body status;
- the cube collision enabled status;
- the finger proxy collision enabled status;
- contact pair generation between cube and finger proxies;
- whether adding the cube before the first `world.reset()` changes the result;
- whether smaller finger proxy shapes, contact/rest offset tuning, and physics materials reduce the DynamicCuboid ejection.
