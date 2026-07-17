# Phase 45 Fingertip-Pad Proxy Contact Gate

## Question

Phase 43 and Phase 44 showed that full finger bbox proxies can move with the finger links, and that a passive object can be dynamic after `world.reset()`. The remaining failure was contact stability during finger closure.

Phase 45 tested a smaller contact-surface proxy: keep only the finger-link proxies and shrink them into fingertip-pad-like boxes instead of using larger link bbox proxies.

## Inputs

- Stage builder:
  `aloha_isaac_replay/scripts/build_aloha1_bbox_proxy_runtime_stage.py`
- Passive contact validator:
  `aloha_isaac_replay/scripts/validate_aloha1_gripper_passive_contact.py`
- New stage:
  `local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_fingertip_pad_proxy_runtime.usda`
- Build artifact:
  `.codex/artifacts/20260718-043046_phase45-fingertip-pad-stage-build`
- Gap artifact:
  `.codex/artifacts/20260718-043101_phase45-fingertip-pad-gap`
- Default contact artifact:
  `.codex/artifacts/20260718-043102_phase45-fingertip-pad-contact`
- Object-offset-only artifact:
  `.codex/artifacts/20260718-043138_phase45-pad-contact-object-offset`
- Proxy-plus-object offset artifact:
  `.codex/artifacts/20260718-043139_phase45-pad-contact-proxy-offset`
- Proxy-contact-only artifact:
  `.codex/artifacts/20260718-043212_phase45-pad-contact-proxy-contact-only`
- Proxy negative-rest-offset artifact:
  `.codex/artifacts/20260718-043400_phase45-pad-contact-proxy-rest-negative`

## Stage Construction

The stage was generated from the clean runtime ALOHA1 stage with root-level `/colliders` disabled. Only rigid bodies whose path matched `finger_link$` received new collision proxies.

The proxy shape was intentionally anisotropic:

- `axis-scale = [0.18, 0.6, 0.18]`
- `min-extent = 0.003`

The build summary:

| metric | value |
| --- | ---: |
| rigid bodies found | 28 |
| selected proxies | 4 |
| skipped rigid bodies | 24 |
| disabled root-level collision prims | 22 |

This is a deliberate move away from full-chain bbox collision. The proxy is only meant to test local gripper-object contact.

## Results

| run | stage | offset setup | status | object displacement | stderr PhysX offset errors | interpretation |
| --- | --- | --- | --- | ---: | --- | --- |
| gap gate | fingertip pad | none | `PASS` | n/a | no | Finger pads still move with gripper DOFs. |
| default passive contact | fingertip pad | default DynamicCuboid offsets | `FAILED_GATE` | 1.186920 | no offset-authored errors | Smaller proxy alone was not enough. |
| object offset only | fingertip pad | object contact offset 0.01, rest offset 0 | `FAILED_GATE` | 1.336716 | no proxy offset errors | Object-side tuning alone was not enough. |
| proxy and object offset 0.01 | fingertip pad | object and proxy contact offset 0.01 | `PASS` | 0.001379 | yes | Numerically stable, but not clean because proxy offset authoring triggered PhysX validation errors. |
| proxy contact offset only | fingertip pad | object contact offset 0.01, proxy contact offset 0.01 | `PASS` | 0.001379 | yes | Same stable motion, but proxy rest offset default still made PhysX reject the proxy offset relationship. |
| proxy contact 0.01, proxy rest -0.001 | fingertip pad | explicit proxy contact/rest offsets | `PASS` | 0.002279 | yes | Still not clean; the script-level offset authoring path is not a final solution. |

## What This Means

The first important result is positive:

The old full bbox proxy is too coarse. A much smaller fingertip-pad proxy can pass the passive contact displacement gate when contact offsets are reduced.

The second important result is a warning:

The passing runs are not final because PhysX emitted offset validation errors for the proxy prims when the test script attempted to author proxy contact/rest offsets at runtime. Therefore the current evidence says:

1. fingertip-sized collision geometry is the right direction;
2. coarse full-link or full-chain bbox collision is the wrong direction;
3. runtime proxy offset authoring in the current script is not yet clean;
4. the next stable version should author contact/rest/material properties in the USD layer or through a verified Isaac API path, then rerun the gate with zero PhysX validation errors.

## Why Full-Chain Bbox Proxy Was Unstable

A bbox is an axis-aligned box around the whole visible mesh. For a robot link, that often includes empty space around curved or irregular geometry.

For gripper contact this is especially bad:

- the true fingertip contact patch is small;
- the bbox is much larger than the actual pad;
- an object can appear to contact the bbox before it contacts the real surface;
- closure then creates artificial penetration;
- PhysX resolves the penetration with a large impulse;
- the object is ejected, even though the gripper DOFs themselves are moving correctly.

So the symptom is not necessarily “bad controller.” It can be “bad collision shape creates impossible contact.”

## Why Root-Level `/colliders` Break Dynamic Control

The clean-stage audit found root-level `/colliders` prims separate from the moving articulation links.

In USD/PhysX semantics, a collision shape must be owned by the correct rigid body if it is supposed to move with that body. A root-level collider without the intended rigid-body ancestor behaves like independent static collision geometry or at least like geometry with the wrong owner.

That can break dynamic control in two ways:

1. The articulation moves, but stale/static robot-shaped colliders remain in space and collide with the moving robot.
2. The controller/drive constraints try to move joints while contact constraints push back from geometry that should not exist as an independent obstacle.

That is why the current runtime stage disables root-level `/colliders` and adds link-owned proxy colliders under the actual rigid-body links.

## Engineering Decision

Do not revive root-level `/colliders`.

Do not expand back to full-chain bbox collision.

Proceed with a controlled fingertip/contact-surface proxy layer:

1. author small finger-pad collision shapes under the finger rigid bodies;
2. author contact/rest offsets and physics material in the USD layer, not ad hoc during a smoke test;
3. require the passive contact gate to pass with zero PhysX validation errors;
4. only then test bottle-sized geometry and grasp closure;
5. keep full-arm collision disabled until each local collision layer is independently validated.

## Next Gate

The next implementation should produce a clean Phase46 gate:

- same fingertip-pad geometry;
- USD-authored contact/rest/material settings;
- passive cube contact `PASS`;
- no PhysX offset validation errors;
- repeated run consistency;
- then a simple cylinder/bottle proxy contact test.
