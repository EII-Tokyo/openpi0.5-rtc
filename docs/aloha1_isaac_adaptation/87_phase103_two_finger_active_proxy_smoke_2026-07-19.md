# Phase103 Two-Finger Active-Contact Proxy Smoke Test

## Purpose

Phase103 proves the current Trossen/Menagerie ALOHA1 scene-base gripper proxies can produce bilateral target contact during the close phase.

This is a two-finger dynamic cube proxy smoke test. It is not a full Bottle500 grasp, and it does not validate lift, friction realism, or task-level bottle insertion.

## Command

```bash
codex-evidence --name aloha-phase103-two-finger-active-proxy-smoke -- \
  .venv/bin/python aloha_isaac_replay/scripts/run_phase103_two_finger_active_proxy_smoke.py
```

## Verified Run

Artifact:

`.codex/artifacts/20260719-003521_aloha-phase103-two-finger-active-proxy-smoke`

Report:

`reports/aloha1_isaac_adaptation/phase103_two_finger_active_proxy_smoke_20260719/gripper_passive_contact_metrics.json`

Observed result:

| Field | Observed value |
| --- | --- |
| validator status | `PASS` |
| failure reasons | `[]` |
| contact trace status | `PASS_BILATERAL_CONTACT_CANDIDATE` |
| active target contact gate | `PASS_ACTIVE_TARGET_CONTACT_FOUND_DURING_CLOSE` |
| first target contact found phase | `close` |
| first target contact found step | `18` |
| left finger target contact | `true` |
| right finger target contact | `true` |
| all expected fingers target contact found | `true` |
| object shape | cube |
| object fill fraction | `0.90` |
| object side length | `0.061761544072211814 m` |
| object displacement | `0.013122011935083753 m` |
| right close sign | `+1.0` |
| gravity | `0.0` |
| proxy contact offset | `0.001` |
| object contact offset | `0.001` |

## Why Earlier Phase103 Probes Failed

The first two-finger probes used the legacy opposed-sign close target. In the current `scene_base_link` ALOHA1 proxy stage, that made the right finger move away from the cube along the contact axis.

The static-object probe then proved the object was not being pushed away, but the right finger still did not reach the cube. The right finger was already near its lower limit, so the synthetic close target needed to keep the right finger at the lower-side close convention instead of increasing it.

The passing gate uses:

```text
--right-finger-close-sign 1.0
--object-fill-fraction 0.90
```

The large-but-not-initially-touching cube leaves a small initial gap and then becomes contacted by both finger proxies during close.

## Scope Boundary

Phase103 proves:

1. the proxy collision bodies can be used by PhysX contact reporting;
2. the active-contact gate can reject settle-phase contact and accept close-phase contact;
3. both expected finger proxies can contact the same target object in the scene-base ALOHA1 namespace.

Phase103 does not prove:

- Bottle500 contact geometry is correct;
- the bottle can be stably grasped under gravity;
- contact material or friction is realistic;
- full-arm approach, lift, or insertion works;
- the current synthetic gripper target convention is the final robot-control convention.

## Engineering Meaning

Phase101 and Phase102 established that the validator distinguishes already-contacting replay from active close-created contact. Phase103 adds bilateral target coverage.

The remaining hard problem is no longer "does PhysX report target contact at all?" It is now:

1. place the real Bottle500 or measured bottle proxy in a physically correct pose relative to the ALOHA1 gripper;
2. use real grasp approach targets rather than synthetic gap-center targets;
3. validate gravity, friction, lift, and non-target contacts.

## Next Gate

The next positive milestone should use a bottle-shaped proxy or Bottle500 USD with a measured gripper-relative pose. It should require:

1. no target contact during settle;
2. first target contact during close;
3. both finger proxies contact the target;
4. object displacement is bounded but not artificially fixed;
5. non-target object contacts are either absent or explicitly justified;
6. controller tracking remains within the chosen contact-aware tolerance.
