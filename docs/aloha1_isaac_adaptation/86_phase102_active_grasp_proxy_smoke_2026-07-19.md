# Phase102 Active-Contact Proxy Smoke Test

## Purpose

Phase101 proved that Phase97 must not be mislabeled as active grasp. Phase102 proves the stricter active-contact gate is reachable in Isaac when the setup actually starts out of target contact and then creates contact during close.

This is still not a full bottle grasp. It is a local single-finger cube-proxy smoke test.

## Command

```bash
codex-evidence --name aloha-phase102-active-grasp-proxy-smoke -- \
  .venv/bin/python aloha_isaac_replay/scripts/run_phase102_active_grasp_proxy_smoke.py
```

## Verified Run

Artifact:

`.codex/artifacts/20260719-002524_aloha-phase102-active-grasp-proxy-smoke`

Report:

`reports/aloha1_isaac_adaptation/phase102_active_gate_proxy_smoke_20260719/gripper_passive_contact_metrics.json`

Observed result:

| Field | Observed value |
| --- | --- |
| validator status | `PASS` |
| failure reasons | `[]` |
| contact trace status | `PASS_SINGLE_FINGER_CONTACT_ISOLATION` |
| target contact gate | active target contact required |
| first target contact found phase | `close` |
| first target contact found step | `14` |
| object shape | cube |
| moving finger | left |
| gravity | `0.0` |
| proxy contact offset | `0.001` |
| object contact offset | `0.001` |
| non-target contact gate | skipped, observed no categories |
| max controlled tracking error | `0.001869181187947589` |

## Why The Earlier Probe Failed

The first single-finger probe placed a cube with 3 mm geometric clearance from the moving finger, but it did not override the runtime proxy contact offset. The generated proxy runtime stage can carry a larger contact offset, so PhysX reported `CONTACT_FOUND` during `settle` even though the visual geometry had clearance.

The passing probe explicitly sets:

```text
--proxy-contact-offset 0.001
--proxy-rest-offset 0.0
--object-contact-offset 0.001
--object-rest-offset 0.0
```

That makes the active-contact test match the intended geometric meaning: no target contact during `settle`, first target contact during `close`.

## Scope Boundary

Phase102 proves the validator can distinguish active close-created contact from already-contacting replay.

It does not prove:

- full Bottle500 grasp;
- gravity-supported bottle lift;
- realistic bottle friction;
- calibrated table/base geometry;
- two-finger bilateral grasp stability;
- task-level insertion into the pipe.

## Next Gate

The next positive milestone should move from local single-finger cube contact to a two-finger object setup. That gate must still require:

1. no target contact during `settle`;
2. first target contact during `close`;
3. bounded object displacement;
4. no blocking non-target object contact;
5. controller tracking within the Phase97 threshold.
