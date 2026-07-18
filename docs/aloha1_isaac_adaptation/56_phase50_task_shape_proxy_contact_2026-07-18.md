# Phase 50 Task-Shape Proxy Contact

Date: 2026-07-18

## Goal

Phase 49 showed that a raw USD cube can pass the single-finger contact isolation gate. Phase 50 moves one step closer to the bottle task by replacing the cube with simple task-shape proxies:

- cylinder;
- capsule.

The goal is still not full bottle grasp. The goal is to prove that a rounded object, closer to a bottle body cross-section, can be contacted by the ALOHA1 fingertip proxy without numerical ejection.

## Implementation Change

`aloha_isaac_replay/scripts/validate_aloha1_gripper_passive_contact.py` now supports:

```text
--object-shape cube|cylinder|capsule
--object-axis X|Y|Z
--object-length-multiplier <float>
```

The default remains:

```text
--object-shape cube
--object-creation raw_usd
```

For Phase 50, both rounded proxies use:

```text
--object-axis X
--object-length-multiplier 4
```

This means the proxy object is elongated along X while the gripper closes across the Y gap. That approximates a finger contacting the side of a bottle-like body rather than contacting a cube corner or cap face.

## Test A: Cylinder

Artifact:

```text
.codex/artifacts/20260718-135300_phase50-left-finger-cylinder-contact
```

Metrics:

```text
reports/aloha1_isaac_adaptation/phase50_left_finger_cylinder_contact_20260718/gripper_passive_contact_metrics.json
```

Result:

```text
status: PASS
contact_trace_status: PASS_SINGLE_FINGER_CONTACT_ISOLATION
target_contact_pair_found: true
target_contact_found_event: true
first_target_contact_step: 0
target_contact_persistence_steps: 10
object_displacement: 0.1275587492835375
max_object_displacement: 0.1275587492835375
no_explosion_ok: true
contact_motion_ok: true
```

Inputs:

```text
object_creation: raw_usd
object_shape: cylinder
object_axis: X
object_length_multiplier: 4.0
```

## Test B: Capsule

Artifact:

```text
.codex/artifacts/20260718-135329_phase50-left-finger-capsule-contact
```

Metrics:

```text
reports/aloha1_isaac_adaptation/phase50_left_finger_capsule_contact_20260718/gripper_passive_contact_metrics.json
```

Result:

```text
status: PASS
contact_trace_status: PASS_SINGLE_FINGER_CONTACT_ISOLATION
target_contact_pair_found: true
target_contact_found_event: true
first_target_contact_step: 0
target_contact_persistence_steps: 9
object_displacement: 0.13519911513398328
max_object_displacement: 0.13519911513398328
no_explosion_ok: true
contact_motion_ok: true
```

Inputs:

```text
object_creation: raw_usd
object_shape: capsule
object_axis: X
object_length_multiplier: 4.0
```

## Interpretation

The contact stack has now passed three increasingly relevant gates:

1. raw USD cube;
2. raw USD cylinder;
3. raw USD capsule.

This is stronger than Phase 49 because the object is now rounded and elongated, closer to the geometry class of a bottle body.

It still does not prove:

- full bottle mesh collision correctness;
- friction correctness;
- closed-loop grasp stability;
- lift success;
- Grasp Editor transform correctness;
- full ALOHA1 arm approach and grasp behavior.

## Next Gate

Phase 51 should move from local fingertip contact to a bottle-like object gate.

Recommended next sequence:

1. Use a simple composite raw USD bottle proxy: body cylinder, neck cylinder, mouth marker.
2. Keep contact-pair tracing.
3. First test local fingertip contact only.
4. Then test bilateral finger contact.
5. Only after those pass, connect to Grasp Editor and whole-arm pregrasp / grasp validation.

Do not jump directly from this result to a claim of bottle grasp success.

