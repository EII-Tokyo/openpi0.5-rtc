# Phase 77 Non-Target Contact Quality Gate

## Question

Phase 76 fixed the contact target matcher and proved that PhysX can report object contact with both left fingers in the `/scene` ALOHA stage.

The next question was whether this is enough for final validation.

It is not.

A bilateral finger contact can coexist with unrealistic object contacts:

```text
object touches target fingers
object also touches wrist / gripper base / opposite arm / workcell geometry
```

Such a test is useful as a smoke test, but it is not a clean physical grasp or bottle replay validation.

## Implementation

The passive contact validator now reports object-contact categories:

```text
target_finger
same_side_robot_non_target
other_side_robot
diagnostic_support
workcell_or_environment
unknown
```

It also writes:

```text
object_contact_pair_count
object_contact_categories
first_target_contact_phase
target_contact_found_during_settle
target_contact_found_during_close
first_non_target_object_contact_pair
first_non_target_object_contact_phase
non_target_object_contact_found
non_target_object_contact_pair_count
non_target_object_contact_categories
non_target_object_contact_ok
```

The default behavior remains diagnostic:

```text
target finger contact can still pass
non-target object contacts are reported but do not fail the smoke test
```

For final-quality gates, use:

```text
--fail-on-non-target-object-contact
```

Then any object contact outside the expected finger target roots fails the trace gate.

The calibrated overlay manifest now includes this flag by default in its generated contact validation command. Diagnostic smoke tests may omit it, but final calibrated contact validation must not.

## Runtime Evidence

The categorized trace was run with the same `/scene` proxy stage:

```text
--stage-usd /tmp/aloha_scene_bbox_proxy_runtime.usda
--stage-units-in-meters 1.0
--contact-proxy-profile scene_base_link
--side left
--object-shape cube
--object-creation dynamic_cuboid
--settle-steps 10
--close-steps 40
--closure-profile linear
--trace-contact-pairs
```

Default diagnostic result:

```text
status: PASS
contact_trace_status: PASS_BILATERAL_CONTACT_CANDIDATE
non_target_object_contact_found: True
non_target_object_contact_pair_count: 1038
```

Category counts:

```text
target_finger: 400 contact pairs, 8 unique pairs
same_side_robot_non_target: 390 contact pairs, 10 unique pairs
other_side_robot: 568 contact pairs, 12 unique pairs
workcell_or_environment: 80 contact pairs, 7 unique pairs
```

Artifact:

```text
.codex/artifacts/20260718-214141_scene-passive-contact-trace-categorized
```

## Strict Gate Evidence

The same trace was rerun with:

```text
--fail-on-non-target-object-contact
```

Strict result:

```text
status: FAILED_GATE
contact_trace_status: FAIL_NON_TARGET_OBJECT_CONTACT
non_target_object_contact_ok: False
non_target_object_contact_found: True
non_target_object_contact_pair_count: 1038
non_target_object_contact_categories:
  - other_side_robot
  - same_side_robot_non_target
  - workcell_or_environment
```

Artifact:

```text
.codex/artifacts/20260718-214207_scene-passive-contact-trace-strict-non-target-gate
```

## Interpretation

This is a useful failure.

It means:

```text
The contact report pipeline is working.
The validator can detect intended finger-object contact.
The validator can now also detect unrealistic extra object contacts.
```

It also means:

```text
The current cube smoke setup is not a final physical validation scene.
The object starts too broadly intersecting or contacting robot/workcell geometry.
The next phase must improve object placement, object size, support plane, and final replay-specific gates.
```

## Gate Policy

Use the gates differently by phase:

```text
gripper-only smoke:
  allow non-target contacts, but report them

geometry/debugging smoke:
  allow non-target contacts, because the goal is to inspect what contacts happen

already-grasped bottle replay:
  fail same-side non-target robot contacts and other-side contacts
  allow calibrated table/support contacts only when explicitly expected

full bottle insertion replay:
  fail all non-target robot/object contacts
  allow table/pipe contacts only if they are part of the task definition
```

## Next Gate

The next implementation should make object placement task-specific:

```text
small object between actual fingertip pads for gripper-only smoke
Bottle500 pose from a real HDF5 grasp for already-grasped replay
calibrated table/support and pipe geometry for final replay
```

Then run:

```text
--trace-contact-pairs
--fail-on-non-target-object-contact
```

on the task-specific scene.

## Validation

Validated locally:

```text
.venv/bin/python -m pytest -q \
  aloha_isaac_replay/tests/test_passive_contact_csv_writer.py \
  aloha_isaac_replay/tests/test_contact_proxy_profiles.py
```

Result:

```text
21 passed
```

No real robot or `192.168.1.103` control command was used.
