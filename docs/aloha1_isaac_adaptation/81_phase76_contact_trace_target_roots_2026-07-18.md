# Phase 76 Contact Trace Target Roots

## Question

Phase 75 proved that the `/scene` ALOHA stage can be loaded with the `scene_base_link` profile and that the finger DOF aliases work.

The stricter question was:

```text
When Isaac PhysX reports real contact pairs, should the validator expect the authored bbox proxy path or the original finger-link collision subtree?
```

## Failure Before The Fix

The first strict trace run did not crash after the contact trace was moved earlier, but it failed the target-contact gate:

```text
status: FAILED_GATE
contact_trace_status: FAIL_NO_TARGET_CONTACT
contact_pair_count: 1838
target_contact_pair_found: False
```

Artifact:

```text
.codex/artifacts/20260718-213422_scene-passive-contact-trace-preauthored
```

This was not a pure "no physics contact" failure. The trace showed contacts between the object and real ALOHA finger collision prims such as:

```text
/scene/left_base_link/left_left_finger_link/collisions/left_left_g0/left_left_g0
/scene/left_base_link/left_left_finger_link/collisions/vx300s_8_custom_finger_left/vx300s_8_custom_finger_left
/scene/left_base_link/left_right_finger_link/collisions/left_right_g0/left_right_g0
/scene/left_base_link/left_right_finger_link/collisions/vx300s_8_custom_finger_right/vx300s_8_custom_finger_right
```

The validator failed because it expected:

```text
/scene/left_base_link/left_left_finger_link/bbox_collision_proxy
/scene/left_base_link/left_right_finger_link/bbox_collision_proxy
```

That mixed two different meanings:

```text
bbox proxy path:
used for bbox inspection and object placement

contact target path:
used to match PhysX contact-report collider paths
```

## Fix

The profile now separates these concepts:

```text
finger_proxy_paths
finger_contact_paths
```

For `scene_base_link`, contact matching now targets the finger link roots:

```text
/scene/left_base_link/left_left_finger_link
/scene/left_base_link/left_right_finger_link
/scene/right_base_link/right_left_finger_link
/scene/right_base_link/right_right_finger_link
```

The existing path matcher already accepts descendant collider paths. Therefore a contact report on:

```text
/scene/left_base_link/left_left_finger_link/collisions/left_left_g0/left_left_g0
```

correctly counts as contact with:

```text
/scene/left_base_link/left_left_finger_link
```

The trace setup also remains pre-authored before articulation creation and `world.reset()`, because applying `PhysxContactReportAPI` after tensor views exist can invalidate the articulation backend.

## Runtime Evidence After The Fix

The same strict trace command was rerun on the generated `/scene` proxy stage:

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

Result:

```text
status: PASS
overall_pass: True
contact_trace_status: PASS_BILATERAL_CONTACT_CANDIDATE
contact_pair_count: 1838
target_contact_pair_found: True
all_expected_fingers_target_contact_pair_found: True
first_target_contact_step: 0
target_contact_persistence_steps: 40
```

Artifact:

```text
.codex/artifacts/20260718-213804_scene-passive-contact-trace-target-roots
```

The expected contact targets in the JSON are now:

```text
/scene/left_base_link/left_left_finger_link
/scene/left_base_link/left_right_finger_link
```

## Important Limitation

This pass is still a local gripper-contact gate, not final bottle insertion validation.

The trace also showed the object contacting many other scene and robot collision prims during settle. That means this pass proves:

```text
PhysX contact tracing works
the /scene finger link contact target matcher works
the target finger collision descendants are detected
the previous FAIL_NO_TARGET_CONTACT was a gate semantics bug
```

It does not yet prove:

```text
the object is placed exactly like the real bottle
the table/support plane is calibrated
the full arm replay is dynamically valid
the bottle grasp remains stable under gravity
the contact set is physically clean enough for final RL validation
```

## Next Gate

The next stricter gate should separate:

```text
target finger-object contacts
non-target robot-object contacts
table/support contacts
cross-side contacts
```

Then it should decide which non-target contacts are acceptable for the specific test:

```text
gripper-only contact smoke
already-grasped bottle replay
full-arm bottle replay
table-supported insertion replay
```

Without that distinction, a test can pass the bilateral finger-contact gate while still hiding unrealistic object placement or table/body overlap.

## Validation

Validated locally:

```text
.venv/bin/python -m pytest -q \
  aloha_isaac_replay/tests/test_contact_proxy_profiles.py \
  aloha_isaac_replay/tests/test_passive_contact_csv_writer.py
```

Result:

```text
20 passed
```

No real robot or `192.168.1.103` control command was used.
