# Phase 58 Support-Plane Gravity Replay

## Question

Phase 56 showed that gravity from frame 0 fails because the bottle falls before the gripper has established a grasp.

Phase 58 asks:

Can a temporary static support plane under the bottle make frame-0 gravity replay usable, and does it preserve the HDF5 left-arm tracking semantics?

## Why This Test Was Needed

Isaac Sim allows static colliders to support dynamic objects. That is useful for checking whether the failure is caused by missing physical support rather than joint mapping.

However, a support collider is only valid if it does not create artificial contacts with the gripper or arm. If the support surface intersects the gripper workspace, the replay may become physically different from the real robot.

## Implementation

The validator now has a diagnostic option:

```text
--support-plane-mode object_bottom
```

It places a static cube under the initial object bounding box:

```text
support z = object min z - clearance - thickness / 2
```

The structured JSON report records:

```text
support_plane.path
support_plane.center
support_plane.size_xy
support_plane.thickness
support_plane.placement_object_box
```

This is a diagnostic support, not a final table model.

## Runs

| Run | Artifact | Support size | Clearance | Status |
| --- | --- | ---: | ---: | --- |
| Phase 58 | `.codex/artifacts/20260718-143353_phase58-gravity-frame0-support-plane-bottle-usd-hdf5-replay` | `2.0` | `0.0` | `PASS` |
| Phase 58b | `.codex/artifacts/20260718-143427_phase58b-gravity-frame0-small-support-bottle-usd-hdf5-replay` | `0.24` | `0.0` | `PASS` |
| Phase 58c | `.codex/artifacts/20260718-143502_phase58c-gravity-frame0-low-support-bottle-usd-hdf5-replay` | `0.24` | `0.05` | `PASS` |

Structured reports:

```text
reports/aloha1_isaac_adaptation/phase58_gravity_frame0_support_plane_bottle_usd_hdf5_replay_20260718/gripper_passive_contact_metrics.json
reports/aloha1_isaac_adaptation/phase58b_gravity_frame0_small_support_bottle_usd_hdf5_replay_20260718/gripper_passive_contact_metrics.json
reports/aloha1_isaac_adaptation/phase58c_gravity_frame0_low_support_bottle_usd_hdf5_replay_20260718/gripper_passive_contact_metrics.json
```

## Tracking Comparison

| Run | Left-arm max abs error | Left-arm mean max abs error | Left-arm final max abs error | Gripper max abs error |
| --- | ---: | ---: | ---: | ---: |
| Phase 58 | `0.6818216` | `0.3125989` | `0.6818216` | `0.0308380` |
| Phase 58b | `0.4811937` | `0.2504563` | `0.0061812` | `0.0337544` |
| Phase 58c | `0.3796113` | `0.1638646` | `0.0069425` | `0.0338054` |

The low support improves tracking relative to the large support, but the early left-arm tracking error remains much worse than Phase 55 and Phase 57.

For comparison:

| Earlier gate | Left-arm max abs error | Meaning |
| --- | ---: | --- |
| Phase 55 | `0.02046` | zero-gravity full HDF5 tracking |
| Phase 57 | `0.02252` | gravity from already-grasped frame 143 |

## Contact Interpretation

The support plane prevents the bottle from falling, but it introduces extra contact pairs.

Phase 58c still has support-related contacts, including:

```text
/World/phase43_passive_contact_cube/... with /World/phase58_static_support_plane
/World/phase58_static_support_plane with /puppet_left_vx300s/puppet_left_right_finger_link/bbox_collision_proxy
```

This means the support surface is not a neutral table. It changes the replay physics by interacting with the object and at least one fingertip proxy.

## Conclusion

Phase 58 answers the narrow question:

The frame-0 gravity failure can be avoided by adding static support under the bottle.

But it also shows that this support-plane shortcut is not a validated final scene:

1. it creates support-object contacts by design;
2. it can also create support-finger contacts;
3. it significantly increases left-arm tracking error;
4. it may turn the replay into a table-interference test rather than a grasp/contact test.

Therefore Phase 58 is useful as a diagnostic, but Phase 57 remains the cleaner gravity-on local contact gate.

## Decision

Keep the support-plane option in the validator, but use it only for diagnostic isolation.

Do not treat `--support-plane-mode object_bottom` as proof that the final ALOHA1 workcell table is correct.

## Next Gate

The next gate should not be another local object-bottom support patch.

Use one of these instead:

1. **measured table geometry**: add the real table as a separate static collider in the correct world pose, away from artificial gripper interference;
2. **already-grasped replay**: keep Phase 57 as the gravity gate when the object pose is only defined relative to the gripper;
3. **measured bottle pose**: estimate the bottle pose relative to the gripper at replay start, then initialize the bottle there before enabling gravity.

