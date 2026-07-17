# Phase 49 Raw USD Contact Stability

Date: 2026-07-18

## Goal

After Phase 48 proved that the target fingertip-to-object contact pair can be traced, Phase 49 asked a narrower question:

Is the object ejection caused by gripper contact geometry, or by the way the passive test object is created?

## Hypothesis

The `dynamic_cuboid` object path may not preserve the intended stage-unit size in this ALOHA1 scene.

If that is true, then a test object created directly with USD geometry and USD physics APIs should be more stable under the same gripper closure.

## Test A: DynamicCuboid

Artifact:

```text
.codex/artifacts/20260718-081250_phase49-left-finger-surface-small-close
```

Metrics:

```text
reports/aloha1_isaac_adaptation/phase49_left_finger_surface_small_close_20260718/gripper_passive_contact_metrics.json
```

Result:

```text
status: FAILED_GATE
contact_trace_status: FAIL_OBJECT_EJECTION
target_contact_pair_found: true
target_contact_found_event: true
object_displacement: 1.3375050826196837
max_object_displacement: 1.3375050826196837
no_explosion_ok: false
```

Important observation:

The intended side length was:

```text
object_side_length_stage_units: 0.007119988153530716
```

But the recorded object bbox thickness along the gripper gap axis was about:

```text
5e-5
```

This is much smaller than the intended diagnostic object. The object behaved like a tiny thin object and was pushed far outside the contact region.

## Test B: Raw USD Cube

Artifact:

```text
.codex/artifacts/20260718-081340_phase49-left-finger-raw-usd-cube
```

Metrics:

```text
reports/aloha1_isaac_adaptation/phase49_left_finger_raw_usd_cube_20260718/gripper_passive_contact_metrics.json
```

Result:

```text
status: PASS
contact_trace_status: PASS_SINGLE_FINGER_CONTACT_ISOLATION
target_contact_pair_found: true
target_contact_found_event: true
first_target_contact_step: 0
target_contact_persistence_steps: 10
object_displacement: 0.10287488408754399
max_object_displacement: 0.10287488408754399
no_explosion_ok: true
contact_motion_ok: true
```

The first target contact pair was:

```text
/puppet_left_vx300s/puppet_left_left_finger_link/bbox_collision_proxy
/World/phase43_passive_contact_cube
```

## Conclusion

The Phase 48 ejection was not proof that the fingertip proxy contact design is unusable.

The stronger conclusion is:

```text
raw_usd object creation is the valid smoke-test path for this stage-unit setup.
dynamic_cuboid is not reliable enough for the current ALOHA1 gripper contact gate.
```

The script default is changed to:

```text
--object-creation raw_usd
```

`dynamic_cuboid` remains available only as an explicit ablation.

## What This Pass Means

This pass means:

- the intended fingertip proxy collider can contact the intended object collider;
- the contact report can identify the correct pair;
- the object moves a bounded amount instead of being ejected;
- the result is stable enough to proceed to a more realistic task-shape object.

This still does not mean:

- bottle grasp is solved;
- friction is realistic;
- full-arm approach is valid;
- the Grasp Editor transform is validated;
- the robot can pick up a bottle.

## Next Gate

Phase 50 should replace the tiny cube with a task-relevant object shape:

1. Use a simple cylinder or capsule proxy before returning to the full bottle mesh.
2. Keep the raw USD creation path.
3. Keep explicit contact-pair tracing.
4. Preserve the same PASS criteria:

```text
target_contact_pair_found = true
target_contact_found_event = true
max_object_displacement <= configured bound
contact_trace_status = PASS_SINGLE_FINGER_CONTACT_ISOLATION
```

