# Phase 48 First Contact Pair Trace

Date: 2026-07-18

## Goal

After the desktop reboot, bring the expert review loop back online and continue the ALOHA1 gripper contact gate without touching the real robot.

The immediate question was:

Can the Isaac Sim passive-object smoke test prove that the intended fingertip proxy contacts the intended object collider, rather than merely reporting unrelated contact pairs elsewhere in the stage?

## Expert Threads Restored

Four review threads were used as standing checks:

- Math / physics reviewer: contact count is not sufficient. The gate must check the target object, target finger, bounded motion, and persistence.
- QA reviewer: do not mark grasp success. The only acceptable pass label here is a single-finger contact isolation smoke pass.
- Isaac Sim reviewer: use PhysX contact reports with descendant path matching and keep USD updates enabled when the script depends on live bbox readback.
- ALOHA1 gripper reviewer: the ALOHA1 active gripper command is effectively left-finger driven with right-finger mimic semantics. Single-finger tests are diagnostic, not hardware control claims.

## Implementation Changes

`aloha_isaac_replay/scripts/validate_aloha1_gripper_passive_contact.py` now records explicit PhysX contact pairs.

The script now:

- applies `PhysxContactReportAPI` with threshold `0` to rigid bodies;
- reads `get_contact_report()` after each physics step;
- maps collider ids back to USD paths;
- distinguishes target object / target fingertip contact from unrelated stage contacts;
- records first contact pair, first target contact pair, target contact steps, and persistence;
- writes axis probe data for left finger, right finger, and object bbox positions;
- supports `--moving-fingers left|right|both`;
- supports `--object-placement moving_finger_surface`;
- keeps live USD updates enabled by default because bbox readback depends on it;
- avoids resetting the world after object placement;
- advances at least one pre-object physics step before measuring finger bboxes, even when `--settle-steps 0`;
- stores placement-time finger bboxes separately from final finger bboxes.

## Why This Was Needed

Earlier contact-count tests were false positives. The stage could report contacts between unrelated proxy colliders, including cross-arm proxy contacts, while the test object never touched the intended fingertip.

A valid smoke gate must ask a narrower question:

```text
Does the intended object collider touch the intended fingertip collider,
and does the object remain bounded after contact?
```

## Key Debugging Findings

### 1. Disabling USD updates hides object ejection

The first contact-trace helper copied a pattern similar to asset-validator contact probing and disabled USD / Fabric updates during contact tracing.

That is wrong for this script because the script also reads live bboxes from USD every step. With USD updates disabled, the object appeared stationary even when the physics object was being ejected.

The default is now:

```text
--trace-disable-usd-updates = false
```

### 2. Reset after object placement invalidates placement

The object is placed using the current open-pose fingertip bboxes. A later `world.reset()` can move the articulation under the already placed object, so the test starts from a different geometry than the one used for placement.

The script now records:

```text
reset_after_object_creation = false
```

### 3. `settle_steps=0` still needs one pre-object update

With no physics step after setting the open target, the bbox used for object placement can be stale. This caused the object to be placed relative to the wrong finger pose.

The script now uses:

```text
pre_object_update_steps = max(settle_steps, 1)
```

The placement bbox is stored separately:

```text
left_finger_placement_box
right_finger_placement_box
left_finger_final_box
right_finger_final_box
```

### 4. Final displacement must come from the simulation loop

After the contact trace is finished, bbox readback can be misleading for the final object pose. The final displacement is now taken from the last recorded simulation-loop bbox, not from a late post-trace read.

## Latest Validation Run

Command artifact:

```text
.codex/artifacts/20260718-081127_phase48-left-finger-surface-final-metrics
```

Metric files:

```text
reports/aloha1_isaac_adaptation/phase48_left_finger_surface_final_metrics_20260718/gripper_passive_contact_metrics.json
reports/aloha1_isaac_adaptation/phase48_left_finger_surface_final_metrics_20260718/gripper_passive_contact_timeseries.csv
reports/aloha1_isaac_adaptation/phase48_left_finger_surface_final_metrics_20260718/gripper_passive_contact_metrics.md
```

Key results:

```text
status: FAILED_GATE
contact_trace_status: FAIL_OBJECT_EJECTION
target_contact_pair_found: true
target_contact_found_event: true
first_target_contact_step: 0
target_contact_persistence_steps: 10
object_settle_displacement: 0.0
object_displacement: 1.2733965229834818
total_object_displacement: 1.2733965229834818
max_object_displacement: 1.2733965229834818
no_explosion_ok: false
contact_motion_ok: true
pre_object_update_steps: 1
```

First target contact pair:

```text
/puppet_left_vx300s/puppet_left_left_finger_link/bbox_collision_proxy
/World/phase43_passive_contact_cube
```

## Interpretation

This is meaningful progress, but it is not grasp success.

The previous failure mode was:

```text
FAIL_NO_TARGET_CONTACT
```

The current failure mode is:

```text
FAIL_OBJECT_EJECTION
```

That means the intended contact pair is now visible to PhysX, but the free test cube is pushed too far during closure.

The next gate is contact stability, not contact discovery.

## Next Engineering Step

Phase 49 should isolate why the object is ejected:

1. Test lower close offset and slower closure profile.
2. Test object linear / angular damping if available through the Isaac rigid body API.
3. Test contact offset / rest offset combinations on both fingertip proxy and object.
4. Test a kinematic or weakly constrained diagnostic object to separate collider direction from free-rigid-body dynamics.
5. Only after bounded passive-object contact passes, move back toward a bottle-shaped object or Grasp Editor validation.

Acceptance for the next gate:

```text
target_contact_pair_found = true
target_contact_found_event = true
object_displacement > min_contact_motion
max_object_displacement <= configured bound
contact_trace_status = PASS_SINGLE_FINGER_CONTACT_ISOLATION
```

