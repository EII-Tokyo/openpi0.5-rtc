# Phase 117: Diagnostic Held-Bottle Replay

## Question

If the bottle is treated as already held by the left gripper, can the longer HDF5 replay from frame `80` be simulated without the object falling or colliding with unrelated geometry?

## Why This Phase Exists

Phase116 showed that earlier start frames fail even though the arm controller tracks correctly. The failure mode was object physics, not joint replay:

- the bottle was not constrained to the gripper;
- it could fall to the floor;
- it could collide with same-side non-target geometry;
- only the later frame `143` had a stable passive contact window.

Therefore Phase117 adds a diagnostic held-object mode.

This is not a dynamic grasp proof. It is equivalent to saying:

```text
Assume the bottle is already held by the gripper.
Replay the arm.
Move the bottle with the gripper using a fixed relative transform.
Check the carried-object trajectory and controller stability.
```

## Evidence

- Runner artifact: `.codex/artifacts/20260719-013701_aloha-phase117-diagnostic-held-bottle-replay`
- Metrics: `reports/aloha1_isaac_adaptation/phase117_diagnostic_held_bottle_replay_20260719/gripper_passive_contact_metrics.json`

## Implementation

The validator now supports:

```bash
--diagnostic-held-object-mode follow_gripper
```

When this mode is enabled:

1. the object is placed at the initial frame;
2. the initial transform from gripper to object is recorded;
3. after every replay step, the object world pose is recomputed from the current gripper world pose;
4. the metrics explicitly mark this as `DIAGNOSTIC_NOT_DYNAMIC_GRASP_PROOF`.

Phase117 also disables the object's rigid body because this mode is kinematic scene replay, not dynamic grasp simulation.

## Result

| Metric | Value |
| --- | --- |
| status | `PASS` |
| contact trace status | `NOT_TRACED` |
| start frame | `80` |
| replay steps | `126` |
| physics steps | `146` |
| support plane | `none` |
| object placement | `grasp_yaml` |
| object rigid body | `false` |
| max controlled error | about `0.0161 rad` |
| target limits | pass |
| total object displacement | about `0.1138 m` |

The useful conclusion is that a longer carried-object trajectory is now reproducible inside Isaac without relying on the unstable passive object dynamics from Phase116.

## Current Boundary

Phase117 should be used to inspect:

- bottle path;
- bottle mouth direction;
- relative bottle-to-pipe trajectory;
- replay/controller stability over a longer segment.

It must not be used as evidence that:

- the gripper dynamically grasps the bottle;
- friction and contact parameters are correct;
- the bottle would stay held without a kinematic assumption.

## Next Work

1. Add trajectory export for bottle mouth and pipe axis.
2. Render or plot the held-bottle replay path.
3. Compare the held-bottle trajectory with real HDF5 video.
4. Only after this diagnostic path is geometrically credible, move toward dynamic grasp/contact validation.

