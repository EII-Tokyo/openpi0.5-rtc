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

- Runner artifact: `.codex/artifacts/20260719-013938_aloha-phase117-held-bottle-replay-with-mouth-frame`
- Metrics: `reports/aloha1_isaac_adaptation/phase117_diagnostic_held_bottle_replay_20260719/gripper_passive_contact_metrics.json`
- Bottle-mouth trajectory plot: `reports/aloha1_isaac_adaptation/phase117_diagnostic_held_bottle_replay_20260719/held_bottle_mouth_trajectory.png`
- Bottle-mouth trajectory summary: `reports/aloha1_isaac_adaptation/phase117_diagnostic_held_bottle_replay_20260719/held_bottle_mouth_trajectory_summary.json`

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

The validator now exports diagnostic object-frame features into the per-step CSV when held-object mode is enabled:

- object origin frame position;
- bottle mouth frame position;
- bottle mouth local axis direction in world coordinates.

These fields let a later plotting step compare the carried bottle mouth with the measured pipe axis.

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

## Bottle Mouth Vs Pipe Axis Diagnostic

The trajectory plot is generated with:

```bash
.venv/bin/python aloha_isaac_replay/scripts/plot_phase117_held_bottle_trajectory.py
```

The plot overlays:

- the carried bottle-mouth trajectory from the replay CSV;
- orange arrows showing the bottle-mouth direction;
- the measured pipe axis derived from `examples/aloha_isaac/config/workcell_user_measured.yaml`;
- the pipe entry point.

Current summary:

| Metric | Value |
| --- | --- |
| mouth samples | `146` |
| pipe start | `[-0.0300, 0.4075, 0.0700]` |
| pipe entry | `[-0.1919, 0.4075, 0.2263]` |
| mouth start | `[-0.1833, 0.3184, 0.3684]` |
| mouth end | `[-0.0637, 0.3298, 0.2989]` |
| mouth displacement | about `0.1388 m` |
| minimum mouth-to-pipe-axis distance | about `0.1566 m` |
| final mouth-to-pipe-axis distance | about `0.1612 m` |

This is a useful negative diagnostic. It means the replayed carried-bottle path and the currently measured pipe axis are not yet geometrically aligned in the same Isaac world frame. The likely remaining problem is table/pipe/robot base calibration, not controller tracking.

Therefore the Phase117 plot must be read as:

```text
The replay path is stable enough to inspect.
The measured workcell geometry is not yet aligned enough to claim insertion realism.
```

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

1. Calibrate the pipe axis and robot/table frame until the real successful insertion trajectory lands near the pipe entry.
2. Compare the held-bottle trajectory with the real HDF5 video from the same segment.
3. Once the diagnostic path is geometrically credible, move toward dynamic grasp/contact validation.
4. Only after that, re-enable dynamic object contact and test whether the bottle stays held without the kinematic assumption.
