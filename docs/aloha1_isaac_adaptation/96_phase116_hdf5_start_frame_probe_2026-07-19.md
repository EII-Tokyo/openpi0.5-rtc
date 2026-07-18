# Phase 116: HDF5 Start-Frame Probe

## Question

Can the strict Phase115 measured-workcell gate be extended by simply replaying the same HDF5 episode from earlier start frames?

## Fixed Setup

All runs inherited Phase115 settings:

- measured-workcell runtime stage;
- `support_plane_mode = none`;
- workcell contact policy enabled;
- object center offset `0.0, 0.0, 0.0`;
- HDF5 drive-target replay at 50 Hz target cadence;
- arm gains `kp=1600`, `kd=100`;
- finger gains `kp=200`, `kd=50`.

Only the HDF5 start frame changed.

## Evidence

- Runner artifact: `.codex/artifacts/20260719-013348_aloha-phase116-hdf5-start-frame-probe`
- Summary: `reports/aloha1_isaac_adaptation/phase116_hdf5_start_frame_probe_20260719/summary.json`

## Results

| Start frame | Result | Contact status | Controller tracking | Object contact categories | Interpretation |
| --- | --- | --- | --- | --- | --- |
| `80` | failed | `FAIL_WORKCELL_CONTACT_POLICY` | pass | target finger + workcell/floor | object falls before stable grasp |
| `100` | failed | `FAIL_NON_TARGET_OBJECT_CONTACT` | pass | target finger + same-side robot non-target | object reaches gripper-base/body contact |
| `120` | failed | `FAIL_WORKCELL_CONTACT_POLICY` | pass | target finger + workcell/floor | object falls before stable grasp |
| `143` | pass | `PASS_BILATERAL_CONTACT_CANDIDATE` | pass | target finger only | current stable contact window |

The controlled joint replay is not the limiting factor. The maximum controlled error stayed below the `0.02 rad` gate in all four runs.

## Interpretation

This is not evidence that the ALOHA1 arm replay is wrong. It is evidence that the bottle physics model is incomplete for longer replay:

1. The object is currently spawned near the gripper at the first replay frame.
2. There is no explicit "bottle is already held by the gripper" constraint.
3. Starting earlier means the gripper pose, bottle pose, and contact geometry can be inconsistent for several frames.
4. The bottle can fall or collide with same-side non-target geometry before the later successful key-region part.

So the next engineering target is not "increase stiffness" or "delete more collisions". The next target is to model held-object semantics.

## Next Gate

Add a diagnostic held-bottle replay mode:

- derive the initial bottle-to-left-gripper transform at the accepted Phase115 contact pose;
- during replay, move the bottle from the current left gripper transform using that fixed relative transform;
- keep this marked as a diagnostic kinematic-held-object gate, not as a final dynamic grasp proof;
- compare bottle trajectory, pipe direction, and contacts.

After that diagnostic path is stable, replace it with a dynamic grasp/contact model and only then treat longer replay as a physics validation.

