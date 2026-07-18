# Phase 57 Gravity Already-Grasped Replay

## Question

Phase 56 failed when gravity was enabled from frame 0 because the bottle fell before the gripper became closed enough.

Phase 57 asks:

If the replay starts from an already-grasped HDF5 frame, can the local Bottle500 USD gate pass with gravity enabled?

## Start Frame Selection

The selected HDF5 episode has 207 frames. The left gripper qpos stays mostly open until late in the segment:

| Frame | left gripper qpos |
| ---: | ---: |
| 0 | `0.6551978` |
| 120 | `0.6551978` |
| 140 | `0.6013092` |
| 143 | `0.3970998` |
| 160 | `0.3006676` |
| 180 | `0.2609603` |
| 206 | `0.2609603` |

Frame 143 is the first frame where the gripper is clearly near the closing phase:

```text
--hdf5-gripper-start-frame 143
```

This changed the physical meaning of the gate:

- Phase 56: object starts before grasp is established, then gravity acts.
- Phase 57: object starts in an already-grasped local state, then gravity acts.

## Command Artifact

```text
.codex/artifacts/20260718-143118_phase57-gravity-start143-left-arm-gripper-bottle-usd-hdf5-replay
```

Structured report:

```text
reports/aloha1_isaac_adaptation/phase57_gravity_start143_left_arm_gripper_bottle_usd_hdf5_replay_20260718/gripper_passive_contact_metrics.json
```

## Result

| Check | Result |
| --- | --- |
| status | `PASS` |
| overall pass | true |
| contact trace status | `PASS_BILATERAL_CONTACT_CANDIDATE` |
| gravity | `-9.81` |
| start frame | 143 |
| replayed steps | 63 |
| wrong contact pairs | 0 |
| both expected fingers contacted object | true |
| both expected fingers had `CONTACT_FOUND` events | true |
| object close displacement | `0.3326705` stage units |
| max object displacement | `0.4068589` stage units |
| no explosion | true |

HDF5 gripper summary for the selected window:

| Signal | Value |
| --- | ---: |
| samples | 64 |
| raw start | `0.3970998` |
| raw end | `0.2609603` |
| raw range | `0.1361395` |
| raw net | `-0.1361395` |
| first left finger target | `0.0352956` |
| last left finger target | `0.0303946` |
| left-arm max absolute net qpos delta | `0.2086214` |

Tracking summary:

| Group | Max abs error | Mean max abs error | Final max abs error |
| --- | ---: | ---: | ---: |
| gripper | `0.0338054` | `0.0288908` | `0.0152716` |
| left arm | `0.0225174` | `0.0098661` | `0.0063456` |
| controlled | `0.0338054` | `0.0290749` | `0.0152716` |

## Interpretation

This resolves the Phase 56 failure mechanism.

The gravity-on gate can pass if the local physical state is initialized as already grasped. The earlier failure was caused by starting from a frame where the bottle was not yet held securely.

This is directly relevant to the real task:

1. key-region replay must know whether the object is already grasped;
2. gravity validation should not start from an open-gripper pre-contact state unless a table/support model is present;
3. start-frame semantics matter as much as collision geometry.

## Decision

Use Phase 57 as the current gravity-on local contact gate:

```text
reports/aloha1_isaac_adaptation/phase57_gravity_start143_left_arm_gripper_bottle_usd_hdf5_replay_20260718/gripper_passive_contact_metrics.json
```

The current validated ladder is:

1. Phase 55: zero-gravity full HDF5 left-arm+gripper replay with tracking diagnostics.
2. Phase 56: gravity from open-gripper frame 0 fails, proving initialization semantics matter.
3. Phase 57: gravity from already-grasped frame 143 passes.

## Next Gate

The next gate should add a static table or support plane while keeping the start-frame semantics explicit:

1. run frame-0 with a support surface and gravity;
2. run frame-143 with the same support surface and gravity;
3. compare contact stability, tracking error, and object displacement;
4. only then add pipe collision.

