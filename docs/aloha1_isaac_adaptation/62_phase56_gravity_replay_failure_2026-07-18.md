# Phase 56 Gravity Replay Failure

## Question

Phase 55 passed with gravity disabled.

Phase 56 asks:

If the same HDF5 left-arm plus gripper replay is run with gravity enabled, does the local Bottle500 contact gate remain stable?

## Command Artifact

```text
.codex/artifacts/20260718-142951_phase56-gravity-left-arm-gripper-tracking-bottle-usd-hdf5-full-replay
```

Structured report:

```text
reports/aloha1_isaac_adaptation/phase56_gravity_left_arm_gripper_tracking_bottle_usd_hdf5_full_replay_20260718/gripper_passive_contact_metrics.json
```

## Result

| Check | Result |
| --- | --- |
| status | `FAILED_GATE` |
| contact trace status | `FAIL_NO_TARGET_CONTACT` |
| wrong contact pairs | 0 |
| all expected fingers contacted object | false |
| object motion finite | true |
| no explosion | false |
| object close displacement | `72.8420` stage units |
| max object displacement | `72.9293` stage units |

The object centers show the core failure:

| State | Center |
| --- | --- |
| reset | `[0.5472, 0.7805, 0.0362]` |
| after settle | `[0.5392, 0.7627, -0.0511]` |
| final | `[0.5493, 0.7896, -72.8931]` |

The bottle fell under gravity.

## Tracking Was Not The Main Failure

The articulation still tracked the HDF5 replay with bounded error:

| Group | Max abs error | Mean max abs error | Final max abs error |
| --- | ---: | ---: | ---: |
| gripper | `0.0323760` | `0.0182076` | `0.0159157` |
| left arm | `0.0251148` | `0.0072564` | `0.0062799` |
| controlled | `0.0323760` | `0.0183529` | `0.0159157` |

Compared with Phase 55, gripper tracking became worse, but not enough to explain the object falling by more than 70 stage units. The dominant failure is missing physical support or grasp hold under gravity.

## Contact Coverage

The contact trace found target contact with the right fingertip proxy, but not both fingertip proxies:

```text
/puppet_left_vx300s/puppet_left_left_finger_link/bbox_collision_proxy: false
/puppet_left_vx300s/puppet_left_right_finger_link/bbox_collision_proxy: true
```

This explains why gravity is too strict for the current local gate. The bottle is not securely held by both sides before gravity is allowed to act.

## Interpretation

This failure is useful.

It shows that Phase 55 is a valid zero-gravity kinematic/contact gate, but the current local setup is not yet a gravity grasp gate.

Do not interpret Phase 56 as evidence that ALOHA1 mapping is wrong. The qpos tracking numbers are still bounded. The problem is that the simulated object is not constrained or supported in a way that matches the real key-region state.

## Decision

Do not proceed by blindly adding the table and pipe while leaving the initial grasp state ambiguous.

The next step should first define one of these gravity-safe semantics:

1. **already-grasped initialization**: start from a closed, bilateral contact frame and only then enable gravity;
2. **temporary support**: add a support surface under the bottle during settling, then test whether gripper closure takes over;
3. **measured object pose replay**: obtain or estimate the bottle pose relative to the gripper from video/FK, then place the bottle in that pose rather than using `gap_center`.

## Next Gate

Recommended Phase 57:

1. scan the same HDF5 qpos sequence for the earliest frame where the gripper is sufficiently closed;
2. initialize Bottle500 at that frame rather than frame 0;
3. run a short gravity-on replay window from that already-grasped state;
4. require bilateral fingertip contact and bounded object displacement.

