# Phase 106 BottleUSD Already-Grasped Gate

## Question

Can the full `/scene` runtime ALOHA1 stage keep the real Bottle500 USD in a physically usable gripper-relative pose, with both left fingertip proxies contacting the bottle and no same-side non-target object contacts?

This is different from asking whether the gripper can pick up a bottle from free space. The target state for bottle-mouth insertion is often already-grasped: the bottle is already in the left gripper, and the next validation step is whether arm replay and insertion can start from that state without false gripper-base or workcell collisions.

## Why Phase104 and Phase105 Failed

Phase104 used the existing GraspSpec YAML pose. The math transform itself was valid, but the resulting physical pose put Bottle500 collision bodies into the gripper base/bar/prop. That is a bad physical grasp candidate, not a PhysX contact-reporting failure.

Phase105 used `gap_center` with no long-axis offset. That put the whole Bottle500 bbox center between the fingertips. For a long bottle, that is not the same thing as putting the gripped bottle section between the fingertips. The bottle length then reached into the gripper base region.

The important correction is:

```text
finger contact center != whole bottle bbox center
```

For a long bottle, the gripper contacts one section of the bottle body. The whole-object bbox center can be several centimeters away along the bottle long axis.

## Code Change

Updated:

```text
aloha_isaac_replay/scripts/validate_aloha1_gripper_passive_contact.py
```

Added:

```text
--object-center-offset DX DY DZ
```

This offset is applied in world coordinates after the nominal `gap_center`, `moving_finger_surface`, or `grasp_yaml` placement. It is recorded in the JSON report as:

```text
object_placement.center_offset_world
object_placement.placed_center_after_offset
```

Added reproducible runner:

```text
aloha_isaac_replay/scripts/run_phase106_bottleusd_already_grasped_gate.py
```

## Selected Pose

Stage:

```text
local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose/aloha2_menagerie_scene_deep_black_real_start_pose_proxy_runtime.usda
```

Bottle:

```text
assets/bottle_500ml/isaac/bottle_500ml_sim.usd
```

Critical placement parameters:

| Parameter | Value |
| --- | ---: |
| object shape | `bottle_usd` |
| object axis | `X` |
| nominal placement | `gap_center` |
| center offset | `[0.08, 0.0, 0.0]` m |
| contact offset | `0.001` m |
| already-in-contact setup | `true` |
| fail on non-target object contact | `true` |

The `+0.08 m` X offset moves the whole Bottle500 bbox center away from the gripper base while leaving the bottle body section inside the fingertip contact region.

## Command

```bash
codex-evidence --name aloha-phase106-bottleusd-already-grasped-runner -- \
  .venv/bin/python aloha_isaac_replay/scripts/run_phase106_bottleusd_already_grasped_gate.py
```

Artifact:

```text
.codex/artifacts/20260719-005036_aloha-phase106-bottleusd-already-grasped-runner
```

Structured report:

```text
reports/aloha1_isaac_adaptation/phase106_bottleusd_already_grasped_gate_20260719/gripper_passive_contact_metrics.json
```

## Result

| Check | Result |
| --- | --- |
| status | `PASS` |
| contact trace status | `PASS_BILATERAL_CONTACT_CANDIDATE` |
| both expected fingers contacted object | `true` |
| non-target object contact categories | `[]` |
| strict non-target gate | `true` |
| target contact persistence steps | `180` |
| object displacement | `0.01206 m` |
| contact motion lower bound | `true` |
| no explosion | `true` |

The initial Bottle500 bbox was:

```text
min = [-0.22753, -0.06705, 0.23663]
max = [-0.02153,  0.00095, 0.30463]
size = [0.20600, 0.06800, 0.06800] m
```

That size is physically meaningful for this model: about `206 mm` long and `68 mm` diameter.

## What This Proves

Phase106 proves:

1. the full `/scene` runtime stage can host the real Bottle500 USD without relying on a left-only stripped stage;
2. the current fingertip proxy contact trace can see Bottle500 child collision meshes;
3. a gripper-relative long-axis offset can remove gripper-base false contact;
4. the strict non-target contact gate can pass in the full scene;
5. an already-grasped Bottle500 state is now available for the next replay gates.

## What This Does Not Prove

Phase106 does not prove:

1. no-contact-to-grasp active closing succeeds;
2. gravity lift succeeds;
3. friction values are calibrated;
4. the bottle-pipe insertion task succeeds;
5. the existing GraspSpec YAML is physically correct;
6. arm IK or full-arm approach is solved.

The no-contact active-grasp gate remains harder because Bottle500 diameter is about `68 mm`, while the current open fingertip gap is only about `68.6 mm`. With a PhysX contact offset of `1 mm`, the full-size bottle is already in contact at setup. That is expected for an already-grasped key-region state, but it is not a clean pick-up-from-free-space test.

## Decision

Treat Phase106 as the current full-scene already-grasped Bottle500 gate.

The next gate should use this object pose and replay a real left-arm plus gripper HDF5 segment. It should keep:

1. `--object-shape bottle_usd`;
2. `--object-axis X`;
3. `--object-center-offset 0.08 0 0`;
4. `--already-in-contact-setup` for insertion-phase replay;
5. `--fail-on-non-target-object-contact`;
6. bounded object displacement and controller tracking gates.
