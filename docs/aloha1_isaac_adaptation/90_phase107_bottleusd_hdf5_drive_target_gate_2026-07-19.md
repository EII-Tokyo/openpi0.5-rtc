# Phase 107 BottleUSD HDF5 Drive-Target Gate

## Question

Can the full `/scene` runtime ALOHA1 stage replay a real HDF5 left-arm plus left-gripper qpos segment while holding the real Bottle500 USD in the Phase106 already-grasped pose?

This combines two previously separate facts:

1. Phase106: full-scene Bottle500 already-grasped contact can be clean.
2. Phase97: full-scene HDF5 drive-target replay can track a real qpos segment.

Phase107 asks whether these two facts hold together under gravity.

## Setup

Stage:

```text
local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose/aloha2_menagerie_scene_deep_black_real_start_pose_proxy_runtime.usda
```

Replay HDF5:

```text
local_rlt_data/raw_from_103/rollouts/key_regions/twist_off_the_bottle_cap/2026-06-17/rl/key_region_15c193959d7d449783517a9c9d257529/episode.hdf5
```

Mapping:

```text
configs/aloha/trossen_scene_base_link_aloha1_left_controller_mapping.yaml
```

Important parameters:

| Parameter | Value |
| --- | ---: |
| replay mode | `left_arm_and_gripper` |
| actuation mode | `drive_target` |
| start frame | `143` |
| target hold steps | `1` |
| gravity | `-9.81` |
| arm gain | `kp=1600`, `kd=100` |
| finger gain | `kp=200`, `kd=50` |
| object shape | `bottle_usd` |
| object axis | `X` |
| object center offset | `[0.08, 0.0, 0.0]` m |
| already-in-contact setup | `true` |
| fail on non-target object contact | `true` |
| allowed non-target category | `workcell_or_environment` |

The `+0.08 m` object-center offset is inherited from Phase106. It means the fingertips contact a body section of the long bottle, not the whole bottle bbox center.

## Runner

Added:

```text
aloha_isaac_replay/scripts/run_phase107_bottleusd_hdf5_drive_target_gate.py
```

Command:

```bash
codex-evidence --name aloha-phase107-bottleusd-hdf5-drive-target-runner -- \
  .venv/bin/python aloha_isaac_replay/scripts/run_phase107_bottleusd_hdf5_drive_target_gate.py
```

Artifact:

```text
.codex/artifacts/20260719-005334_aloha-phase107-bottleusd-hdf5-drive-target-runner
```

Structured report:

```text
reports/aloha1_isaac_adaptation/phase107_bottleusd_hdf5_drive_target_gate_20260719/gripper_passive_contact_metrics.json
```

## Result

| Check | Result |
| --- | --- |
| status | `PASS` |
| contact trace status | `PASS_BILATERAL_CONTACT_CANDIDATE` |
| both expected fingers contacted object | `true` |
| non-target object categories | `workcell_or_environment` |
| strict non-target gate | `true` |
| target contact persistence steps | `27` |
| object displacement during close | `0.26479 m` |
| total object displacement | `0.20596 m` |
| max object displacement | `0.20600 m` |
| contact motion lower bound | `true` |
| no explosion | `true` |

Tracking gate:

| Group | Max abs error | Final max abs error | Mean max abs error |
| --- | ---: | ---: | ---: |
| gripper | `0.00602` | `0.00315` | `0.00356` |
| left arm | `0.01287` | `0.00845` | `0.00685` |
| controlled | `0.01287` | `0.00845` | `0.00689` |

The controller tracking gate passed:

```text
PASS_POST_STEP_TRACKING_WITHIN_THRESHOLD
threshold = 0.02
max controlled error = 0.01287
worst DOF = left_shoulder
```

## What This Proves

Phase107 proves:

1. the full `/scene` ALOHA1 stage can replay a real HDF5 left-arm plus gripper segment;
2. the replay uses drive targets, not state teleport;
3. the Phase106 Bottle500 already-grasped pose remains usable while the arm moves;
4. controller tracking remains within the current `0.02` threshold;
5. Bottle500 contact remains finite and bounded under gravity;
6. same-side gripper base/bar/prop non-target contacts are not present in this run.

## What This Does Not Prove

Phase107 does not yet prove:

1. table contact is calibrated;
2. pipe collision is present or correctly positioned;
3. bottle-pipe insertion succeeds;
4. friction and mass are realistic enough for lift or insertion;
5. visual camera replay is aligned with the simulated scene;
6. the robot can autonomously choose this trajectory.

The object motion is not a task-success score. In this phase it is only a numerical stability signal: the object moved but did not explode, disappear, or enter disallowed non-target contacts.

## Decision

Treat Phase107 as the current strongest replay gate:

```text
full /scene runtime stage
+ real Bottle500 USD
+ Phase106 gripper-relative offset
+ real HDF5 left-arm and gripper qpos
+ drive-target control
+ gravity
+ tracking gate
+ non-target contact gate
```

## Next Gate

The next gate should add one realism dimension:

1. add a calibrated support plane/table collision;
2. keep the same HDF5 start frame and Bottle500 offset;
3. require the same tracking gate;
4. require the same non-target object-contact gate;
5. then add the measured pipe as a fixed collider.

Do not jump directly to insertion RL before table and pipe geometry are validated.
