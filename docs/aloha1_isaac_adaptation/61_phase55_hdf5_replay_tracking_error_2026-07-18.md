# Phase 55 HDF5 Replay Tracking Error

## Question

Phase 54 proved that a real HDF5 left-arm plus gripper qpos sequence can pass the local Bottle500 USD contact gate.

Phase 55 asks a more diagnostic question:

While replaying that same sequence, how closely does the Isaac articulation readback follow the target qpos?

This is important because contact can look correct even when joint tracking is poor. Before adding gravity, table collision, pipe collision, or dual-arm replay, the validation script needs to report target-vs-readback error directly.

## Code Change

Updated:

```text
aloha_isaac_replay/scripts/validate_aloha1_gripper_passive_contact.py
```

The script now records tracking diagnostics for each simulation step:

```text
tracking_controlled_max_abs_error
tracking_controlled_rms_error
tracking_gripper_max_abs_error
tracking_gripper_rms_error
tracking_left_arm_max_abs_error
tracking_left_arm_rms_error
```

It also writes a `tracking_summary` section to the JSON report.

For HDF5 `left_arm_and_gripper` replay, the controlled group contains:

```text
waist
shoulder
elbow
forearm_roll
wrist_angle
wrist_rotate
left_finger
right_finger
```

The tracking metrics are observational in this phase. They do not yet decide pass/fail.

## Regression Test

Added:

```text
aloha_isaac_replay/tests/test_passive_contact_csv_writer.py
```

This test protects the CSV writer from dropping or rejecting late-added diagnostic columns. It reproduced the first Phase55 failure:

```text
ValueError: dict contains fields not in fieldnames
```

The fix keeps the legacy CSV column order and appends any additional diagnostic columns found in later rows.

## Command Artifact

```text
.codex/artifacts/20260718-142819_phase55-left-arm-gripper-tracking-bottle-usd-hdf5-full-replay
```

Structured report:

```text
reports/aloha1_isaac_adaptation/phase55_left_arm_gripper_tracking_bottle_usd_hdf5_full_replay_20260718/gripper_passive_contact_metrics.json
```

Timeseries CSV:

```text
reports/aloha1_isaac_adaptation/phase55_left_arm_gripper_tracking_bottle_usd_hdf5_full_replay_20260718/gripper_passive_contact_timeseries.csv
```

## Result

The contact gate still passed.

| Check | Result |
| --- | --- |
| status | `PASS` |
| overall pass | true |
| contact trace status | `PASS_BILATERAL_CONTACT_CANDIDATE` |
| wrong contact pairs | 0 |
| HDF5 samples | 207 |
| recorded tracking rows | 226 |
| object close displacement | `0.0903985` stage units |
| max object displacement | `0.0903985` stage units |

Tracking summary:

| Group | Max abs error | Mean max abs error | Final max abs error | Max RMS error | Mean RMS error | Final RMS error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gripper | `0.0212044` | `0.0082135` | `0.0088465` | `0.0150065` | `0.0061512` | `0.0086798` |
| left arm | `0.0204613` | `0.0071070` | `0.0092029` | `0.0096246` | `0.0036559` | `0.0042106` |
| controlled | `0.0212044` | `0.0098104` | `0.0092029` | `0.0093348` | `0.0046344` | `0.0056684` |

## Interpretation

This is a good diagnostic result for the current local gate:

1. the articulation follows the real HDF5 target sequence with finite, bounded error;
2. the left-arm group stays around `0.02` rad worst-case max absolute error in this run;
3. the gripper group has similar worst-case error;
4. the contact gate remains clean after adding tracking instrumentation.

However, this is not yet a calibrated control-quality threshold.

Reasons:

1. gravity is still disabled in this gate;
2. table and pipe collision are absent;
3. object motion is allowed because this phase focuses on local contact stability;
4. only the left ALOHA1 stage is loaded;
5. no full task trajectory, grasp lift, or insertion is being judged.

## Decision

Keep Phase 55 as the current tracking-observable contact gate:

```text
reports/aloha1_isaac_adaptation/phase55_left_arm_gripper_tracking_bottle_usd_hdf5_full_replay_20260718/gripper_passive_contact_metrics.json
```

The next gate can now use tracking error as an explicit decision criterion instead of adding new physics blindly.

## Next Gate

Phase 56 should introduce one new realism dimension at a time.

Recommended order:

1. repeat Phase55 with gravity enabled and the same local left-only setup;
2. compare tracking error and contact stability against Phase55;
3. if gravity is stable, add static table collision;
4. only after table collision is stable, add the measured pipe geometry.

