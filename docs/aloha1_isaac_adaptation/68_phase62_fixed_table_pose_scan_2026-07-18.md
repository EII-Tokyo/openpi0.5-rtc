# Phase 62 Fixed Table Pose Scan

## Question

Phase 60 used the measured table footprint, but the table was still placed under the bottle with:

```text
--support-plane-mode object_bottom
```

Phase 62 asks:

Can the same measured table footprint be placed as a fixed box in the replay stage, and can we find a height that does not collide with the left gripper proxy during the already-grasped HDF5 replay?

## Implementation

The passive-contact validator now supports fixed support placement:

```text
--support-plane-mode fixed_box
--support-plane-center X Y Z
```

The difference is important:

| Mode | Placement semantics | Use |
| --- | --- | --- |
| `object_bottom` | Derives the table/support height from the bottle bounding box | Diagnostic support only |
| `fixed_box` | Places the support collider at an explicit center pose | Fixed workcell pose scan |

The measured table footprint stayed constant in all Phase 62 runs:

```text
size_x = 1.220 m
size_y = 0.625 m
thickness = 0.040 m
```

The fixed table XY center was copied from the Phase 60 diagnostic placement:

```text
x = 0.593227851197621
y = 0.7853100288947757
```

Only the table center z was scanned.

## Command Artifacts

| Run | Artifact | Structured report |
| --- | --- | --- |
| same z | `.codex/artifacts/20260718-184902_phase62-fixed-measured-table-same-pose` | `reports/aloha1_isaac_adaptation/phase62_fixed_measured_table_same_pose_20260718/gripper_passive_contact_metrics.json` |
| minus 4 cm | `.codex/artifacts/20260718-185121_phase62c-fixed-measured-table-minus4cm` | `reports/aloha1_isaac_adaptation/phase62c_fixed_measured_table_minus4cm_20260718/gripper_passive_contact_metrics.json` |
| minus 6 cm | `.codex/artifacts/20260718-185149_phase62d-fixed-measured-table-minus6cm` | `reports/aloha1_isaac_adaptation/phase62d_fixed_measured_table_minus6cm_20260718/gripper_passive_contact_metrics.json` |
| minus 8 cm | `.codex/artifacts/20260718-184931_phase62b-fixed-measured-table-lower-z` | `reports/aloha1_isaac_adaptation/phase62b_fixed_measured_table_lower_z_20260718/gripper_passive_contact_metrics.json` |

## Result

All four runs passed the existing already-grasped replay gate:

```text
status = PASS
contact_trace_status = PASS_BILATERAL_CONTACT_CANDIDATE
```

The discriminating metric is whether the fixed table touches the gripper proxy.

| Run | Table center z | Table top z | Object displacement | Max object displacement | Left arm max error | Gripper max error | Table-object rows | Table-finger rows | Other table rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| same z | `-0.257145` | `-0.237145` | `0.022796` | `0.056195` | `0.021372` | `0.032376` | `770` | `30` | `0` |
| minus 4 cm | `-0.297145` | `-0.277145` | `0.048747` | `0.102409` | `0.020295` | `0.033805` | `728` | `14` | `0` |
| minus 6 cm | `-0.317145` | `-0.297145` | `0.078851` | `0.130171` | `0.020295` | `0.033805` | `655` | `0` | `0` |
| minus 8 cm | `-0.337145` | `-0.317145` | `0.106097` | `0.155298` | `0.022517` | `0.033805` | `563` | `0` | `0` |

## Interpretation

The same-z fixed table still collides with the right-finger proxy:

```text
/World/phase58_static_support_plane
/puppet_left_vx300s/puppet_left_right_finger_link/bbox_collision_proxy
```

Lowering the fixed table reduces that artificial table-finger collision:

```text
same z     -> 30 table-finger rows
minus 4 cm -> 14 table-finger rows
minus 6 cm -> 0 table-finger rows
minus 8 cm -> 0 table-finger rows
```

The highest scanned placement with zero table-finger contact is:

```text
center = [0.593227851197621, 0.7853100288947757, -0.3171450733686908]
size   = [1.22, 0.625, 0.04]
```

This is the current best fixed-table diagnostic candidate.

## Decision

Phase 62 passes as a fixed-table pose scan.

For this already-grasped HDF5 replay, a fixed measured-footprint table can be added without touching the gripper proxy if its center z is lowered by 6 cm from the Phase 60 diagnostic support pose.

## Limitation

This is not yet proof of the true real-world table pose.

The scan still borrows the XY center from the Phase 60 diagnostic setup. The result means:

1. the validator can now test a fixed table pose;
2. the measured table footprint does not inherently break the HDF5 replay;
3. the table-finger proxy collision is height-sensitive;
4. the next physical calibration step must determine the true table-to-robot transform.

Do not treat the minus-6-cm z as a measured physical fact. Treat it as the best current non-interfering fixed-table simulation candidate.

## Next Gate

Phase 63 should stop borrowing table XY from the bottle-relative diagnostic placement.

The next gate should define an explicit table frame:

```text
T_world_table
T_table_left_base
T_table_right_base
```

Then rerun the same already-grasped replay with:

1. fixed measured table geometry;
2. calibrated table-to-robot transform;
3. zero table-finger proxy contact;
4. low left-arm tracking error;
5. no unexpected contact pairs outside table-object and intended gripper-object contact.
