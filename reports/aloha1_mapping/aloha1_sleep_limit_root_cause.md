# ALOHA1 Sleep limit root-cause audit

- Status: `VERIFIED_ROOT_CAUSE`
- Classification: `OFFICIAL_ROS2_ALOHA_SLEEP_CONFIGURATION_OUTSIDE_ITS_OWN_URDF_LIMITS`
- Video interpretation: `PASS_TRAJECTORY_VISUAL`
- Exact Sleep endpoint: `FAIL`
- Signal correspondence: `PARTIAL`
- Real execution: `NOT_RUN_UNAUTHORIZED`

## What happened

The video is valid evidence that the simulated arm moved smoothly through three cycles and returned Home. It was not a visual-motion failure. The mismatch is at the command boundary: the pinned ROS 2 ALOHA Sleep vector exceeds the pinned `aloha_vx300s` URDF limits for three joints.

The official ALOHA helper interpolates and calls `set_joint_positions`. The Interbotix Python API rejects the whole group sample as soon as any one joint is illegal. The Isaac runner instead kept submitting commands and PhysX stopped each joint independently. These two behaviors can look similarly safe in a video while being different signals.

## Conflicting joints

| Joint | Sleep target | URDF lower | URDF upper | Violation |
|---|---:|---:|---:|---:|
| `shoulder` | -2.050000000 | -1.850049007 | 1.256637061 | 0.199950993 |
| `elbow` | 1.700000000 | -1.762782545 | 1.605702912 | 0.094297088 |
| `wrist_angle` | -2.000000000 | -1.867502300 | 2.234021443 | 0.132497700 |

## Deterministic API emulation

- First rejected outbound sample: `204` (zero-based of 250).
- Accepted outbound samples: `204`.
- First rejecting joint: `['shoulder']`.
- Last publishable command: `[0.0, -1.8486345381526104, 1.6002409638554216, 0.0, -1.6859437751004016, 0.0]` rad.
- Semantics: `REJECT_WHOLE_GROUP_SAMPLE`; no per-joint clamp.

## Source-history boundary

The original official ALOHA variant used the in-range Sleep vector `[0.0, -1.8, 1.55, 0.0, -1.57, 0.0]`. PR #225 changed only the ROS 2 motor-config Sleep vector to reduce arm drop after torque-off; the ALOHA ViperX URDF limits were not widened in that change. The ROS 1 `main/noetic` branches stopped before this ROS 2 change, so their older value is not an automatic replacement for the pinned `humble` configuration.

## Decision

Preserve the visual run as trajectory evidence, but model the selected real API path explicitly before any hardware comparison. Do not widen USD limits and do not silently replace the pinned humble Sleep vector.

No real robot was contacted and no final/default asset was modified.
