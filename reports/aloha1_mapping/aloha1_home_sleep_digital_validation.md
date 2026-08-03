# ALOHA1 Home/Sleep digital validation

- Status: `FAIL`
- Classification: `OFFICIAL_SLEEP_TARGET_OUTSIDE_FROZEN_JOINT_LIMITS`
- Numeric repeatability: `PASS`
- Fresh Isaac processes: `2`
- Real preflight: `NOT_RUN_DIGITAL_GATE_FAILED`
- Real execution: `NOT_RUN_DIGITAL_GATE_FAILED`
- Real execution authorized: `false`

## Limit conflicts

| Joint | Official Sleep | Frozen lower | Frozen upper | Violation |
|---|---:|---:|---:|---:|
| `shoulder` | -2.050000 | -1.850049 | 1.256637 | 0.199951 |
| `elbow` | 1.700000 | -1.762782 | 1.605703 | 0.094297 |
| `wrist_angle` | -2.000000 | -1.867502 | 2.234021 | 0.132498 |

The official ALOHA Sleep command is preserved exactly. The frozen USD/URDF limits are also preserved exactly. PhysX clamps the three out-of-range targets, so direction, repeatability, stationary bodies, contact absence, and final Home pass, but the Sleep endpoint and legal-target gates fail.

No real-robot command was sent and this report does not authorize one.

## Source boundary

- Pinned exact-model official Sleep: `[0, -2.05, 1.7, 0, -2.0, 0]` rad.
- Local third-party mirror Sleep: `[0, -1.8, 1.55, 0, -1.57, 0]` rad.
- The local mirror differs and is explicitly not treated as official authority.
- A historical read-only robot report also differs; it is retained as project evidence, not used to authorize or generate motion in this run.
