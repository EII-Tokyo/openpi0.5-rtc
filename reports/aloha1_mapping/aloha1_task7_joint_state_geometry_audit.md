# ALOHA1 Task 7 joint-state geometry audit

- Status: `PASS`
- Findings: `10`
- Candidate authoring: `NOT_RUN`
- Task 8: `NOT_RUN`

| Follower | Joint | Authored | Geometry-derived | Translation residual | Quaternion residual |
|---|---|---:|---:|---:|---:|
| follower_left | `elbow` | 66.4631042 | 0 | 0 | 0 |
| follower_left | `left_finger` | 0.0223900005 | 0 | 0 | 0 |
| follower_left | `right_finger` | -0.0223900005 | 0 | 0 | 0 |
| follower_left | `shoulder` | -55.0039482 | 0 | 7.4505806e-09 | 0 |
| follower_left | `wrist_angle` | -17.1887341 | 0 | 7.4505806e-09 | 0 |
| follower_right | `elbow` | 66.4631042 | 0 | 0 | 0 |
| follower_right | `left_finger` | 0.0223900005 | 0 | 0 | 0 |
| follower_right | `right_finger` | -0.0223900005 | 0 | 0 | 0 |
| follower_right | `shoulder` | -55.0039482 | 0 | 7.4505806e-09 | 0 |
| follower_right | `wrist_angle` | -17.1887341 | 0 | 7.4505806e-09 | 0 |

The geometry-derived values reproduce the installed 1.1.0 rule's body-transform equation only. No state, drive, body transform, or asset was authored. Runtime/home/source comparison remains required before any candidate may be built.
