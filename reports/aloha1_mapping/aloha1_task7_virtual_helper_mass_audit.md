# ALOHA1 Task 7 virtual-helper mass audit

- Status: `PASS`
- Uncompensated collapse allowed: `false`
- Final/default assets modified: `false`
- Task 8: `NOT_RUN`
- Physical calibration: `SOURCE_AUTHORED_PLACEHOLDER_NOT_PHYSICALLY_VERIFIED`

| Follower | Helper | Transfer target | Mass (kg) | Diagonal inertia (kg m^2) |
|---|---|---|---:|---|
| `follower_left` | `/vx300s_left/follower_left_ee_arm_link` | `/vx300s_left/follower_left_gripper_link` | 0.00100000005 | `[9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05]` |
| `follower_left` | `/vx300s_left/follower_left_fingers_link` | `/vx300s_left/follower_left_gripper_bar_link` | 0.00100000005 | `[9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05]` |
| `follower_left` | `/vx300s_left/follower_left_ee_gripper_link` | `/vx300s_left/follower_left_gripper_bar_link` | 0.00100000005 | `[9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05]` |
| `follower_right` | `/vx300s_right/follower_right_ee_arm_link` | `/vx300s_right/follower_right_gripper_link` | 0.00100000005 | `[9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05]` |
| `follower_right` | `/vx300s_right/follower_right_fingers_link` | `/vx300s_right/follower_right_gripper_bar_link` | 0.00100000005 | `[9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05]` |
| `follower_right` | `/vx300s_right/follower_right_ee_gripper_link` | `/vx300s_right/follower_right_gripper_bar_link` | 0.00100000005 | `[9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05]` |

The first topology-collapse candidate is diagnostic only. Its small runtime trace change is consistent with removing these authored masses. A promotable candidate must preserve total mass, COM and inertia through an exact rigid-body aggregation, then rerun validator and runtime regressions.

The USD API returns `[-inf, -inf, -inf]` for an unauthored center-of-mass attribute. This is retained as raw readback, not interpreted as a physical COM. The generated URDF omits the inertial origin on these helper links, so the effective local origin is recorded as `[0, 0, 0]` under URDF semantics.
