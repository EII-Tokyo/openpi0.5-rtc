# ALOHA1 official-source collider surface certificate

- Status: **PARTIAL**
- Numerical coverage: **COMPLETE_NUMERICAL**
- Acceptance: **HARD_BLOCKER_ERROR_BUDGET_NOT_DEFINED**
- Isaac runtime started: `false`
- Final/default asset modified: `false`

| Link | Components/hulls | Hull/source volume | source→hull max (m) | hull→source max (m) |
|---|---:|---:|---:|---:|
| `base_link` | 1 | 1.66777 | 0.0459624 | 0.0317542 |
| `shoulder_link` | 1 | 2.16218 | 0.0253224 | 0.0271684 |
| `upper_arm_link` | 4 | 3.6801 | 0.0307681 | 0.0536846 |
| `upper_forearm_link` | 1 | 2.31849 | 0.0205557 | 0.0305995 |
| `lower_forearm_link` | 5 | 1.11733 | 0.0106356 | 0.0113755 |
| `wrist_link` | 3 | 2.52044 | 0.0124422 | 0.0252788 |
| `gripper_link` | 12 | 1.22295 | 0.00883482 | 0.00817152 |
| `gripper_prop_link` | 1 | 1.26767 | 0.00453784 | 0.00601245 |
| `gripper_bar_link` | 6 | 3.07834 | 0.0211547 | 0.0156883 |
| `left_finger_link` | 1 | 1.58093 | 0.0104828 | 0.0139306 |
| `right_finger_link` | 1 | 1.58093 | 0.0106951 | 0.0139306 |

Every physical link now has a deterministic numerical source-to-convex-hull record. This does not automatically make every hull acceptable: the finite-sample surface errors and volume growth require a task-derived or official error budget. No tolerance was fitted from successful grasp videos.
