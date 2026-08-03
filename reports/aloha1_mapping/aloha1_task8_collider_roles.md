# ALOHA1 Task 8 collider role audit

- Status: `PASS`
- Candidate link suffixes: `upper_arm_link`
- Baseline static poses: `24`
- Baseline swept waypoints: `809`
- Candidate promoted: `false`

| Robot/link | Role | Source pieces | Active meshes | Simplification |
|---|---|---:|---:|---|
| follower_left_base_link | environment_clearance_critical | 1 | 1 | NONE |
| follower_left_shoulder_link | environment_clearance_critical | 1 | 1 | NONE |
| follower_left_upper_arm_link | environment_clearance_critical | 4 | 4 | SINGLE_HULL_CANDIDATE_PENDING_CANDIDATE_STATIC_AND_SWEPT_REGRESSION |
| follower_left_upper_forearm_link | environment_clearance_critical | 1 | 1 | NONE |
| follower_left_lower_forearm_link | environment_clearance_critical | 1 | 1 | NONE |
| follower_left_wrist_link | environment_clearance_critical | 0 | 1 | NONE |
| follower_left_gripper_link | task_contact_critical | 9 | 9 | NONE |
| follower_left_gripper_bar_link | task_contact_critical | 0 | 0 | NONE |
| follower_left_gripper_prop_link | task_contact_critical | 0 | 1 | NONE |
| follower_left_left_finger_link | task_contact_critical | 1 | 1 | NONE |
| follower_left_right_finger_link | task_contact_critical | 1 | 1 | NONE |
| follower_left_ee_arm_link | non_contact_visual_only | 0 | 0 | NONE |
| follower_left_ee_gripper_link | non_contact_visual_only | 0 | 0 | NONE |
| follower_left_fingers_link | non_contact_visual_only | 0 | 0 | NONE |
| follower_right_base_link | environment_clearance_critical | 1 | 1 | NONE |
| follower_right_shoulder_link | environment_clearance_critical | 1 | 1 | NONE |
| follower_right_upper_arm_link | environment_clearance_critical | 4 | 4 | SINGLE_HULL_CANDIDATE_PENDING_CANDIDATE_STATIC_AND_SWEPT_REGRESSION |
| follower_right_upper_forearm_link | environment_clearance_critical | 1 | 1 | NONE |
| follower_right_lower_forearm_link | environment_clearance_critical | 1 | 1 | NONE |
| follower_right_wrist_link | environment_clearance_critical | 0 | 1 | NONE |
| follower_right_gripper_link | task_contact_critical | 9 | 9 | NONE |
| follower_right_gripper_bar_link | task_contact_critical | 0 | 0 | NONE |
| follower_right_gripper_prop_link | task_contact_critical | 0 | 1 | NONE |
| follower_right_left_finger_link | task_contact_critical | 1 | 1 | NONE |
| follower_right_right_finger_link | task_contact_critical | 1 | 1 | NONE |
| follower_right_ee_arm_link | non_contact_visual_only | 0 | 0 | NONE |
| follower_right_ee_gripper_link | non_contact_visual_only | 0 | 0 | NONE |
| follower_right_fingers_link | non_contact_visual_only | 0 | 0 | NONE |

Only the two `upper_arm_link` instances enter the diagnostic candidate. Their four supplier-CAD components may be represented by one outer convex hull only if candidate static and swept collision regressions pass. Gripper, fingers, Bottle500 and the tabletop support region are unchanged.
