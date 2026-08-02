# ALOHA1 Task 7 helper-link collider audit

- Status: `PASS`
- Findings: `8`
- Classes: `{"PHYSICAL_LINK_REQUIRES_COLLIDER": 2, "VIRTUAL_HELPER_SHOULD_NOT_BE_RIGID_BODY": 6}`
- Frozen/default assets modified: `false`
- Task 8: `NOT_RUN`

| Follower | Link | Source geometry | Class | Existing active collider | Fixed-group coverage |
|---|---|---:|---|---:|---:|
| follower_left | `ee_arm_link` | V=0/C=0 | `VIRTUAL_HELPER_SHOULD_NOT_BE_RIGID_BODY` | 0 | False |
| follower_left | `ee_gripper_link` | V=0/C=0 | `VIRTUAL_HELPER_SHOULD_NOT_BE_RIGID_BODY` | 0 | False |
| follower_left | `fingers_link` | V=0/C=0 | `VIRTUAL_HELPER_SHOULD_NOT_BE_RIGID_BODY` | 0 | False |
| follower_left | `gripper_bar_link` | V=1/C=1 | `PHYSICAL_LINK_REQUIRES_COLLIDER` | 0 | True |
| follower_right | `ee_arm_link` | V=0/C=0 | `VIRTUAL_HELPER_SHOULD_NOT_BE_RIGID_BODY` | 0 | False |
| follower_right | `ee_gripper_link` | V=0/C=0 | `VIRTUAL_HELPER_SHOULD_NOT_BE_RIGID_BODY` | 0 | False |
| follower_right | `fingers_link` | V=0/C=0 | `VIRTUAL_HELPER_SHOULD_NOT_BE_RIGID_BODY` | 0 | False |
| follower_right | `gripper_bar_link` | V=1/C=1 | `PHYSICAL_LINK_REQUIRES_COLLIDER` | 0 | True |

The six empty fixed-frame links have no source visual/collision geometry; inventing colliders is prohibited. The two gripper-bar links are different: the pinned URDF contains a real bar mesh and collider, while the CAD diagnostic deactivates that collider because supplier Part__Feature006 is already authored as one compound collider for the fixed gripper+bar group. The literal validator finding is therefore reproduced, but the correct repair cannot be chosen until collider ownership and fixed-body topology are tested separately.
