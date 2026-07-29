# ALOHA1 Task 7A helper-link semantics

- Status: `PASS`
- Frozen Stage unchanged: `True`
- Official findings covered: `6`
- Helper records: `6`

## Result

The six findings are real literal Isaac Sim 5.1 `RigidBodyHasCollider` failures. The pinned Xacro and generated URDFs define no visual or collision geometry for these links. `ee_arm_link` and `fingers_link` are geometry-free kinematic helper frames; `ee_gripper_link` is a fixed end-effector frame alias. Their 1 g inertial blocks do not define a physical shape.

No collider was invented and no `RigidBodyAPI` was removed. Either change could alter articulation semantics and requires a separate source-backed promotion candidate plus regression.

## Per-link evidence

| Prim | Source semantic class | RigidBodyAPI | Descendant colliders |
|---|---|---:|---:|
| `/World/follower_left/vx300s_left/follower_left_ee_arm_link` | `VIRTUAL_KINEMATIC_HELPER` | `True` | `0` |
| `/World/follower_left/vx300s_left/follower_left_fingers_link` | `VIRTUAL_KINEMATIC_HELPER` | `True` | `0` |
| `/World/follower_left/vx300s_left/follower_left_ee_gripper_link` | `FIXED_FRAME_ALIAS` | `True` | `0` |
| `/World/follower_right/vx300s_right/follower_right_ee_arm_link` | `VIRTUAL_KINEMATIC_HELPER` | `True` | `0` |
| `/World/follower_right/vx300s_right/follower_right_fingers_link` | `VIRTUAL_KINEMATIC_HELPER` | `True` | `0` |
| `/World/follower_right/vx300s_right/follower_right_ee_gripper_link` | `FIXED_FRAME_ALIAS` | `True` | `0` |

## Acceptance boundary

- Runtime control: these findings do not invalidate measured DOF motion, target/readback, or deterministic swept-path results.
- Asset promotion: remains `PARTIAL`; literal official failures remain unsuppressed.
- Supplier CAD maps geometry to the handed finger links, not to these six abstract helper frames.
- Task 7B: `NOT_RUN`.
- Task 8: `NOT_RUN`.
