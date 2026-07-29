# Supplier-CAD follower_right robot-local asset

- Status: `PASS`
- Scope: `ROBOT_LOCAL_DIAGNOSTIC_NOT_WORKCELL_PLACEMENT`
- Root prim: `/follower_right`
- Robot product: `/follower_right/vx300s_right`
- Articulation count: `1`
- DOF order: `waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, gripper, left_finger, right_finger`
- Workcell placement: `NOT_AUTHORED`
- Geometry mirroring: `false`
- Task 8: `NOT_RUN`

The robot structure is an explicit reference to the pinned `follower_right` import. Supplier embedded-v2 handed finger geometry is composed in separate diagnostic geometry, configuration, and physics layers. No left-asset string rename, robot mirroring, final-collider promotion, or workcell transform is used.

## Remaining blocker

- `HARD_BLOCKER_FOLLOWER_RIGHT_WORKCELL_INSTALL_TRANSFORM`
