# ALOHA1 Task 7 JointStateAPI physics candidate

- Candidate status: `PASS`
- Task 7: `PARTIAL`
- Task 8: `NOT_RUN`

## follower_left

- Gripper joint: `/follower_left/vx300s_left/joints/gripper`
- Joint type/axis: `RevoluteJoint / angular`
- Baseline blocking findings: `5`
- Candidate blocking findings: `4`
- Removed rule: `JointHasJointStateAPI`
- Remaining: `MimicAPICheck x1`, `RigidBodyHasCollider x3`
- Fresh-process repeat: `PASS`

## follower_right

- Gripper joint: `/follower_right/vx300s_right/joints/gripper`
- Joint type/axis: `RevoluteJoint / angular`
- Baseline blocking findings: `5`
- Candidate blocking findings: `4`
- Removed rule: `JointHasJointStateAPI`
- Remaining: `MimicAPICheck x1`, `RigidBodyHasCollider x3`
- Fresh-process repeat: `PASS`

The candidate authors only `PhysicsJointStateAPI:angular` in a dedicated `_physics.usd` layer. It authors no state or drive values and does not change geometry, colliders, mimic, drives, the source Stage, or final/default assets.

The literal PhysicsRules result remains `FAIL` because the two version-specific mimic findings and six missing-source-collider findings remain unsuppressed. Therefore Task 7 remains `PARTIAL` and Task 8 remains `NOT_RUN`.
