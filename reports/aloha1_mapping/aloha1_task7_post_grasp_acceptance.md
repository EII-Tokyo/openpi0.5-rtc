# ALOHA1 Task 7 post-grasp acceptance

- Runtime/grasp acceptance: `PASS`
- Asset-promotion readiness: `PARTIAL`
- Literal NVIDIA official-rule status: `FAIL`
- Task 7 aggregate: `PARTIAL`
- Task 8: `NOT_RUN`

## Runtime and grasp gates

- `runtime_control`: `PASS`
- `workcell_physics`: `PASS`
- `aloha_6dof_ik_correspondence`: `PASS`
- `table_support_alignment`: `PASS`
- `static_bottle_hold`: `PASS`
- `dynamic_five_pose_grasp`: `PASS`
- `visual_model_review`: `PASS`
- `user_confirmation`: `PASS`

The five-pose grasp is machine, visual-model and user `PASS`. This does not make the robot package SimReady. The Task 7 aggregate remains `PARTIAL` because literal NVIDIA rule findings keep asset-promotion readiness `PARTIAL`.

The table-aligned Stage composes the frozen signal Stage and changes only tabletop translation. Robot geometry, colliders, drives and physics parameters are unchanged.

No real robot or 192.168.1.103 access occurred. Task 8 remains `NOT_RUN`.
