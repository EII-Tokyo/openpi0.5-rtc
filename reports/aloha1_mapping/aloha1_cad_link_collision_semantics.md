# ALOHA1 CAD link collision semantics

- Status: `PARTIAL`
- Links classified: `28`
- Unclassified: `0`
- Final/default asset modified: `false`
- Task 8: `NOT_RUN`

## Classification counts

- `HARD_BLOCKER_CAD_TO_LINK_IDENTITY`: `4`
- `PHYSICAL_CAD_DERIVABLE`: `14`
- `PHYSICAL_EXISTING_VALIDATED_COLLIDER`: `4`
- `VIRTUAL_FRAME_NO_COLLIDER`: `6`

The six `ee_arm_link`, `fingers_link`, and `ee_gripper_link` records are geometry-free helper frames in the pinned URDF. This audit does not invent colliders or remove RigidBodyAPI.

The seven main CAD solids per follower have explicit supplier object identity but still require numerical CAD-to-link registration in Phase 3. `gripper_prop_link` and `gripper_bar_link` do not yet have independently proven CAD subpart identity and remain hard blockers.
