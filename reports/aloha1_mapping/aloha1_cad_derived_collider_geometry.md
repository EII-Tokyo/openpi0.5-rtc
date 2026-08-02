# ALOHA1 CAD-derived collider geometry

- Status: `PARTIAL`
- Profile: `CAD_SUBPART_COMPOUND_CONVEX_HULL`
- Two-run determinism: `PASS`
- Final/default collider modified: `false`
- Task 8: `NOT_RUN`

| Robot/link | Source | Result | Pieces | Triangles | Registration |
|---|---|---|---:|---:|---|
| follower_left_base_link | Part__Feature | PASS | 1 | 2430 | URDF_ZERO_POSE_FK_INVERSE_WITH_VERIFIED_CAD_AXIS_ROTATION |
| follower_left_shoulder_link | Part__Feature001 | PASS | 1 | 1756 | URDF_ZERO_POSE_FK_INVERSE_WITH_VERIFIED_CAD_AXIS_ROTATION |
| follower_left_upper_arm_link | Part__Feature002 | PASS | 4 | 3682 | URDF_ZERO_POSE_FK_INVERSE_WITH_VERIFIED_CAD_AXIS_ROTATION |
| follower_left_upper_forearm_link | Part__Feature003 | PASS | 1 | 986 | URDF_ZERO_POSE_FK_INVERSE_WITH_VERIFIED_CAD_AXIS_ROTATION |
| follower_left_lower_forearm_link | Part__Feature004 | PASS | 1 | 1160 | URDF_ZERO_POSE_FK_INVERSE_WITH_VERIFIED_CAD_AXIS_ROTATION |
| follower_left_wrist_link | Part__Feature005 | HARD_BLOCKER_INVALID_BREP | 0 | 0 | None |
| follower_left_gripper_link | Part__Feature006 | PASS | 9 | 8522 | VERIFIED_GRIPPER_PLANAR_DATUM_REGISTRATION |
| follower_right_base_link | Part__Feature | PASS | 1 | 2430 | URDF_ZERO_POSE_FK_INVERSE_WITH_VERIFIED_CAD_AXIS_ROTATION |
| follower_right_shoulder_link | Part__Feature001 | PASS | 1 | 1756 | URDF_ZERO_POSE_FK_INVERSE_WITH_VERIFIED_CAD_AXIS_ROTATION |
| follower_right_upper_arm_link | Part__Feature002 | PASS | 4 | 3682 | URDF_ZERO_POSE_FK_INVERSE_WITH_VERIFIED_CAD_AXIS_ROTATION |
| follower_right_upper_forearm_link | Part__Feature003 | PASS | 1 | 986 | URDF_ZERO_POSE_FK_INVERSE_WITH_VERIFIED_CAD_AXIS_ROTATION |
| follower_right_lower_forearm_link | Part__Feature004 | PASS | 1 | 1160 | URDF_ZERO_POSE_FK_INVERSE_WITH_VERIFIED_CAD_AXIS_ROTATION |
| follower_right_wrist_link | Part__Feature005 | HARD_BLOCKER_INVALID_BREP | 0 | 0 | None |
| follower_right_gripper_link | Part__Feature006 | PASS | 9 | 8522 | VERIFIED_GRIPPER_PLANAR_DATUM_REGISTRATION |
| follower_left_left_finger_link | Part__Feature007 | PASS | 1 | 1662 | VERIFIED_EXISTING_SUPPLIER_CAD_REGISTRATION |
| follower_left_right_finger_link | Part__Feature008 | PASS | 1 | 1662 | VERIFIED_EXISTING_SUPPLIER_CAD_REGISTRATION |
| follower_right_left_finger_link | Part__Feature007 | PASS | 1 | 1662 | VERIFIED_EXISTING_SUPPLIER_CAD_REGISTRATION |
| follower_right_right_finger_link | Part__Feature008 | PASS | 1 | 1662 | VERIFIED_EXISTING_SUPPLIER_CAD_REGISTRATION |

`wrist_link` remains blocked because supplier `Part__Feature005` fails the B-Rep validity gate. No repair was applied. `gripper_prop_link` and `gripper_bar_link` remain identity blockers; no collider was invented for them.

Surface-distance values compare different supplier/URDF revisions and are diagnostic only; they do not select orientation or hide registration failures.
