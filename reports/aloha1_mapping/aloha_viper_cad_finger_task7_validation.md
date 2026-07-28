# Supplier-CAD follower_left Task 7 validation

- Status: `FAIL`
- Stage: `/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/cad_finger_task5_bottle/aloha_viperx_supplier_cad_bottle_task5.usda`
- Stage SHA-256: `62697e4b25a7ec82234cc9ebd79d4a6d530a6ead0165519cbd275c0fa3f32178`
- Task 5 static hold: `20/20 PASS`, maximum drop `0.000453919172 m`
- Repeat validation signature: `b52db5132cbd96c311df95f31f577fb65c078f51d1bf5e6b8a75ebe87f1abd82`
- Task 8: `NOT_RUN`

| Check | Status |
|---|---|
| approved_source_hash_immutable | PASS |
| diagnostic_stage_hash | PASS |
| external_references_resolve | PASS |
| one_robot_articulation_root | PASS |
| dof_name_and_order | PASS |
| all_nonfixed_joints_have_drive_or_mimic | PASS |
| finite_positive_max_velocity_and_force | PASS |
| initial_joint_state_matches_drive_target | FAIL |
| mass_and_inertia_finite_positive | FAIL |
| first_frame_jump_and_static_structure | PASS |
| one_joint_direction_and_range | PARTIAL |
| mimic_or_symmetric_control_mapping | PARTIAL |
| initial_overlap | PARTIAL |
| bilateral_contact_and_static_hold | PASS |
| screenshot_visual_review | PASS |
| task5_repeat_determinism | PASS |
| IsaacSim.PhysicsRules | FAIL |
| IsaacSim.RobotRules | FAIL |
| IsaacSim.SimReadyAssetRules | PASS |

| Official category | Status | Blocking | Warnings |
|---|---|---:|---:|
| IsaacSim.PhysicsRules | FAIL | 17 | 17 |
| IsaacSim.RobotRules | FAIL | 4 | 5 |
| IsaacSim.SimReadyAssetRules | PASS | 0 | 0 |

## HARD_BLOCKER

- `HARD_BLOCKER_APPROVED_STAGE_CONTAINS_FOLLOWER_LEFT_ONLY`
- `HARD_BLOCKER_NO_USER_APPROVED_SUPPLIER_STAGE_LIFT_TRAJECTORY`
- `HARD_BLOCKER_UNCALIBRATED_FINGER_BOTTLE_FRICTION`
- `HARD_BLOCKER_INCOMPLETE_BOTTLE_GEOMETRY_AND_INERTIA`
- `HARD_BLOCKER_PRODUCTION_ANGULAR_TESSELLATION`

This validates only the isolated supplier-CAD follower_left diagnostic. It does not promote the collider/configuration, claim calibrated dynamics, validate follower_right, or run a lift trajectory.
