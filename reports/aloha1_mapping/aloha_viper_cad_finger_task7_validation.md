# Supplier-CAD follower_left Task 7 validation

- Status: `PARTIAL`
- Stage: `/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/supplier_cad_follower_left/1.6/supplier_cad_follower_left.usda`
- Stage SHA-256: `232ea1f61dc07f391baf7497b0cf6c2455593f9655ae9b3f541fde81c8ef73ad`
- Task 5 static hold: `20/20 PASS`, maximum drop `0.000453919172 m`
- Repeat validation signature: `34c2c067682987edac88049f60e0b69511fe0c008ddb1cf95f5c2b8f3085139b`
- Task 8: `NOT_RUN`

| Check | Status |
|---|---|
| approved_source_hash_immutable | PASS |
| task5_diagnostic_stage_hash_immutable | PASS |
| robot_scoped_diagnostic_hash | PASS |
| robot_schema_diagnostic_hash | PASS |
| angular_controlled_tessellation | PASS |
| external_references_resolve | PASS |
| one_robot_articulation_root | PASS |
| dof_name_and_order | PASS |
| all_nonfixed_joints_have_drive_or_mimic | PASS |
| finite_positive_max_velocity_and_force | PASS |
| initial_joint_state_matches_drive_target | PASS |
| mass_and_inertia_finite_positive | PASS |
| first_frame_jump_and_static_structure | PASS |
| one_joint_direction_and_range | PARTIAL |
| mimic_or_symmetric_control_mapping | PARTIAL |
| initial_overlap | PARTIAL |
| bilateral_contact_and_static_hold | PASS |
| screenshot_visual_review | PASS |
| task5_repeat_determinism | PASS |
| IsaacSim.PhysicsRules | PARTIAL |
| IsaacSim.RobotRules | PARTIAL |
| IsaacSim.SimReadyAssetRules | PASS |

| Official category | Status | Blocking | Warnings |
|---|---|---:|---:|
| IsaacSim.PhysicsRules | PARTIAL | 0 | 9 |
| IsaacSim.RobotRules | PARTIAL | 0 | 4 |
| IsaacSim.SimReadyAssetRules | PASS | 0 | 0 |

## HARD_BLOCKER

- `HARD_BLOCKER_APPROVED_STAGE_CONTAINS_FOLLOWER_LEFT_ONLY`
- `HARD_BLOCKER_NO_USER_APPROVED_SUPPLIER_STAGE_LIFT_TRAJECTORY`
- `HARD_BLOCKER_UNCALIBRATED_FINGER_BOTTLE_FRICTION`
- `HARD_BLOCKER_INCOMPLETE_BOTTLE_GEOMETRY_AND_INERTIA`

PhysicsRules and SimReadyAssetRules run on the isolated physical supplier-CAD follower_left diagnostic. RobotRules runs on the schema-only wrapper of the same robot hierarchy so diagnostic physics opinions are not misclassified as prohibited robot schema overrides. Task 5 runtime evidence remains on the immutable bottle workcell. This does not promote the collider or configuration, claim calibrated dynamics, validate follower_right, or run a lift trajectory.
