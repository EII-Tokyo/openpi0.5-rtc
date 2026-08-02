# ALOHA1 Task 7 validator rule-scope audit

- Status: `PASS`
- Original blocking issues: `89` (RobotRules 63, PhysicsRules 26)
- Classification counts: `{"INCONCLUSIVE": 3, "TRUE_ASSET_DEFECT": 23, "WRONG_SCOPE": 63}`
- Remaining Task 7-blocking issue records: `26`
- Task 8: `NOT_RUN`

The 63 RobotRules errors are `WRONG_SCOPE`: they were produced by running robot-package rules on the two-robot workcell wrapper. They are not suppressed; the correct standalone left/right package runs are reported separately.

The 26 PhysicsRules records remain individually classified. Applicable literal defects and unresolved validator/runtime conflicts continue to block asset promotion.

| # | Family | Rule | Owner | Classification | Task 7 blocker | Prim |
|---:|---|---|---|---|---:|---|
| 1 | IsaacSim.RobotRules | NoOverrides | diagnostic layer | WRONG_SCOPE | False | Prim </World/environment/worldBody/user_confirmed_table> |
| 2 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_left/vx300s_left/follower_left_base_link/cad_derived_collisions/cad_derived_base_link/piece_000/mesh> |
| 3 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_left/vx300s_left/follower_left_gripper_link/cad_derived_collisions/cad_derived_gripper_link/piece_000/mesh> |
| 4 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_left/vx300s_left/follower_left_gripper_link/cad_derived_collisions/cad_derived_gripper_link/piece_001/mesh> |
| 5 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_left/vx300s_left/follower_left_gripper_link/cad_derived_collisions/cad_derived_gripper_link/piece_002/mesh> |
| 6 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_left/vx300s_left/follower_left_gripper_link/cad_derived_collisions/cad_derived_gripper_link/piece_003/mesh> |
| 7 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_left/vx300s_left/follower_left_gripper_link/cad_derived_collisions/cad_derived_gripper_link/piece_004/mesh> |
| 8 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_left/vx300s_left/follower_left_gripper_link/cad_derived_collisions/cad_derived_gripper_link/piece_005/mesh> |
| 9 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_left/vx300s_left/follower_left_gripper_link/cad_derived_collisions/cad_derived_gripper_link/piece_006/mesh> |
| 10 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_left/vx300s_left/follower_left_gripper_link/cad_derived_collisions/cad_derived_gripper_link/piece_007/mesh> |
| 11 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_left/vx300s_left/follower_left_gripper_link/cad_derived_collisions/cad_derived_gripper_link/piece_008/mesh> |
| 12 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_left/vx300s_left/follower_left_gripper_prop_link/materialized_baseline_fallback/gripper_prop_link/mesh> |
| 13 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_left/vx300s_left/follower_left_lower_forearm_link/cad_derived_collisions/cad_derived_lower_forearm_link/piece_000/mesh> |
| 14 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_left/vx300s_left/follower_left_shoulder_link/cad_derived_collisions/cad_derived_shoulder_link/piece_000/mesh> |
| 15 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_left/vx300s_left/follower_left_upper_arm_link/cad_derived_collisions/cad_derived_upper_arm_link/piece_000/mesh> |
| 16 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_left/vx300s_left/follower_left_upper_arm_link/cad_derived_collisions/cad_derived_upper_arm_link/piece_001/mesh> |
| 17 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_left/vx300s_left/follower_left_upper_arm_link/cad_derived_collisions/cad_derived_upper_arm_link/piece_002/mesh> |
| 18 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_left/vx300s_left/follower_left_upper_arm_link/cad_derived_collisions/cad_derived_upper_arm_link/piece_003/mesh> |
| 19 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_left/vx300s_left/follower_left_upper_forearm_link/cad_derived_collisions/cad_derived_upper_forearm_link/piece_000/mesh> |
| 20 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_left/vx300s_left/follower_left_wrist_link/materialized_baseline_fallback/wrist_link/mesh> |
| 21 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_left/vx300s_left/joints/elbow> |
| 22 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_left/vx300s_left/joints/forearm_roll> |
| 23 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_left/vx300s_left/joints/gripper> |
| 24 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_left/vx300s_left/joints/left_finger> |
| 25 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_left/vx300s_left/joints/right_finger> |
| 26 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_left/vx300s_left/joints/shoulder> |
| 27 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_left/vx300s_left/joints/waist> |
| 28 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_left/vx300s_left/joints/wrist_angle> |
| 29 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_left/vx300s_left/joints/wrist_rotate> |
| 30 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_left> |
| 31 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_right/vx300s_right/follower_right_base_link/cad_derived_collisions/cad_derived_base_link/piece_000/mesh> |
| 32 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_right/vx300s_right/follower_right_gripper_link/cad_derived_collisions/cad_derived_gripper_link/piece_000/mesh> |
| 33 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_right/vx300s_right/follower_right_gripper_link/cad_derived_collisions/cad_derived_gripper_link/piece_001/mesh> |
| 34 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_right/vx300s_right/follower_right_gripper_link/cad_derived_collisions/cad_derived_gripper_link/piece_002/mesh> |
| 35 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_right/vx300s_right/follower_right_gripper_link/cad_derived_collisions/cad_derived_gripper_link/piece_003/mesh> |
| 36 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_right/vx300s_right/follower_right_gripper_link/cad_derived_collisions/cad_derived_gripper_link/piece_004/mesh> |
| 37 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_right/vx300s_right/follower_right_gripper_link/cad_derived_collisions/cad_derived_gripper_link/piece_005/mesh> |
| 38 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_right/vx300s_right/follower_right_gripper_link/cad_derived_collisions/cad_derived_gripper_link/piece_006/mesh> |
| 39 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_right/vx300s_right/follower_right_gripper_link/cad_derived_collisions/cad_derived_gripper_link/piece_007/mesh> |
| 40 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_right/vx300s_right/follower_right_gripper_link/cad_derived_collisions/cad_derived_gripper_link/piece_008/mesh> |
| 41 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_right/vx300s_right/follower_right_gripper_prop_link/materialized_baseline_fallback/gripper_prop_link/mesh> |
| 42 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_right/vx300s_right/follower_right_lower_forearm_link/cad_derived_collisions/cad_derived_lower_forearm_link/piece_000/mesh> |
| 43 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_right/vx300s_right/follower_right_shoulder_link/cad_derived_collisions/cad_derived_shoulder_link/piece_000/mesh> |
| 44 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_right/vx300s_right/follower_right_upper_arm_link/cad_derived_collisions/cad_derived_upper_arm_link/piece_000/mesh> |
| 45 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_right/vx300s_right/follower_right_upper_arm_link/cad_derived_collisions/cad_derived_upper_arm_link/piece_001/mesh> |
| 46 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_right/vx300s_right/follower_right_upper_arm_link/cad_derived_collisions/cad_derived_upper_arm_link/piece_002/mesh> |
| 47 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_right/vx300s_right/follower_right_upper_arm_link/cad_derived_collisions/cad_derived_upper_arm_link/piece_003/mesh> |
| 48 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_right/vx300s_right/follower_right_upper_forearm_link/cad_derived_collisions/cad_derived_upper_forearm_link/piece_000/mesh> |
| 49 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_right/vx300s_right/follower_right_wrist_link/materialized_baseline_fallback/wrist_link/mesh> |
| 50 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_right/vx300s_right/joints/elbow> |
| 51 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_right/vx300s_right/joints/forearm_roll> |
| 52 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_right/vx300s_right/joints/gripper> |
| 53 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_right/vx300s_right/joints/left_finger> |
| 54 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_right/vx300s_right/joints/right_finger> |
| 55 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_right/vx300s_right/joints/shoulder> |
| 56 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_right/vx300s_right/joints/waist> |
| 57 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_right/vx300s_right/joints/wrist_angle> |
| 58 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_right/vx300s_right/joints/wrist_rotate> |
| 59 | IsaacSim.RobotRules | NoOverrides | robot | WRONG_SCOPE | False | Prim </World/follower_right> |
| 60 | IsaacSim.RobotRules | RobotNaming | workcell wrapper | WRONG_SCOPE | False | Stage </home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_z_up_meters_diagnostic.usda> |
| 61 | IsaacSim.RobotRules | RobotSchema | workcell wrapper | WRONG_SCOPE | False | Prim </World> |
| 62 | IsaacSim.RobotRules | RobotSchema | workcell wrapper | WRONG_SCOPE | False | Prim </World> |
| 63 | IsaacSim.RobotRules | RobotSchema | workcell wrapper | WRONG_SCOPE | False | Prim </World> |
| 64 | IsaacSim.PhysicsRules | JointHasCorrectTransformAndState | robot | TRUE_ASSET_DEFECT | True | Prim </World/follower_left/vx300s_left/joints/elbow> |
| 65 | IsaacSim.PhysicsRules | JointHasCorrectTransformAndState | robot | TRUE_ASSET_DEFECT | True | Prim </World/follower_left/vx300s_left/joints/left_finger> |
| 66 | IsaacSim.PhysicsRules | JointHasCorrectTransformAndState | robot | TRUE_ASSET_DEFECT | True | Prim </World/follower_left/vx300s_left/joints/right_finger> |
| 67 | IsaacSim.PhysicsRules | JointHasCorrectTransformAndState | robot | TRUE_ASSET_DEFECT | True | Prim </World/follower_left/vx300s_left/joints/shoulder> |
| 68 | IsaacSim.PhysicsRules | JointHasCorrectTransformAndState | robot | TRUE_ASSET_DEFECT | True | Prim </World/follower_left/vx300s_left/joints/wrist_angle> |
| 69 | IsaacSim.PhysicsRules | JointHasCorrectTransformAndState | robot | TRUE_ASSET_DEFECT | True | Prim </World/follower_right/vx300s_right/joints/elbow> |
| 70 | IsaacSim.PhysicsRules | JointHasCorrectTransformAndState | robot | TRUE_ASSET_DEFECT | True | Prim </World/follower_right/vx300s_right/joints/left_finger> |
| 71 | IsaacSim.PhysicsRules | JointHasCorrectTransformAndState | robot | TRUE_ASSET_DEFECT | True | Prim </World/follower_right/vx300s_right/joints/right_finger> |
| 72 | IsaacSim.PhysicsRules | JointHasCorrectTransformAndState | robot | TRUE_ASSET_DEFECT | True | Prim </World/follower_right/vx300s_right/joints/shoulder> |
| 73 | IsaacSim.PhysicsRules | JointHasCorrectTransformAndState | robot | TRUE_ASSET_DEFECT | True | Prim </World/follower_right/vx300s_right/joints/wrist_angle> |
| 74 | IsaacSim.PhysicsRules | MimicAPICheck | robot | INCONCLUSIVE | True | Prim </World/follower_left/vx300s_left/joints/right_finger> |
| 75 | IsaacSim.PhysicsRules | MimicAPICheck | robot | INCONCLUSIVE | True | Prim </World/follower_right/vx300s_right/joints/right_finger> |
| 76 | IsaacSim.PhysicsRules | RigidBodyHasCollider | diagnostic layer | TRUE_ASSET_DEFECT | True | Prim </World/environment/worldBody/_> |
| 77 | IsaacSim.PhysicsRules | RigidBodyHasCollider | diagnostic layer | TRUE_ASSET_DEFECT | True | Prim </World/environment/worldBody/__1> |
| 78 | IsaacSim.PhysicsRules | RigidBodyHasCollider | robot | TRUE_ASSET_DEFECT | True | Prim </World/follower_left/vx300s_left/follower_left_ee_arm_link> |
| 79 | IsaacSim.PhysicsRules | RigidBodyHasCollider | robot | TRUE_ASSET_DEFECT | True | Prim </World/follower_left/vx300s_left/follower_left_ee_gripper_link> |
| 80 | IsaacSim.PhysicsRules | RigidBodyHasCollider | robot | TRUE_ASSET_DEFECT | True | Prim </World/follower_left/vx300s_left/follower_left_fingers_link> |
| 81 | IsaacSim.PhysicsRules | RigidBodyHasCollider | robot | TRUE_ASSET_DEFECT | True | Prim </World/follower_left/vx300s_left/follower_left_gripper_bar_link> |
| 82 | IsaacSim.PhysicsRules | RigidBodyHasCollider | robot | TRUE_ASSET_DEFECT | True | Prim </World/follower_right/vx300s_right/follower_right_ee_arm_link> |
| 83 | IsaacSim.PhysicsRules | RigidBodyHasCollider | robot | TRUE_ASSET_DEFECT | True | Prim </World/follower_right/vx300s_right/follower_right_ee_gripper_link> |
| 84 | IsaacSim.PhysicsRules | RigidBodyHasCollider | robot | TRUE_ASSET_DEFECT | True | Prim </World/follower_right/vx300s_right/follower_right_fingers_link> |
| 85 | IsaacSim.PhysicsRules | RigidBodyHasCollider | robot | TRUE_ASSET_DEFECT | True | Prim </World/follower_right/vx300s_right/follower_right_gripper_bar_link> |
| 86 | IsaacSim.PhysicsRules | RigidBodyHasMassAPI | workcell wrapper | INCONCLUSIVE | True | None |
| 87 | IsaacSim.PhysicsRules | RigidBodyHasMassAPI | table | TRUE_ASSET_DEFECT | True | Prim </World/environment/worldBody/floor> |
| 88 | IsaacSim.PhysicsRules | RigidBodyHasMassAPI | table | TRUE_ASSET_DEFECT | True | Prim </World/environment/worldBody/floor> |
| 89 | IsaacSim.PhysicsRules | RigidBodyHasMassAPI | table | TRUE_ASSET_DEFECT | True | Prim </World/environment/worldBody/floor> |
