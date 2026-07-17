# ALOHA1 Isaac Sim Adaptation Research

This directory records the evidence base for adapting Isaac Sim assets and controller examples to the user's real ALOHA1 setup.

The main conclusion is conservative:

- Trossen AI Isaac is a strong engineering starting point for Isaac Sim 5.1, USD organization, controllers, and Isaac Lab task structure.
- Google DeepMind MuJoCo Menagerie ALOHA is a strong reference for ALOHA2/MJCF modeling choices.
- Neither source is automatically the user's ALOHA1 truth. ALOHA1 joint semantics, gripper semantics, camera extrinsics, controller signals, and real workcell geometry must be validated explicitly.
- Phase 14 shows that the current Trossen-backed scaffold should not be forced into an ALOHA1 controller target by scalar sign/offset fitting. Rebuild around trusted ALOHA1 kinematics and reuse Trossen's framework patterns selectively.

## Documents

- [01 Google MuJoCo ALOHA assets](01_google_mujoco_aloha_assets_2026-07-17.md)
- [02 Trossen AI Isaac](02_trossen_ai_isaac_2026-07-17.md)
- [03 Current project verified facts](03_current_project_verified_facts_2026-07-17.md)
- [04 ALOHA1 vs ALOHA2/Trossen adaptation matrix](04_aloha1_vs_aloha2_trossen_adaptation_matrix_2026-07-17.md)
- [05 Execution plan](05_execution_plan_2026-07-17.md)
- [06 Phase 1 asset comparison result](06_phase1_asset_comparison_result_2026-07-17.md)
- [07 Phase 2 runtime decision](07_phase2_runtime_decision_2026-07-17.md)
- [08 Phase 3 scaffold runtime result](08_phase3_scaffold_runtime_result_2026-07-17.md)
- [09 Phase 4 real ALOHA1 joint signal probe](09_phase4_real_aloha1_joint_signal_probe_2026-07-17.md)
- [10 Phase 5 one-joint static validation](10_phase5_one_joint_static_validation_2026-07-17.md)
- [11 Phase 6 affine candidate inference](11_phase6_affine_candidate_inference_2026-07-17.md)
- [12 Phase 6 reference FK smoke](12_phase6_reference_fk_smoke_2026-07-17.md)
- [13 Phase 7 Trossen FK candidate check](13_phase7_trossen_fk_candidate_check_2026-07-17.md)
- [14 Phase 8 FK mapping search](14_phase8_fk_mapping_search_2026-07-17.md)
- [15 Phase 9 FK mapping holdout](15_phase9_fk_mapping_holdout_2026-07-17.md)
- [16 Phase 10 full-dataset mapping limits](16_phase10_full_dataset_mapping_limits_2026-07-18.md)
- [17 Phase 11 orientation consistency](17_phase11_orientation_consistency_2026-07-18.md)
- [18 Phase 12 Trossen terminal body scan](18_phase12_trossen_terminal_body_scan_2026-07-18.md)
- [19 Phase 13 joint schema comparison](19_phase13_joint_schema_comparison_2026-07-18.md)
- [20 Phase 14 orientation-aware mapping](20_phase14_orientation_aware_mapping_2026-07-18.md)
- [21 Phase 15 ALOHA1 native source audit](21_phase15_aloha1_native_source_audit_2026-07-18.md)
- [22 Phase 16 URDF importer mesh probe](22_phase16_urdf_importer_mesh_probe_2026-07-18.md)
- [23 Phase 17 physics layer wrapper](23_phase17_physics_layer_wrapper_2026-07-18.md)
- [24 Phase 18 runtime articulation validation](24_phase18_runtime_articulation_validation_2026-07-18.md)
- [25 Phase 19 native asset candidate](25_phase19_native_asset_candidate_2026-07-18.md)
- [26 Phase 20 DOF / drive / limit validation](26_phase20_dof_drive_limits_2026-07-18.md)

## Operating Rule

Any future implementation step for ALOHA1 Isaac adaptation should cite one of these documents or add a new dated investigation document first.

Do not proceed from visual similarity alone. A visible robot mesh is not enough. The minimum gates are:

1. Asset identity and unit system.
2. Articulation DOF names, order, limits, and signs.
3. Gripper command and measured opening semantics.
4. End-effector frame and grasp transform semantics.
5. Collision, mass, drive, damping, and contact material semantics.
6. Camera intrinsics/extrinsics and image encoding.
7. Replay or controller validation against real ALOHA1 data.
