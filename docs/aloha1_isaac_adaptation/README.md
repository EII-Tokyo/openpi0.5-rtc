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
- [27 Phase 21 native wrapper arm-only qpos replay](27_phase21_arm_qpos_replay_2026-07-18.md)
- [28 Phase 22 native wrapper arm-only qpos replay batch](28_phase22_arm_qpos_replay_batch_2026-07-18.md)
- [29 Phase 23 native dynamic tracking smoke](29_phase23_native_dynamic_tracking_smoke_2026-07-18.md)
- [30 Phase 24 native qpos-next target tracking](30_phase24_native_qpos_target_tracking_2026-07-18.md)
- [31 Phase 25 native single-joint dynamic response](31_phase25_native_single_joint_response_2026-07-18.md)
- [32 Phase 26 minimal drive and native failure isolation](32_phase26_minimal_drive_and_native_failure_isolation_2026-07-18.md)
- [33 Phase 27 runtime collision composition](33_phase27_runtime_collision_composition_2026-07-18.md)
- [34 Phase 28 physics property comparison](34_phase28_physics_property_comparison_2026-07-18.md)
- [35 Phase 29 defaultPrim composition trap](35_phase29_default_prim_composition_trap_2026-07-18.md)
- [36 Phase 30 sublayer runtime composition](36_phase30_sublayer_runtime_composition_2026-07-18.md)
- [37 Phase 31/32 visual reference repair attempts](37_phase31_32_visual_reference_repair_attempts_2026-07-18.md)
- [38 Phase 33 clean runtime asset](38_phase33_clean_runtime_asset_2026-07-18.md)
- [39 Phase 34 clean stage qpos replay](39_phase34_clean_stage_qpos_replay_2026-07-18.md)
- [40 Phase 35 clean stage single-joint response](40_phase35_clean_stage_single_joint_response_2026-07-18.md)
- [41 Phase 36 collision isolation single-joint response](41_phase36_collision_isolation_single_joint_response_2026-07-18.md)
- [42 Phase 37 clean collision prim audit](42_phase37_clean_collision_prim_audit_2026-07-18.md)
- [43 Phase 38 controller runtime stage](43_phase38_controller_runtime_stage_2026-07-18.md)
- [44 Phase 39 link visual proxy candidate audit](44_phase39_link_visual_proxy_candidate_audit_2026-07-18.md)
- [45 Phase 40 bbox proxy runtime stage](45_phase40_bbox_proxy_runtime_stage_2026-07-18.md)
- [46 Phase 41 gripper DOF smoke](46_phase41_gripper_dof_smoke_2026-07-18.md)
- [47 Phase 42 gripper proxy gap](47_phase42_gripper_proxy_gap_2026-07-18.md)
- [48 Collision repair research](48_collision_repair_research_2026-07-18.md)
- [49 Phase 43 gripper passive contact](49_phase43_gripper_passive_contact_2026-07-18.md)
- [50 Phase 44 gripper contact runtime inspection](50_phase44_gripper_contact_runtime_inspection_2026-07-18.md)
- [51 Phase 45 fingertip-pad proxy](51_phase45_fingertip_pad_proxy_2026-07-18.md)
- [52 Phase 46 USD-authored proxy offsets](52_phase46_usd_authored_proxy_offsets_2026-07-18.md)
- [53 Phase 47 closure and object-size isolation](53_phase47_closure_and_object_size_isolation_2026-07-18.md)
- [54 Phase 48 first contact pair trace](54_phase48_first_contact_pair_trace_2026-07-18.md)
- [55 Phase 49 raw USD contact stability](55_phase49_raw_usd_contact_stability_2026-07-18.md)
- [56 Phase 50 task-shape proxy contact](56_phase50_task_shape_proxy_contact_2026-07-18.md)
- [57 Phase 51 bottle-proxy contact gate](57_phase51_bottle_proxy_contact_2026-07-18.md)
- [58 Phase 52 Bottle500 USD contact gate](58_phase52_bottle500_usd_contact_2026-07-18.md)
- [59 Phase 53 HDF5 gripper-replay contact gate](59_phase53_hdf5_gripper_replay_contact_2026-07-18.md)
- [60 Phase 54 HDF5 left-arm + gripper replay contact gate](60_phase54_hdf5_left_arm_gripper_replay_contact_2026-07-18.md)
- [61 Phase 55 HDF5 replay tracking error](61_phase55_hdf5_replay_tracking_error_2026-07-18.md)
- [62 Phase 56 gravity replay failure](62_phase56_gravity_replay_failure_2026-07-18.md)
- [63 Phase 57 gravity already-grasped replay](63_phase57_gravity_already_grasped_replay_2026-07-18.md)

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
