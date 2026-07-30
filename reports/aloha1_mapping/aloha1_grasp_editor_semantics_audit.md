# ALOHA 1 Grasp Editor semantics audit

- Status: `PARTIAL`
- Classification: `COORDINATE_CONTRACT_VERIFIED_NATIVE_SIMULATE_UNSUITABLE_EXTERNAL_SKIP_SIM_EXPORTABLE_MIMIC_BLOCKED`
- Frozen Stage: `/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/table_support_alignment/1.0/aloha1_table_support_aligned_workcell.usda` (`2b3f76365ed67532f478d995ae859a88b5639975ac07cb7ac8a53ac679e8205c`)
- Task 8: `NOT_RUN`

## Confirmed coordinate contract

- Stored grasp transform: `T_O_G`.
- Application: `T_W_G = T_W_O @ T_O_G`.
- Inverse authoring placement: `T_W_O = T_W_G @ inverse(T_O_G)`.
- Bottle frame: bottom center, local `+Z` toward the mouth.
- Canonical gripper/IK frame: `follower_left_ee_gripper_link`.

## Corrected Grasp Editor setting

- `Position When Open`: `0.057 m`.
- `Position When Closed`: `0.021 m` (verified legal fully-closed lower limit).
- `0.048316874538855845 m` is a CAD contact candidate, not the fully-closed setting.

## Native SIMULATE finding

- Native SIMULATE is `FAIL` as a sole ALOHA acceptance gate.
- With bilateral bottle contact, mimic residual: `0.020771507 m`.
- With zero bottle contact, mimic residual: `0.001420334 m`.
- Amplification ratio: `14.624`.
- The no-contact control still returned native success, proving a false positive when contact reports are omitted.

The Isaac Sim 5.1 tutorial explicitly supports an external programmatic closing trajectory followed by **Skip Sim** for heavily coupled grippers.

## External close + Skip Sim result

- Bilateral runtime contact: `PASS` (125 contact points).
- Native raw Skip Sim export: `PASS`.
- Derived pregrasp export: `PASS` (`DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING`).
- Runtime mimic residual: `0.001779459 m` (gate `0.001000 m`).
- Overall external path: `FAIL_MIMIC_ACCURACY`; IK promotion remains forbidden.

## Evidence

- Native raw YAML: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260730-aloha1-grasp-editor-ik-evidence/frame_contract_correction/native_grasp_editor_variant_b_fully_closed_contact/grasp_editor_variant_b_native_raw.yaml`
- Native raw YAML SHA-256: `8a477af435b24e7c93a2d60ddfea6f1213097967f52a124b001c73a2a60b41fc`
- Official 5.1 tutorial: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_simulation/grasp_editor.html
- Full JSON report: `/home/eii/project/openpi0.5-rtc-reward-learning/reports/aloha1_mapping/aloha1_grasp_editor_semantics_audit.json`

## Open blockers

- `HARD_BLOCKER_MIMIC_ACCURACY`
- `HARD_BLOCKER_UNCALIBRATED_MIMIC_PARAMETERS`
- IK: `NOT_RUN`
- Five random-bottle videos: `NOT_RUN`
