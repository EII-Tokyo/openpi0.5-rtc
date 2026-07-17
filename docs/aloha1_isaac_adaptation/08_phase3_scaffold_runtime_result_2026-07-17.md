# Phase 3 Scaffold Runtime Result - 2026-07-17

## Result

The first Trossen-backed ALOHA1 scaffold passes the Isaac runtime structure gate.

This is not yet an ALOHA1-true physical simulation. It is a clean Isaac runtime scaffold that uses Trossen `stationary_ai` as the working structure standard while keeping ALOHA1 physical and electrical semantics blocked for verification.

## Generated Scaffold

- Directory: `local_eval_assets/aloha1_trossen_backed_scaffold_20260717/`
- USD: `local_eval_assets/aloha1_trossen_backed_scaffold_20260717/aloha1_trossen_backed_scaffold.usda`
- Adapter contract: `local_eval_assets/aloha1_trossen_backed_scaffold_20260717/trossen_backed_aloha1_scaffold_contract.json`

The scaffold references:

```text
external/trossen_ai_isaac/assets/robots/stationary_ai/stationary_ai.usd
```

## Validation Evidence

- Runtime report: `reports/aloha1_isaac_adaptation/phase3_scaffold_runtime_inspection_20260717/phase2_runtime_inspection.md`
- Runtime JSON: `reports/aloha1_isaac_adaptation/phase3_scaffold_runtime_inspection_20260717/phase2_runtime_inspection.json`
- Bounded log artifact: `.codex/artifacts/20260717-231626_phase3-scaffold-runtime-inspection`

The runtime check did not touch the real robot and did not save an Isaac stage.

## Scaffold Runtime Counts

For `local_eval_assets/aloha1_trossen_backed_scaffold_20260717/aloha1_trossen_backed_scaffold.usda`:

```text
joint_count = 32
mesh_count = 60
collider_count = 32
camera_count = 4
material_count = 63
```

Runtime articulation:

```text
/World/trossen_stationary_ai/Aloha1TrossenBackedScaffold/root_joint
```

Runtime DOF count:

```text
num_dof = 16
```

Runtime DOF order:

```text
follower_left_joint_0
follower_right_joint_0
follower_left_joint_1
follower_right_joint_1
follower_left_joint_2
follower_right_joint_2
follower_left_joint_3
follower_right_joint_3
follower_left_joint_4
follower_right_joint_4
follower_left_joint_5
follower_right_joint_5
follower_left_left_carriage_joint
follower_left_right_carriage_joint
follower_right_left_carriage_joint
follower_right_right_carriage_joint
```

## Unresolved Reference Check

The Phase 3 log contains 12 unresolved reference warnings, but all 12 come from the old generated ALOHA1 assets that the comparison script still inspects.

For the new scaffold:

```text
unresolved_scaffold = 0
```

This means the scaffold reference layer is structurally clean enough for the next gate.

## Gate Status

```text
scaffold_runtime_structure = PASS
current_generated_aloha1_training_asset = FAIL_NOT_SIM_READY
aloha1_adapter_semantics = BLOCKED_REQUIRES_REAL_DATA_VERIFICATION
controller_reuse = BLOCKED_UNTIL_ONE_JOINT_VALIDATION
gripper = BLOCKED_UNTIL_OPEN_CLOSE_CALIBRATION
camera = BLOCKED_UNTIL_EXTRINSIC_PROJECTION_TEST
contact_rl = BLOCKED_UNTIL_COLLIDER_AND_MATERIAL_REVIEW
```

## Next Step

The next safe step is one-joint validation design, not controller reuse.

For each ALOHA1 canonical arm joint, the system must establish:

```text
canonical field -> real ALOHA1 joint -> Trossen candidate DOF -> sign -> offset -> limit -> evidence source
```

If a field cannot be proven from local files, it must be verified from the real robot stack on `192.168.1.103` using read-only diagnostics. Do not guess physical or electrical truth.

