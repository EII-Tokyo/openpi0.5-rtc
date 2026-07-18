# Phase97 Drive-Target Controller Gain Reference

## Result

Phase97 is the current controller/reference PASS for native `/scene` ALOHA1 HDF5 replay in `drive_target` mode under the current validator gates.

This is an important milestone, but its scope is deliberately narrow. It proves that Isaac articulation drives can follow the selected HDF5 target sequence with bounded tracking error while preserving the current bilateral contact candidate and strict object non-target contact gate. It does not prove full bottle grasp success, calibrated table/base geometry, realistic bottle friction, or complete task-level bottle manipulation.

Report:

`reports/aloha1_isaac_adaptation/phase97_scene_proxy_hdf5_replay_drive_target_arm1600_kd100_finger200_native_workcell_20260718/gripper_passive_contact_metrics.json`

Artifact:

`.codex/artifacts/20260718-234743_aloha-phase97-native-workcell-drive-target-arm1600-kd100-finger200`

## Command Parameters That Matter

- Mapping: `configs/aloha/trossen_scene_base_link_aloha1_left_controller_mapping.yaml`
- Stage: `local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose/aloha2_menagerie_scene_deep_black_real_start_pose_proxy_runtime.usda`
- Replay mode: `left_arm_and_gripper`
- Actuation mode: `drive_target`
- Target hold steps: `1`
- Arm gains: `--arm-kp 1600 --arm-kd 100`
- Finger gains: `--finger-kp 200 --finger-kd 50`
- Strict non-target gate:
  - `--fail-on-non-target-object-contact`
  - `--allowed-non-target-object-contact-category workcell_or_environment`
- Contact phase semantics:
  - `--already-in-contact-setup`

## Gate Evidence

| Metric | Phase97 value |
| --- | --- |
| status | `PASS` |
| failure reasons | `[]` |
| target limit gate | `true` |
| controller tracking gate | `true` |
| max controlled error | `0.012857437133789062` |
| max error DOF | `left_shoulder` |
| contact trace status | `PASS_BILATERAL_CONTACT_CANDIDATE` |
| non-target contact gate | `PASS_NON_TARGET_CONTACTS_ALLOWED` |
| active target contact gate | `SKIPPED_ALREADY_IN_CONTACT_SETUP` |
| max object displacement | `0.11387294474651785` |

## Scope Boundaries

Phase97 should not be used as a broad "ALOHA1 grasp solved" claim.

- Target contact starts in the settle window. This means Phase97 validates an already-contacting or contact-candidate replay setup, not active approach-and-grasp from free space.
- The support-plane calibration gate is not active in this run. `support_plane_mode` is `none`, so the pass does not validate the real table frame or real workcell calibration.
- The strict non-target gate checks object contacts by category. It does not yet evaluate contact force, penetration depth, friction realism, or gripper mesh fidelity.
- PhysX still emits mesh-collision fallback warnings in the artifact logs. The pass remains valid for this validator, but collision-shape fidelity is still a residual risk.
- `wrong_contact_pairs` can contain non-object workcell or proxy contacts. Those are not the same as object-to-wrist, object-to-gripper-base, or object-to-opposite-arm failures.

## Why This Matters

Earlier state-teleport runs proved geometry and mapping could be made consistent, but they did not prove the Isaac articulation drives could actually follow the HDF5 trajectory. Phase97 uses `drive_target`, so the controller must follow the same target sequence through PhysX.

The pass did not come from slowing the replay down. `hdf5_replay_target_hold_steps` remained `1`, so each recorded 50 Hz target is still applied for one physics step.

## Failed Alternatives

| Phase | Main setting | Result | Lesson |
| --- | --- | --- | --- |
| 92 | arm `kp=1600`, arm `kd=200`, no finger override | tracking failed at `left_left_finger`, max error `0.027605427145957942` | finger DOFs need separate runtime gain control |
| 93 | arm `kp=1600`, arm `kd=200`, finger `kp=200`, finger `kd=50` | contact passed, tracking failed at `left_shoulder`, max error `0.024291887879371643` | finger gain worked; arm damping was still too high |
| 94 | arm `kp=2400`, arm `kd=200`, finger `kp=200`, finger `kd=50` | tracking passed, strict contact failed | higher stiffness can force the object into same-side gripper-base contact |
| 95 | arm `kp=2000`, arm `kd=200`, finger `kp=200`, finger `kd=50` | tracking and strict contact both failed | still too stiff for contact while not enough tracking margin |
| 96 | arm `kp=1800`, arm `kd=200`, finger `kp=200`, finger `kd=50` | tracking and strict contact both failed | the contact failure starts below `kp=2000` |
| 97 | arm `kp=1600`, arm `kd=100`, finger `kp=200`, finger `kd=50` | PASS | `kp=1600` kept the contact setup stable, and lower arm damping gave enough tracking margin |

## Current Interpretation

The root cause was not FK mapping or bottle contact geometry. The remaining failure was articulation-drive tuning:

- the imported `/scene` arm DOF names are prefixed, so the original arm-gain helper did not tune them;
- after fixing prefixed DOF matching, arm tracking improved;
- finger DOFs needed a separate gain override;
- too much arm stiffness passed tracking but introduced bad object-to-gripper-base contact;
- at `kp=1600`, reducing arm damping from `200` to `100` gave tracking margin without corrupting contact.

## Next Gate

Use Phase97 as the current drive-target controller reference before trying full bottle manipulation or policy replay. Do not advance to a harder grasp/replay task unless all Phase97 gates remain passing after any stage or controller change.

The next milestone must separate contact-candidate replay from real grasp success. If a future report claims active grasp, it should require target contact to first appear during the close or grasp phase, or it should explicitly mark the setup as `already_in_contact_setup=true`.
