# Phase97 Drive-Target Controller Gain Reference

## Result

Phase97 is the current first full-gate PASS for native `/scene` ALOHA1 HDF5 replay in `drive_target` mode.

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
| max object displacement | `0.11387294474651785` |

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
| 97 | arm `kp=1600`, arm `kd=100`, finger `kp=200`, finger `kd=50` | PASS | lowering arm damping resolved lag without causing non-target contact |

## Current Interpretation

The root cause was not FK mapping or bottle contact geometry. The remaining failure was articulation-drive tuning:

- the imported `/scene` arm DOF names are prefixed, so the original arm-gain helper did not tune them;
- after fixing prefixed DOF matching, arm tracking improved;
- finger DOFs needed a separate gain override;
- too much arm stiffness passed tracking but introduced bad object-to-gripper-base contact;
- reducing arm damping from `200` to `100` gave tracking margin without corrupting contact.

## Next Gate

Use Phase97 as the current drive-target reference before trying full bottle manipulation or policy replay. Do not advance to a harder grasp/replay task unless all Phase97 gates remain passing after any stage or controller change.
