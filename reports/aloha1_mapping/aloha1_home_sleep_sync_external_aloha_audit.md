# ALOHA1 external ROS1 source audit

Status: **READY_FOR_MINIMAL_START_AUTHORIZATION**

This audit was read-only. It started no ROS node or robot driver, constructed
no publisher, sent no command, and changed no torque state.

## Frozen source

- Root: `/home/eii/openpi0.5-rtc/third_party/aloha`
- Git top-level: `/home/eii/openpi0.5-rtc`
- Origin: `https://github.com/EII-Tokyo/openpi0.5-rtc.git`
- Branch/commit: `codex/minimal-aloha-real` / `f2e6a34c0433285f31f4cc575650cc3f978ac874`
- Dirty entries: `11` (preserved)
- License: `MIT`

- `/home/eii/openpi0.5-rtc/third_party/aloha/launch/ros_nodes.launch` — `3307a47cdf66bc5c9e5e6362b12182250f18736b9fd0d7d0ca42c5982684c90e`
- `/home/eii/openpi0.5-rtc/third_party/aloha/launch/4arms_teleop.launch` — `d8543ec40081a807bfa43c7b149c7945f933856f180f277fb937cffc2cdaa632`
- `/home/eii/openpi0.5-rtc/third_party/aloha/config/master_modes_left.yaml` — `dd755f147fb707ea84b9533e7bbba4bef5348a13a7a807c97f9bf01cda9605e9`
- `/home/eii/openpi0.5-rtc/third_party/aloha/config/master_modes_right.yaml` — `ec95fa410c6a8a2a9427f8566365108550d57c5fc233a6c28a5c8717959cf8bc`
- `/home/eii/openpi0.5-rtc/third_party/aloha/config/puppet_modes_left.yaml` — `862940f33e2459255873d5839a401a5f2d0c0576aa9439b6b58d57118a182cb6`
- `/home/eii/openpi0.5-rtc/third_party/aloha/config/puppet_modes_right.yaml` — `a744f6a24b9565027c39770d63d1e1381b223db920815ce55c060d28af30503f`
- `/home/eii/openpi0.5-rtc/third_party/aloha/aloha_scripts/realsense_publisher.py` — `a80d80cb2c8d85ec5f89487c830b5776ec9001655682bfb812aaf44b90bb9c26`
- `/home/eii/openpi0.5-rtc/third_party/aloha/aloha_scripts/robot_utils.py` — `60150e66ca6caf44fe5846f4a67dfae08f2faabc40337f269fb7a9ad2e547733`
- `/home/eii/openpi0.5-rtc/third_party/aloha/aloha_scripts/sleep.py` — `c3901b50f2a0c35f7a71fc04299ae9370147c47b750e87b30cb2ea3543cbfa6e`
- `/home/eii/openpi0.5-rtc/third_party/aloha/msg/RGBGrayscaleImage.msg` — `d7f80679cbe72f43c639191caa5a23be8c587ffa18a4e9b558af87868af9fc73`
- `/home/eii/openpi0.5-rtc/third_party/aloha/LICENSE` — `4666c312da313e6c46929f6695d06cf98a2e7359b9c7dcbb0ea232d01b32cd42`

## Existing deployment boundary

Existing `ros_nodes.launch`: **REJECTED_FOR_LEFT_ONLY_SUPERVISED_REPLAY**.
It includes `4` robot drivers:
`master_left, master_right, puppet_left, puppet_right`. The left follower arm
and gripper mode configuration both have torque enabled. The bundled camera
publisher requires four camera serials and calls `hardware_reset()`. The
bundled `sleep.py` constructs and commands both puppet arms. None of these
entry points is accepted for the left-only supervised replay.

## Isolated launch candidate

Candidate static status: **PASS_STATIC_LEFT_ONLY_SCOPE**. It includes only
`puppet_left` / `vx300s`, uses the deployed left mode configuration, keeps
`load_configs=false`, and contains no camera node. This is an inert source
file, not runtime evidence. Starting it would touch real hardware and remains
**NOT_RUN_AUTHORIZATION_REQUIRED**.

## Remaining gates

- `explicit_authorization_to_start_puppet_left_driver`
- `runtime_joint_order_and_position_mode_readback`
- `operator_tested_stop_hold_path`
- `cam_high_single_camera_runtime_path`
- `operator_workspace_clear`
- `explicit_real_motion_authorization`

Real execution remains **NOT_RUN_AUTHORIZATION_REQUIRED**.
