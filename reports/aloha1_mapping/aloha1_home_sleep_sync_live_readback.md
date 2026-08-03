# ALOHA1 follower_left live read-only runtime

- Overall: `PASS_READ_ONLY_RUNTIME_MOTION_NOT_RUN`
- Real motion: `NOT_RUN_AUTHORIZATION_REQUIRED`
- Real/digital correspondence: `NOT_RUN_REAL_MOTION_EVIDENCE_MISSING`
- Workspace motion gate: `FAIL_CLUTTERED_TABLE`
- Stop/hold gate: `NOT_VERIFIED`

## Verified runtime evidence

- Driver running at final readback: `True`.
- Arm/gripper modes: `position` / `linear_position`.
- ROS status: `PASS_PUPPET_LEFT_READ_ONLY_RUNTIME`.
- Explicit arm order: `waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate`.
- Command messages published by diagnostics: `0`.
- Joint-state samples: `20`; max position span `0.001533985 rad`; maximum reported velocity `0.000000000 rad/s`.
- cam_high pre/post frames: `600` / `600`; hardware resets: `0`.

## Safety boundary

This phase enabled the existing follower-left arm and gripper torque through the isolated driver configuration, but did not construct a robot command publisher and did not send Home, Sleep, or any other motion command. Torque enable is inferred from the deployed mode configuration and driver startup log; it is not a direct register readback.

The cam_high images show a cluttered tabletop. They are auxiliary workspace-safety evidence and do not prove signal correspondence. Real motion remains blocked pending a cleared workspace, an operator-tested stop/hold path, and fresh explicit authorization.

## Remaining gates

- `operator_clear_table_and_confirm_workspace`
- `operator_tested_stop_hold_path`
- `explicit_real_motion_authorization`
- `home_sleep_three_cycle_real_execution`
- `real_digital_signal_comparison`

## Evidence files

- `camera_pre_image`: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260803-aloha1-synchronized-real-sim/live_readback/cam_high_pre_driver.jpg` (SHA-256 `c4bfb20024d756d7f21cffe5443a5fc668547915f2a9899ba28df02b01947171`)
- `camera_post_image`: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260803-aloha1-synchronized-real-sim/live_readback/cam_high_post_driver.jpg` (SHA-256 `6d93f84e81c2263c3ccbbbdaba988df19083036c837b54a4ecf8d7e08f371386`)
- `container_state`: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260803-aloha1-synchronized-real-sim/live_readback/container_state.jsonl` (SHA-256 `c6b930c1ad6216bff0fa1e57fc26beb0d6016757dc5cf8e04d431a6ae6a94176`)
- `ros_report`: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260803-aloha1-synchronized-real-sim/live_readback/aloha1_home_sleep_live_ros_readback.json` (SHA-256 `e1ba33c0dbccd089555022fb1ff21e50a9fef9cd378b22404341a1e2bade0cf4`)
- `camera_pre_report`: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260803-aloha1-synchronized-real-sim/live_readback/cam_high_probe_pre_driver.json` (SHA-256 `8b5ee6171ca775107bd88d28db75ae78ffca47ec5fb67f8e6c8a102b06622dc4`)
- `camera_post_report`: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260803-aloha1-synchronized-real-sim/live_readback/cam_high_probe_post_driver.json` (SHA-256 `a537215530e76abadd534548213d7c189f9f73f5fec907db642ebfc9c7f37483`)
- `driver_log`: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260803-aloha1-synchronized-real-sim/live_readback/puppet_left_driver.log` (SHA-256 `b0c2bc52624646d8642f998df4e55d0d3a57f6faf059c0df760779a8448693ac`)
- `joint_state_csv`: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260803-aloha1-synchronized-real-sim/live_readback/puppet_left_joint_states_20.csv` (SHA-256 `148a5c9d0faec7e75ad1ec387fb91a9e479fcad81bfd073d28441a6a3ffe889c`)
