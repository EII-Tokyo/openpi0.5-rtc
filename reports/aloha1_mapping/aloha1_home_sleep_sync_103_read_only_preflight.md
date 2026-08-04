# ALOHA1 103 read-only preflight

Status: **PARTIAL_RUNTIME_READBACK_REQUIRED**

The inspection was restricted to
`/home/eii/openpi0.5-rtc-reward-learning`. It created no ROS publisher, sent
no command and changed no torque state.

## Static checks

- `project_root`: PASS
- `compose_hash`: PASS
- `robot_utils_hash`: PASS
- `constants_hash`: PASS
- `joint_order_declared`: PASS
- `joint_state_topic_declared`: PASS
- `joint_command_topic_declared`: PASS
- `cam_high_declared`: PASS
- `launch_command_declared`: PASS
- `external_mount_boundary_recorded`: PASS

Remote HEAD is `ea818494bf9ee7756c955864ba3b0d62be6ce649` on
`paper_actor_sample` with
`45` dirty/untracked entries. They
must be preserved. The robot stack and ROS master are stopped, so static
source declarations are not treated as runtime readback.

The compose file declares an external ALOHA mount at
`/home/eii/openpi0.5-rtc/third_party/aloha`. It was not inspected because it lies
outside the approved remote project boundary.

## Remaining gates

- `authorization_to_start_robot_driver`
- `deployed_runtime_joint_order`
- `deployed_runtime_position_mode`
- `cam_high_runtime_message`
- `operator_tested_stop_hold_path`
- `operator_workspace_clear`
- `real_motion_authorized`

The next operation would start the real robot driver. It requires explicit
user authorization before execution.
