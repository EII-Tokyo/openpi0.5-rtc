# ALOHA1 synchronized Home/Sleep offline gate

Status: **READY_FOR_SUPERVISED_REAL_EXECUTION**

This status means the offline protocol and Isaac worker are ready for a
separately authorized, supervised real-hardware run. It does **not** mean that
the real robot was accessed or that real/digital correspondence already passed.

## Offline gates

- `fake_coordinator`: PASS
- `isaac_process_statuses`: PASS
- `isaac_deterministic_signature`: PASS
- `ros1_official_source_audit`: PASS
- `prohibited_side_effects_absent`: PASS

## Isaac evidence

- Runtime: Isaac Sim `5.1.0.0`, Kit
  `107.3.3`, PhysX `107.3.26`
- Fresh processes: `3`
- Identical signature: `d93ae226dcb2a11a728f4abda1dc821867d1eae0893c3cc01fdb4e8113696562`
- Paced start skew: `35 ns`
- Paced maximum lateness: `11435 ns`
- Burst catch-up: `false`

## Remaining live gates

- `real_access_authorized`
- `read_only_103_preflight_pass`
- `deployed_joint_order_verified`
- `deployed_position_mode_verified`
- `stop_path_verified`
- `cam_high_stream_verified`
- `operator_workspace_clear`
- `real_motion_authorized`

Real execution remains **NOT_RUN_AUTHORIZATION_REQUIRED**. No ROS publisher,
network connection, motor command, or torque change was made by this gate.
Task 8 remains `COMPLETE_WITH_NO_PROMOTION`.
