# Phase 73 Robot-State Source Guard

## Question

The ALOHA1 expert review highlighted a recurring trap:

```text
robot state signal != workcell geometry calibration
```

Real robot recordings can prove joint positions, commanded actions, gripper opening, or motor state. They cannot by themselves prove:

```text
T_world_table
T_table_left_base
T_table_right_base
```

Those transforms must come from physical measurement or a trusted explicit geometry source.

## Rejected Sources

The table-to-base worksheet and readiness gate now reject these `measurement.source` values for table/base calibration:

```text
hdf5_qpos
joint_states
dynamixel_registers
ros_static_transform_default
```

Why:

- `hdf5_qpos` records robot joint state, not table-to-base geometry.
- `joint_states` records ROS robot joint state, not table-to-base geometry.
- `dynamixel_registers` records actuator state/configuration, not table-to-base geometry.
- `ros_static_transform_default` is a default/static transform and is not measured workcell calibration.

These signals can still be used for other checks, such as replay tracking, joint-order validation, gripper state validation, or controller debugging. They cannot be promoted into final calibrated table/base geometry.

## Implementation

Shared source validation lives in:

```text
aloha_isaac_replay/calibration/table_measurement_guidance.py
```

The guard is consumed by:

```text
aloha_isaac_replay/scripts/create_table_to_base_calibration_from_worksheet.py
aloha_isaac_replay/scripts/summarize_table_calibration_readiness.py
```

This means both paths now agree:

1. Building a calibration file from a worksheet rejects forbidden robot-state sources.
2. Readiness reporting also marks the worksheet blocked before generation.

## Validation

Validated locally:

```text
.venv/bin/ruff format aloha_isaac_replay/calibration/table_measurement_guidance.py aloha_isaac_replay/scripts/create_table_to_base_calibration_from_worksheet.py aloha_isaac_replay/scripts/summarize_table_calibration_readiness.py aloha_isaac_replay/tests/test_table_to_base_measurement_worksheet.py aloha_isaac_replay/tests/test_table_calibration_readiness.py
.venv/bin/ruff check aloha_isaac_replay/calibration/table_measurement_guidance.py aloha_isaac_replay/scripts/create_table_to_base_calibration_from_worksheet.py aloha_isaac_replay/scripts/summarize_table_calibration_readiness.py aloha_isaac_replay/tests/test_table_to_base_measurement_worksheet.py aloha_isaac_replay/tests/test_table_calibration_readiness.py
.venv/bin/python -m pytest -q aloha_isaac_replay/tests/test_table_to_base_measurement_worksheet.py aloha_isaac_replay/tests/test_table_calibration_readiness.py aloha_isaac_replay/tests/test_calibrated_table_base_overlay.py aloha_isaac_replay/tests/test_create_table_to_base_calibration.py aloha_isaac_replay/tests/test_table_frame_candidate_audit.py
git diff --check
```

Result:

```text
All checks passed
20 passed
```

No real robot, `192.168.1.103`, or Isaac runtime action was used.

## Interpretation

This closes a false-positive path where a robot-state source could accidentally make the pipeline look calibrated.

It does not solve the remaining measurement blocker. Final validated replay/contact work still requires table/base transforms from physical measurement or trusted explicit geometry.
