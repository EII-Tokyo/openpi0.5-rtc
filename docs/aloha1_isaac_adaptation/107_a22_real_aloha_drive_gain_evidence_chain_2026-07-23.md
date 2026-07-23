# A22 Real ALOHA Drive-Gain Evidence Chain - 2026-07-23

## Purpose

This document is the required evidence entry point for A22 and for any later
ALOHA1 Isaac drive-gain change. It prevents three different parameter domains
from being conflated:

1. real DYNAMIXEL controller registers;
2. Gazebo or Isaac simulator controller gains;
3. authored or runtime PhysX articulation-drive properties.

The central correction is:

> A19 `stiffness=0` describes the current USD drive state. It is not evidence
> that the real ALOHA1 has no position feedback or no tuned servo controller.

## Evidence Classification

### Confirmed On The User's Real ALOHA1

Phase 4 ran a read-only probe against the real ALOHA1 ROS stack on
`192.168.1.103`. It confirmed:

- puppet arm mode `position`;
- profile type `velocity`;
- arm joint IDs, names, limits, and velocity limits;
- `Operating_Mode=3`;
- `Profile_Velocity=0`;
- `Profile_Acceleration=0`.

Source:

- `docs/aloha1_isaac_adaptation/09_phase4_real_aloha1_joint_signal_probe_2026-07-17.md`
- `scripts/probe_103_aloha_readonly_joint_facts.sh`

The script's register loop queried exactly:

```text
Operating_Mode
Min_Position_Limit
Max_Position_Limit
Profile_Velocity
Profile_Acceleration
```

It did not query:

```text
Position_P_Gain
Position_I_Gain
Position_D_Gain
Velocity_P_Gain
Velocity_I_Gain
PWM_Limit
Current_Limit
```

Therefore Phase 4 is genuine real-hardware evidence, but it is not a complete
real-hardware drive-gain snapshot.

### Confirmed In The User's 103 Project Code

Read-only inspection on 2026-07-23 verified the remote project boundary and
revision:

```text
project: /home/eii/openpi0.5-rtc-reward-learning
revision: ea818494bf9ee7756c955864ba3b0d62be6ce649
```

`examples/aloha_real/robot_utils.py` defines:

```text
standard arm gains:
  Position_P_Gain = 800
  Position_I_Gain = 0

low arm gains:
  Position_P_Gain = 100
  Position_I_Gain = 0
```

The code does not write `Position_D_Gain`, velocity gains, arm current limits,
or arm PWM limits. Repository-wide call-site inspection found definitions but
no current call sites for `set_standard_pid_gains` or `set_low_pid_gains`.
These values are therefore source-code intent, not proof of current motor
register contents.

The current code also writes configurable puppet-gripper `Current_Limit`
values during gripper setup. That is a gripper safety/actuation setting and
must not be generalized to the six arm joints.

### Confirmed In Tony Zhao's Public ALOHA Source

Tony Zhao's official ALOHA repository, commit
`06369f03cd8e0a47e16d3a90167853fd33af7557`, contains the same group-wide
helpers:

- standard `Position_P_Gain=800`, `Position_I_Gain=0`;
- low `Position_P_Gain=100`, `Position_I_Gain=0`;
- no explicit Position D write.

Source:

- https://github.com/tonyzhaozh/aloha/blob/06369f03cd8e0a47e16d3a90167853fd33af7557/aloha_scripts/robot_utils.py#L173-L179

The same source configures puppet arms for position mode with velocity-profile
semantics and zero profile velocity/acceleration:

- https://github.com/tonyzhaozh/aloha/blob/06369f03cd8e0a47e16d3a90167853fd33af7557/config/puppet_modes_left.yaml

The helpers have no call site in that public commit. They establish the
author's standard/low gain convention, not a measured register snapshot of the
user's robot.

### Confirmed In ROBOTIS/Trossen/Interbotix Sources

ROBOTIS publishes these defaults for both XM540-W270 and XM430-W350:

```text
Position P/I/D = 800 / 0 / 0
Velocity P/I = 100 / 1920
Profile Velocity/Acceleration = 0 / 0
```

Sources:

- https://emanual.robotis.com/docs/en/dxl/x/xm540-w270/#control-table-of-ram-area
- https://emanual.robotis.com/docs/en/dxl/x/xm430-w350/#control-table-of-ram-area

ROBOTIS also documents the internal position-gain scaling:

```text
internal P = table P / 128
internal I = table I / 65536
internal D = table D / 16
```

Source:

- https://emanual.robotis.com/docs/en/dxl/x/xm540-w270/#position-pid-gain80-82-84-feedforward-1st2nd-gains88-90

The official Interbotix `aloha_vx300s.yaml` does not provide a per-joint
Position/Velocity PID table. It configures motor IDs, direction/drive mode,
velocity and position limits, shadow motors, and an ALOHA gripper current
limit:

- https://github.com/Interbotix/interbotix_ros_manipulators/blob/0bb2b0e6d0e619bff02cf74dbd5af5681dcf80c9/interbotix_ros_xsarms/interbotix_xsarm_control/config/aloha_vx300s.yaml

Trossen's ViperX-300s specification identifies the motor family:

- waist through wrist angle primarily use XM540-W270;
- wrist rotate and gripper use XM430-W350.

Source:

- https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/vx300s.html

### Confirmed Simulator References

Interbotix publishes a joint-specific Gazebo effort-controller reference:

| Joint | P | I | D |
| --- | ---: | ---: | ---: |
| waist | 100 | 5 | 1 |
| shoulder | 500 | 10 | 1 |
| elbow | 300 | 10 | 1 |
| forearm_roll | 100 | 3 | 0.1 |
| wrist_angle | 100 | 3 | 0.1 |
| wrist_rotate | 100 | 1 | 0.1 |
| gripper | 100 | 0 | 1 |

Source:

- https://github.com/Interbotix/interbotix_ros_manipulators/blob/0bb2b0e6d0e619bff02cf74dbd5af5681dcf80c9/interbotix_ros_xsarms/interbotix_xsarm_gazebo/config/trajectory_controllers/vx300s_trajectory_controllers.yaml

This is useful evidence about relative joint-control strength. It is not a
DYNAMIXEL register snapshot and is not numerically interchangeable with a
PhysX drive.

The local Trossen Stationary AI USD has nonzero, joint-specific stiffness and
damping. It is a known-working Isaac organization and physics reference, but
its robot, inertias, force limits, units, and joint structure differ from the
user's ALOHA1. Do not copy its values directly.

### Confirmed Same-Lineage Isaac ALOHA Result

Phase 97 is the strongest direct Isaac gain prior currently available. It ran
the Menagerie-derived `/scene` ALOHA asset family in PhysX `drive_target` mode
at the original 50 Hz target cadence and passed the recorded tracking and
contact-candidate gates with:

```text
arm kp = 1600
arm kd = 100
finger kp = 200
finger kd = 50
```

Evidence:

- `docs/aloha1_isaac_adaptation/83_phase97_drive_target_controller_gain_reference_2026-07-18.md`
- `reports/aloha1_isaac_adaptation/phase97_scene_proxy_hdf5_replay_drive_target_arm1600_kd100_finger200_native_workcell_20260718/gripper_passive_contact_metrics.json`
- `.codex/artifacts/20260718-234743_aloha-phase97-native-workcell-drive-target-arm1600-kd100-finger200`

Phase 97 is not proof that the same numbers are automatically stable on A19's
new single-root articulation. It is nevertheless a much closer starting prior
than directly copying Stationary AI because the geometry and physical source
lineage are shared.

## Why The Number Domains Must Not Be Copied

The DYNAMIXEL position register is part of a discrete nested servo controller.
Its behavior also depends on internal velocity PI, PWM/current limits, motor
constants, gear reduction, shadow motors, friction, and structural compliance.

PhysX articulation drives apply an implicit joint-space PD model. For a
rotational force drive:

- stiffness is expressed as joint-end torque per angular error;
- damping is expressed as joint-end torque per angular velocity error;
- max force limits the drive output.

Official PhysX source:

- https://nvidia-omniverse.github.io/PhysX/physx/5.3.0/docs/Articulations.html#articulation-joint-drives

Consequently:

```text
DYNAMIXEL Position_P_Gain = 800
```

does not imply:

```text
PhysX stiffness = 800
```

The hardware values provide a real-controller truth and qualitative stiffness
anchor. Phase 97 provides the direct Isaac numerical prior. A22 must determine
whether that prior remains stable on the new A19 articulation through bounded
runtime-only micro-motion.

## Historical Search Audit

On 2026-07-23, the following archives were searched for complete hardware
register names and readback commands:

- every `*phase*.md` under `docs/aloha1_isaac_adaptation/`;
- text/JSON/YAML reports under `reports/aloha1_isaac_adaptation/`;
- existing text artifacts under `.codex/artifacts/`;
- 2026 Codex session JSONL files for exact `get_motor_registers` plus gain,
  current, or PWM register names.

Results:

- Phase 4 is the only Phase document that records a real 103 DYNAMIXEL
  register probe;
- its saved script and report do not query the complete gain/current/PWM set;
- no saved Phase report or artifact contains returned values for the missing
  complete register set;
- session-history matches contain source-code definitions and gripper-current
  code, not a complete arm register readback result.

The user remembers that a more complete read may have been run. That memory is
recorded as an unresolved historical claim, not rejected. Missing session
records, earlier untracked terminal output, or incomplete persistence may
explain why the output is absent. Absence from the current archives is not
proof that the read never occurred.

## Required Future Read-Only Register Snapshot

If the user explicitly authorizes a new real-hardware read-only probe, capture
all six puppet-arm motors on both sides, per motor ID, for:

```text
Model_Number
Firmware_Version
Operating_Mode
Drive_Mode
Position_P_Gain
Position_I_Gain
Position_D_Gain
Velocity_P_Gain
Velocity_I_Gain
Feedforward_1st_Gain
Feedforward_2nd_Gain
Profile_Velocity
Profile_Acceleration
Velocity_Limit
PWM_Limit
Current_Limit
```

Also record which values are EEPROM versus RAM, which ROS/container revision
performed the read, and whether any startup code may have rewritten RAM
registers. The probe must not write registers, enable torque, reboot motors,
or command motion. Starting an SDK/container that can initialize hardware
still requires explicit user authorization and the real-hardware safety rules.

## A22 Decision Rule

A22 should use this priority:

1. Phase 97 same-lineage Isaac gains as the first numerical candidate;
2. real ALOHA and ROBOTIS evidence as the controller-intent anchor;
3. Interbotix Gazebo gains as relative per-joint-strength evidence;
4. Stationary AI as a cross-robot physical sanity reference.

A22 remains runtime-only, gravity-off, collision-off, single-joint,
small-delta, separately batched left/right micro-motion. It must measure
direction, non-target motion, peak excursion, velocity decay, settling, and
baseline restoration before any gain is authored into USD or any gravity,
collision, contact, or replay gate is attempted.
