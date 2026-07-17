# Phase 4 Real ALOHA1 Joint Signal Probe - 2026-07-17

## Result

The real ALOHA1 ROS stack on `192.168.1.103` confirms the ALOHA1 joint names,
joint IDs, joint limits, operating modes, and gripper command mode used by the
user's robot project.

This moves part of the Trossen-backed ALOHA1 adapter contract from completely
unknown to partially verified.

It does **not** yet prove the full ALOHA1-to-Trossen Isaac control mapping.
The sign, zero offset, and real visual direction of each simulated Trossen DOF
remain blocked until one-joint validation.

## Safety Boundary

The probe used the user's project directory on `192.168.1.103`:

```text
/home/eii/openpi0.5-rtc-reward-learning
```

Before probing, the active robot-related containers were from compose project:

```text
openpi05-rlt
```

Following the user's instruction for uncertain physical/electrical facts, those
non-user-project containers were stopped and the user's minimal ROS read stack
was started:

```text
ros_master
redis
aloha_ros_nodes
rosbridge
```

The following were **not** started or called:

```text
rlt_warmup_runtime
openpi_server
runtime actor task commands
home / sleep commands
torque_enable
set_operating_modes
set_motor_registers
reboot_motors
```

The probe only read ROS topics and read-only Interbotix services.

## Repeatable Probe

Use:

```bash
scripts/probe_103_aloha_readonly_joint_facts.sh
```

If the 103 machine is running another compose project and the user explicitly
wants the user's ROS stack active, use:

```bash
scripts/probe_103_aloha_readonly_joint_facts.sh --switch-to-user-ros
```

The `--switch-to-user-ros` mode stops the known `openpi05-rlt` web/ROS
containers and starts only the user's minimal ROS read stack.

## Confirmed ROS Topics And Nodes

Confirmed joint state topics:

```text
/puppet_left/joint_states
/puppet_right/joint_states
/master_left/joint_states
/master_right/joint_states
```

Confirmed camera topics:

```text
/cam_high
/cam_low
/cam_left_wrist
/cam_right_wrist
```

Confirmed nodes include:

```text
/puppet_left/xs_sdk
/puppet_right/xs_sdk
/master_left/xs_sdk
/master_right/xs_sdk
/realsense_publisher
/rosapi
/rosbridge_websocket
```

## Confirmed Joint State Order

All four real robot arms publish this joint order:

```text
waist
shoulder
elbow
forearm_roll
wrist_angle
wrist_rotate
gripper
left_finger
right_finger
```

This confirms the ROS joint state order used by real ALOHA1.

For the 14D OpenPI/RLT adapter, the arm part maps naturally to:

```text
left_waist
left_shoulder
left_elbow
left_forearm_roll
left_wrist_angle
left_wrist_rotate
left_gripper
right_waist
right_shoulder
right_elbow
right_forearm_roll
right_wrist_angle
right_wrist_rotate
right_gripper
```

This confirms name order only. It does not yet confirm sign or zero offset
against Trossen Isaac DOFs.

## Confirmed Arm Robot Info

### Puppet Arms

For both `puppet_left` and `puppet_right`:

```text
mode = position
profile_type = velocity
joint_names = waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate
joint_ids = 1, 2, 4, 6, 7, 8
joint_lower_limits = -3.14158, -1.85005, -1.76278, -3.14158, -1.86750, -3.14158
joint_upper_limits =  3.14158,  1.25664,  1.60570,  3.14158,  2.23402,  3.14158
joint_velocity_limits = 3.14159 for all six arm joints
joint_sleep_positions = 0.0, -1.85, 1.55, 0.0, 0.8, 0.0
joint_state_indices = 0, 1, 2, 3, 4, 5
```

### Master Arms

For both `master_left` and `master_right`:

```text
mode = position
profile_type = velocity
joint_names = waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate
joint_ids = 1, 2, 4, 6, 7, 8
joint_lower_limits = -3.14158, -1.88496, -2.14675, -3.14158, -1.74533, -3.14158
joint_upper_limits =  3.14158,  1.98968,  1.60570,  3.14158,  2.14675,  3.14158
joint_velocity_limits = 3.14159 for all six arm joints
joint_sleep_positions = 0.0, -1.8, 1.55, 0.0, 0.8, 0.0
joint_state_indices = 0, 1, 2, 3, 4, 5
```

## Confirmed Gripper Robot Info

### Puppet Grippers

For both puppet grippers:

```text
mode = linear_position
profile_type = velocity
joint_names = left_finger
joint_ids = 9
joint_lower_limits = 0.021
joint_upper_limits = 0.057
joint_velocity_limits = 1.0
joint_sleep_positions = 0.02850
joint_state_indices = 7
```

### Master Grippers

For both master grippers:

```text
mode = position
profile_type = velocity
joint_names = left_finger
joint_ids = 9
joint_lower_limits = 0.015
joint_upper_limits = 0.037
joint_velocity_limits = 1.0
joint_sleep_positions = 0.01949
joint_state_indices = 7
```

Important consequence: puppet gripper command semantics are not the same as
master gripper command semantics. The Isaac adapter must not copy a single
generic gripper formula across both.

## Confirmed DYNAMIXEL Register Values

Read-only register checks confirmed:

```text
Operating_Mode = 3 for all arm joints and gripper joints
Profile_Velocity = 0
Profile_Acceleration = 0
```

Raw tick limits also matched the robot-info limits. Examples:

```text
puppet arm Min_Position_Limit = 0, 841, 898, 0, 830, 0
puppet arm Max_Position_Limit = 4095, 2867, 3094, 4095, 3504, 4095
master arm Min_Position_Limit = 0, 819, 648, 0, 910, 0
master arm Max_Position_Limit = 4095, 3345, 3094, 4095, 3447, 4095
```

These are useful as hardware evidence, but Isaac should use radian limits from
`get_robot_info` for arm DOFs and measured meters/opening semantics for gripper
DOFs.

## Adapter Fields That Can Now Be Confirmed

For the six arm joints on each side:

```text
real ALOHA1 joint name
real ROS joint order
real DYNAMIXEL ID
real arm lower limit in radians
real arm upper limit in radians
real arm velocity limit in radians per second
real sleep pose in radians
real operating mode from ROS robot_info
```

For the puppet gripper:

```text
real gripper command mode = linear_position
real gripper reported command lower/upper range
real gripper joint_state index
real gripper DYNAMIXEL ID
```

## Adapter Fields Still Blocked

The following must remain blocked:

```text
ALOHA1-to-Trossen sign
ALOHA1-to-Trossen zero offset
Trossen DOF command unit for the adapter layer
real visual direction for positive joint motion
controller replay correctness
gripper carriage mimic direction
gripper true physical opening in meters or millimeters
camera extrinsics and optical-frame convention
contact/collision material correctness
```

The reason is mathematical, not just practical: static robot-info gives limits
and joint identity, but it does not identify whether a positive real ALOHA1
joint increment produces the same physical transform direction as the candidate
Trossen DOF.

## Math Review Notes

Independent math review agrees with the conservative split:

```text
ROS joint order / IDs / ROS-side limits = confirmed
ROS-to-Trossen sign / offset / FK equivalence = blocked
```

The review also highlights a Trossen-specific ordering trap. Trossen runtime DOF
order is interleaved:

```text
L0, R0, L1, R1, L2, R2, L3, R3, L4, R4, L5, R5
```

It is not:

```text
L0, L1, L2, L3, L4, L5, R0, R1, R2, R3, R4, R5
```

Therefore any adapter that builds a full Trossen DOF vector must scatter from
the 14D ALOHA1 canonical order into the interleaved runtime order explicitly.
Passing a concatenated left-then-right arm vector directly into Trossen would be
a silent mathematical bug.

## One-Joint Validation Gate

For each canonical joint, define:

```text
q_real      = real ALOHA1 joint position from ROS
q_sim       = Trossen Isaac candidate DOF position
s           = candidate sign, either +1 or -1
b           = candidate offset
q_sim_pred  = s * q_real + b
```

A candidate mapping is acceptable only if:

```text
q_sim_pred is inside the Trossen DOF limit
```

and a small positive change has matching kinematic direction:

```text
delta_p_real(j) and delta_p_sim(j) point in the same expected direction
```

For a purely joint-space first gate, the minimum acceptance is:

```text
joint_identity = confirmed by name and side
joint_limit_overlap = nonempty
sign = confirmed by positive-motion direction test
offset = confirmed by at least two matched poses
```

The offset cannot be solved from one pose alone unless the zero convention is
already known. With two matched poses:

```text
q_sim_1 = s * q_real_1 + b
q_sim_2 = s * q_real_2 + b
```

Then:

```text
s = sign((q_sim_2 - q_sim_1) / (q_real_2 - q_real_1))
b = q_sim_1 - s * q_real_1
```

This is why the next validation needs either:

1. two or more trusted real ALOHA1 poses matched to Isaac candidate poses; or
2. a safe one-joint positive-direction test observed in real data; or
3. a trusted URDF/USD/MJCF conversion chain with explicit axis agreement.

## Updated Gate Status

```text
real_joint_order = PASS
real_joint_ids = PASS
real_joint_limits = PASS
real_sleep_positions = PASS
real_gripper_mode = PASS
aloha1_to_trossen_sign = BLOCKED
aloha1_to_trossen_offset = BLOCKED
one_joint_validation = NEXT
controller_reuse = BLOCKED
gripper_physical_opening = BLOCKED
camera_extrinsic_projection = BLOCKED
contact_rl = BLOCKED
```

## Next Implementation Step

Build a one-joint validation artifact that starts from the Trossen-backed
scaffold and checks the adapter mapping row-by-row.

The first version should not command the real robot. It should:

1. read real ROS joint facts from this probe output or rerun the read-only probe;
2. load the scaffold in Isaac headless;
3. set candidate simulated DOFs to the real current pose using an explicit
   provisional sign and offset;
4. assert limits and read back DOF state;
5. mark sign and offset as provisional, not confirmed;
6. produce a report listing exactly which rows still require real positive
   direction evidence.
