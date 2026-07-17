# Phase 5 One-Joint Static Validation - 2026-07-17

## Result

The Trossen-backed ALOHA1 scaffold passes the first offline Isaac static
validation gate.

The most important finding is negative but useful:

```text
identity mapping q_isaac = q_ros fails the runtime limit check
```

This is expected for a real adapter. It proves the system must include an
explicit ALOHA1-to-Trossen sign/offset layer before controller replay or RL can
be trusted.

## Evidence

- Script: `aloha_isaac_replay/scripts/validate_trossen_backed_aloha1_one_joint_mapping.py`
- Report: `reports/aloha1_isaac_adaptation/phase5_one_joint_static_validation_20260717/one_joint_static_validation.md`
- JSON: `reports/aloha1_isaac_adaptation/phase5_one_joint_static_validation_20260717/one_joint_static_validation.json`
- Bounded log artifact: `.codex/artifacts/20260717-234010_phase5-one-joint-static-validation-renamed-gates`

The validation was offline/headless Isaac only:

```text
real_robot_touched = false
stage_saved = false
runtime/actor not started
```

## Gates

```text
isaac_runtime_started = PASS
real_robot_touched = PASS_FALSE
stage_saved = PASS_FALSE
dof_order_interleaved_confirmed = PASS
all_arm_candidate_dofs_present = PASS
identity_mapping_limit_check = FAIL_IDENTITY_MAPPING_LIMIT_CHECK
scatter_set_readback = PASS_SCATTER_SET_READBACK_ONLY
sign = BLOCKED_REQUIRES_POSITIVE_DIRECTION_EVIDENCE
offset = BLOCKED_REQUIRES_MATCHED_REFERENCE_POSES
fk = BLOCKED_REQUIRES_TRUSTED_FK_OR_REFERENCE_POSES
gripper_mapping = BLOCKED_REQUIRES_CARRIAGE_AND_PHYSICAL_OPENING_VALIDATION
```

## Confirmed Trossen Runtime DOF Order

Trossen runtime DOF order is interleaved:

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

Therefore any adapter must scatter the ALOHA1 canonical 14D order into this
runtime order. It must not pass a left-then-right arm vector directly to Trossen.

## Why Direct Identity Mapping Fails

The real ALOHA1 puppet shoulder values are around:

```text
left_shoulder sample ~= -1.848 rad
right_shoulder sample ~= -1.858 rad
```

But the Trossen runtime shoulder candidate limits are:

```text
follower_left_joint_1  in [0, pi]
follower_right_joint_1 in [0, pi]
```

So a direct identity mapping would put the shoulder outside the Trossen runtime
limit:

```text
q_isaac = q_ros
```

This fails for shoulder immediately. It also fails for the current wrist-angle
samples, which are outside the Trossen wrist-angle candidate limit.

The correct adapter needs an explicit mapping:

```text
q_isaac = s * q_ros + b
```

where:

```text
s = sign, either +1 or -1
b = offset
```

Neither `s` nor `b` is confirmed yet.

## What The Readback PASS Means

The script only set provisional identity values that were already inside the
Trossen runtime limits. This tests scatter indices and Isaac readback behavior,
not physical correctness.

The readback maximum absolute error was below `1e-3 rad` for the values that
were safe to set inside the Trossen limits, so:

```text
scatter_set_readback = PASS_SCATTER_SET_READBACK_ONLY
```

This means:

```text
candidate DOF index resolution works
set/readback is numerically stable enough for static checks
```

The script skipped four identity samples because the raw ALOHA1 value was
outside the Trossen runtime limit:

```text
skipped_identity_sample_count = 4
```

It does **not** mean:

```text
ALOHA1 sign is correct
ALOHA1 offset is correct
Trossen FK matches real ALOHA1
the gripper mapping is correct
```

## Updated Status

Confirmed:

```text
Trossen scaffold loads
Trossen arm candidate DOFs all exist
Trossen DOF order is interleaved
ALOHA1 real ROS joint order/IDs/limits are known
direct identity mapping is invalid
```

Blocked:

```text
sign
offset
FK equivalence
gripper carriage mapping
camera extrinsics
contact/material validation
controller replay
```

## Next Gate

The next stage should solve sign and offset without touching the real robot
first.

Preferred evidence order:

1. Use existing HDF5/ROS logs with multiple poses to infer candidate sign and
   offset.
2. Compare Isaac FK under each candidate sign/offset against a trusted ALOHA1
   FK chain if available.
3. If logs and FK are insufficient, design a real-hardware one-joint positive
   direction test as a separate safety-reviewed plan.

The real-hardware test must not be the default next action.
