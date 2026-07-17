# Phase 20: DOF / Drive / Limit Validation

## Goal

After Phase 19 proved that the ALOHA1 native wrapper candidate can initialize as Isaac runtime articulations, this phase checks whether the joint-level physics metadata is good enough for the next step: replaying real ALOHA1 joint trajectories.

This phase is still read-only. It does not tune stiffness, damping, friction, mass, contacts, or controller gains.

## Official Isaac Basis

The official NVIDIA Isaac MCP physics and robot setup guidance was used before implementation.

Relevant Isaac validation principles:

- non-fixed active joints need drive or mimic configuration;
- max velocity and max effort should be present and positive for active joints;
- mimic joints should not be evaluated as normal actively driven joints;
- joint limits, drive values, and joint state consistency must be checked before controller work;
- a robot that initializes as an articulation is not automatically ready for policy/control replay.

## Script

```text
aloha_isaac_replay/scripts/validate_aloha1_native_dof_drive_limits.py
```

Command:

```bash
.venv_issac/bin/python aloha_isaac_replay/scripts/validate_aloha1_native_dof_drive_limits.py
```

Evidence artifact:

```text
.codex/artifacts/20260718-004551_phase20-dof-drive-limit-validation-mimic-aware
```

Report:

```text
reports/aloha1_isaac_adaptation/phase20_dof_drive_limits_20260718/dof_drive_limits.json
reports/aloha1_isaac_adaptation/phase20_dof_drive_limits_20260718/dof_drive_limits.md
```

## Result

The mimic-aware validation passed:

```text
status = PASS
overall_pass = true
```

| Side | DOFs | finite ordered limits | active effort | positive velocity | static joints found | mimic found | Gate |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| left | 9 | PASS | PASS | PASS | PASS | PASS | PASS |
| right | 9 | PASS | PASS | PASS | PASS | PASS | PASS |

## Important Interpretation

Each side exposes 9 runtime DOFs:

```text
waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, gripper, left_finger, right_finger
```

The first 6 are the arm joints. They have finite ordered limits, positive effort limits, and positive velocity limits.

The `right_finger` DOF is not an actively driven joint. It is marked with:

```text
PhysxMimicJointAPI:rotY
```

Therefore, its zero effort is not a failure. It should be treated as a passive mimic joint.

## Remaining Gripper Warning

The `gripper` DOF reports a very large runtime limit:

```text
-3.4028234663852886e+38 to 3.4028234663852886e+38
```

This is mathematically finite, but semantically it is effectively unbounded. Do not interpret this as a validated real ALOHA gripper range.

Current conclusion:

- arm DOF metadata is good enough for qpos replay validation;
- gripper/finger semantics still require a dedicated calibration phase;
- full grasp simulation must not proceed from this gate alone.

## Why The First Gate Failed

The first version of this validator incorrectly treated every runtime DOF as actively driven. That caused `right_finger` to fail because it has zero effort.

After inspecting the static USD joint schemas, the validator was corrected:

- active joints require positive effort;
- mimic joints are reported separately;
- mimic joints do not fail the active-drive gate.

This matches Isaac's robot validation guidance for mimic joints.

## Next Gates

The next valid step is now:

1. Replay a real ALOHA1 arm-only qpos trajectory into this candidate asset.
2. Compare commanded qpos and Isaac readback qpos.
3. Only after arm replay is stable, calibrate gripper/finger semantics.
4. Only after gripper semantics are clear, attempt bottle grasp/release logic.

Do not tune stiffness/damping yet. First verify that the asset follows real recorded ALOHA1 joint trajectories.
