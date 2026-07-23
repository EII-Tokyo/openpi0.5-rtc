# A21 Policy Target Limit And Runtime Readback Design

## Goal

Advance the A19 clean ALOHA articulation from read-only discovery to a
bounded target-path validation gate without stepping physics, moving links,
changing drive gains, enabling collision, or saving the stage.

A21 must prove two separate facts:

1. every supported 14D ALOHA/OpenPI command expands into 16 finite Isaac DOF
   targets inside the live runtime limits; and
2. those targets reach exactly the intended raw PhysX indices, can be read
   back, and can be restored in a fresh headless process.

A21 does not claim motion, hold, collision, contact, replay, or training
readiness.

## Current Evidence And Problem

A20 passes Asset Validator, authored-USD metadata validation, and three-process
no-step articulation discovery. The live articulation has 16 DOFs. Its raw
order alternates the twelve arm DOFs left/right and then groups the four finger
DOFs.

The current A20 policy adapter passes a mathematical
`14D -> 16D -> 14D` round trip, but it does not check the resulting targets
against runtime joint limits. That missing check exposes a real mismatch:

- `/aloha/joints/left_right_finger` and
  `/aloha/joints/right_right_finger` currently use `offset=-0.021` and
  `scale=-0.036`;
- the A19 authored and PhysX runtime limits for both DOFs are positive
  `0.01844 .. 0.058` metres;
- therefore all normal normalized gripper targets produced for these two DOFs
  are outside the current Isaac limits.

The negative mapping originates in URDF mimic-joint semantics recorded by
`configs/aloha/original_stationary_aloha_mapping.yaml`. The imported ALOHA1
Isaac USD has already expressed the mirrored finger motion in its joint frames;
its two finger DOF coordinates and limits are both positive. Reusing the URDF
mimic sign in the clean Isaac runtime adapter applies the mirror twice.

The A19 joint drives also have zero position stiffness. A position target is
therefore only a target-path test at A21; it must not be described as proof
that the articulation can move to or hold that target.

## Approaches Considered

### Selected: A21a Limit Preflight, Then A21b No-Step Target Readback

First reconcile the clean-runtime mapping explicitly and require a pure,
fail-closed limit preflight. Only after that passes may a separate fresh
headless Isaac process write tiny position targets, read them back, and restore
the original targets without advancing simulation time.

This preserves a narrow failure domain: mapping and unit failures are found
before runtime mutation, while target-index and API failures are checked
without introducing motion dynamics.

### Rejected: Test Only The Twelve Arm DOFs

This would reach a runtime write sooner, but it would leave both 14D gripper
commands invalid and falsely suggest that the complete ALOHA policy interface
is ready. A21 is a 14D-policy/16-DOF gate, not an arm-only demonstration.

### Rejected: Combine Mapping, Gain Tuning, Motion, And Collision

Adding nonzero stiffness and stepping physics in the same gate would make it
impossible to distinguish mapping errors from gain, inertia, frame, collider,
or solver errors. It would also repeat the earlier uncontrolled Play failure.
Drive tuning and actual micro-motion belong to A22.

## Clean-Runtime Mapping Reconciliation

The repository-wide
`configs/aloha/original_stationary_aloha_mapping.yaml` remains unchanged
because it records source robot-description semantics and may serve legacy
paths.

The clean reconstruction config gains explicit per-path runtime transform
overrides for:

- `/aloha/joints/left_right_finger`;
- `/aloha/joints/right_right_finger`.

For each clean Isaac DOF, the override uses:

```text
offset = 0.021 m
scale = 0.036 m
sign = +1
policy value 0 -> 0.021 m
policy value 1 -> 0.057 m
```

The override must include a rationale and provenance indicating that it is a
clean Isaac DOF-coordinate transform verified against the imported USD and
runtime limits. It must not silently overwrite the source mapping. The A17
mapping artifact and A20 policy contract must record both the original
transform and the applied clean-runtime override.

The override is accepted only when both normalized endpoints are finite,
strictly inside or on the configured joint limits with a small numeric
tolerance, and the transform is monotonic in the expected ALOHA gripper
opening convention.

## A21a: Pure Policy-Target Limit Preflight

A focused pure-Python component consumes:

- the versioned A20 order adapter;
- the exact A20 runtime DOF records and limits;
- the clean-runtime override provenance;
- deterministic 14D policy samples.

It performs no Isaac import and no filesystem mutation. It validates:

- schema versions, input hashes, dimensions, and unique DOF paths;
- all 14 policy indices and all 16 raw runtime indices;
- finite offsets, scales, policy values, and expanded targets;
- radians for revolute runtime targets and metres for prismatic targets;
- every expanded target against the path-aligned runtime limit;
- both gripper endpoints `0`, `0.5`, and `1`;
- a home/baseline arm vector and signed `0.25 degree` arm perturbations that
  remain inside their limits;
- `16D -> 14D` inverse agreement after every expansion;
- paired-finger agreement in normalized policy space.

The preflight fails closed before launching Isaac if any target is invalid.
Its success status is:

`PASS_A21_POLICY_TARGET_LIMIT_PREFLIGHT`

## A21b: Runtime Target Write, Readback, And Restore

A21b launches a fresh headless Isaac Sim process and follows the already
reviewed A20 stage-open and tensor-view initialization sequence. It does not
call timeline Play, `World.step`, `SimulationContext.step`, `simulate`,
`update_simulation`, application update, or any equivalent time-advancing
operation.

The probe:

1. opens the exact hash-bound A19 stage;
2. creates exactly one 16-DOF tensor articulation view;
3. reads live DOF limits, current positions, and current position targets;
4. verifies the baseline targets are finite and in limits;
5. writes one reviewed micro-target batch;
6. immediately reads position targets back;
7. checks that intended paths changed by the exact requested deltas and all
   non-target paths remained unchanged;
8. restores the complete original 16-DOF target array;
9. reads targets back again and verifies restoration;
10. closes the fresh process without saving the USD.

Target write/readback is performed in two explicit batches, each in a fresh
process:

- Batch L: six left-arm policy indices and the paired left gripper;
- Batch R: six right-arm policy indices and the paired right gripper.

Arm target deltas are deterministic signed values with magnitude at most
`0.25 degree` (`0.00436332313 rad`). Finger deltas are at most `0.00025 m`
and are selected toward the interior of the live limit interval. Each batch
uses path-resolved runtime indices from the A20 adapter, never assumed array
positions.

The probe records that targets were written, but no action or physical step
occurred:

```text
physics_stepped = false
actions_applied = false
targets_written = true
targets_restored = true
stage_saved = false
```

Its success status is:

`PASS_A21_RUNTIME_TARGET_READBACK_RESTORED_NO_STEP`

## Failure Semantics

A21 fails closed for any of the following:

- missing, duplicate, stale, or hash-mismatched inputs;
- an unreviewed or provenance-free runtime mapping override;
- a non-finite value, invalid unit, or target outside runtime limits;
- a policy/runtime round-trip disagreement;
- a runtime joint inventory or order different from the bound A20 evidence;
- failure to read the original target array;
- a readback mismatch at an intended index;
- any change at a non-target index;
- failure to restore the complete original target array;
- any physics step, action application, gain/effort/velocity write, stage
  save, or USD file hash change;
- a probe crash, timeout, extra JSON marker, or unclean child-process exit.

A failed Batch L prevents Batch R from running. Any runtime failure keeps the
overall A21 status `NOT_READY`.

## Components And Files

The implementation plan may create focused A21 modules rather than expanding
the already large A20 coordinator:

- a pure target-plan and limit-validation module;
- a one-shot Isaac runtime target-readback probe;
- a two-batch coordinator with bounded subprocess output;
- focused unit and integration tests;
- versioned JSON evidence and a bounded Markdown report.

The reconstruction config supplies output paths and clean-runtime overrides.
Existing A17/A20 generators are changed only where necessary to preserve
original and effective transform provenance.

Expected generated outputs:

- `aloha_isaac_rebuild/artifacts/validation/a21_policy_target_limit_preflight.json`;
- `aloha_isaac_rebuild/artifacts/validation/a21_runtime_target_readback.json`;
- `aloha_isaac_rebuild/reports/a21_target_limit_and_readback.md`.

## Testing And Verification

Implementation follows strict test-driven development.

Pure tests cover:

- the current negative right-finger transform failing against positive runtime
  limits;
- the explicit clean-runtime override passing gripper endpoints and midpoint;
- rejection of an override that changes legacy/source mapping in place;
- all 14 policy indices and 16 runtime indices;
- radians/metres unit handling;
- arm and finger perturbation clamping toward the limit interior;
- duplicate paths, stale hashes, invalid dimensions, non-finite values, and
  inverse disagreement;
- exact intended-index changes, non-target immutability, and full restoration
  using a fake tensor articulation view;
- failure propagation from Batch L to the coordinator;
- safety flags and report status semantics.

Before live A21b execution, the A19/A20 regression suite, A19 static audit,
A20 Asset Validator, A20 Layer 1, and three-process A20 Layer 2 must still
pass. Live probe output goes through `codex-evidence`; full Isaac logs remain
in `.codex/artifacts/`.

After A21b, the A19 USD SHA-256 must exactly match its pre-run hash. No real
robot process, ROS bridge, HDF5 replay, contact object, camera stream, or
training process is involved.

## Acceptance Criteria

A21 is complete only when:

1. source and effective clean-runtime gripper transforms are both explicit and
   provenance-bound;
2. all reviewed 14D samples expand to 16 finite, in-limit runtime targets;
3. both fresh runtime batches discover the exact A20 articulation contract;
4. every requested target is read back at the correct path-resolved index;
5. no non-target index changes;
6. the complete baseline target array is restored and verified;
7. physics is never stepped and the USD hash is unchanged;
8. all A19/A20/A21 tests and prerequisite gates pass;
9. the report explicitly keeps motion, hold, collision, contact, replay, and
   training readiness false.

## Route To Collision And Scene Accessories

After A21:

- A22 defines reviewed temporary/runtime drive gains, then performs
  single-joint micro-motion, direction, settling, and gravity-off hold checks;
- A23 validates collision geometry statically, then runs bounded self-collision
  and fixed-environment collision tests without replay;
- after A23, collision-enabled support-frame and water-pipe layers may be
  integrated;
- lower-camera visuals, transforms, and sensor configuration may be integrated
  earlier as a collision-disabled reference layer, but its housing collider is
  enabled only after A23.

No A21 result authorizes pressing Play in the GUI or proceeding directly to
contact, replay, or RL.
