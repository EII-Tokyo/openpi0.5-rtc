# Phase 35: Clean runtime stage single-joint response

## Question

Phase 34 proved that the clean runtime stage can be opened directly and that deterministic qpos set/readback is exact.

This phase asks the next stricter question:

Can the same clean runtime stage track small one-joint position commands through PhysX dynamics?

## Code change

Updated:

```text
aloha_isaac_replay/scripts/validate_aloha1_native_single_joint_response.py
```

New arguments:

```text
--stage-usd
--stage-units-in-meters
```

When `--stage-usd` is provided, the validator:

- opens the given stage directly with `stage_utils.open_stage`;
- does not call `add_reference_to_stage` for left/right wrappers;
- defaults articulation roots to:
  - `/puppet_left_vx300s/root_joint`
  - `/puppet_right_vx300s/root_joint`
- rejects `--base-separation`, because that option only applies to the older `/World/left` and `/World/right` reference-loading mode.

## Validation

Primary command:

```bash
codex-evidence --name phase35-clean-stage-single-joint-smoke -- \
  .venv_issac/bin/python \
  aloha_isaac_replay/scripts/validate_aloha1_native_single_joint_response.py \
  --stage-usd local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_clean_runtime.usda \
  --joint left:waist \
  --joint right:shoulder \
  --phase-offset 0 \
  --phase-offset 0.02 \
  --phase-offset 0 \
  --phase-steps 40 \
  --settle-steps 5 \
  --arm-kp 1000 \
  --arm-kd 100 \
  --output-dir reports/aloha1_isaac_adaptation/phase35_clean_stage_single_joint_smoke_20260718
```

Evidence:

```text
.codex/artifacts/20260718-015954_phase35-clean-stage-single-joint-smoke
```

Result:

| Check | Result |
| --- | --- |
| Validator status | `FAILED_GATE` |
| Clean stage opened | yes |
| Unresolved visual reference warnings | 0 observed in the result-specific failure path |
| Left articulation root | `/puppet_left_vx300s/root_joint` |
| Right articulation root | `/puppet_right_vx300s/root_joint` |
| Left tested joint | `waist` |
| Right tested joint | `shoulder` |
| Left result | FAIL |
| Right result | FAIL |

The dynamic response was unstable with the stage's centimeter unit setting:

| Joint | Max final absolute error |
| --- | ---: |
| left `waist` | 1805.4327 |
| right `shoulder` | 1456.0531 |

I also ran a controlled unit override:

```bash
codex-evidence --name phase35-clean-stage-single-joint-smoke-units1 -- \
  .venv_issac/bin/python \
  aloha_isaac_replay/scripts/validate_aloha1_native_single_joint_response.py \
  --stage-usd local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_clean_runtime.usda \
  --stage-units-in-meters 1.0 \
  --joint left:waist \
  --joint right:shoulder \
  --phase-offset 0 \
  --phase-offset 0.02 \
  --phase-offset 0 \
  --phase-steps 40 \
  --settle-steps 5 \
  --arm-kp 1000 \
  --arm-kd 100 \
  --output-dir reports/aloha1_isaac_adaptation/phase35_clean_stage_single_joint_smoke_units1_20260718
```

Evidence:

```text
.codex/artifacts/20260718-020045_phase35-clean-stage-single-joint-smoke-units1
```

This also failed, but the error scale was much smaller:

| Joint | Max final absolute error |
| --- | ---: |
| left `waist` | 2.8124 |
| right `shoulder` | 0.2879 |

## Interpretation

This phase separates two facts that were previously easy to mix together:

1. The clean runtime stage is now a valid composition target for direct opening.
2. The clean runtime stage does not yet have trustworthy dynamic drive tracking.

Phase 34's qpos replay gate used direct state setting and readback. It did not prove that Isaac's drive, mass, inertia, damping, timestep, unit scaling, and collision setup can track commands through the physics solver.

The `--stage-units-in-meters 1.0` control run reducing the failure magnitude suggests the dynamic response is sensitive to stage-unit handling. However, because that run still failed, the next fix should not be a blind unit flip. The next investigation must inspect:

- authored mass and inertia scale;
- drive stiffness and damping units;
- joint limits and axes after composition;
- articulation solver settings;
- whether `World(stage_units_in_meters=...)` should be omitted or aligned differently when opening an already-authored USD stage;
- whether the imported centimeter-scale asset needs an explicit normalized physics layer before dynamic control.

## Current status

This is a failed dynamic gate, not a blocker to the clean-stage composition work.

The clean-stage path is still useful and should be kept. The next phase should repair or normalize the dynamic physics layer until this same single-joint gate passes before moving to multi-joint replay, grasping, bottle contact, or RL simulation.

