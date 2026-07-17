# Phase 34: Clean runtime stage qpos replay gate

## Question

Now that Phase 33 produced a clean local runtime stage:

```text
local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_clean_runtime.usda
```

can the existing batch qpos replay validation run against that stage directly, instead of using the older `/World/left` and `/World/right` defaultPrim reference path?

## Code change

Updated:

```text
aloha_isaac_replay/scripts/validate_aloha1_native_arm_qpos_replay_batch.py
```

New arguments:

```text
--stage-usd
--stage-units-in-meters
```

When `--stage-usd` is provided, the validator:

- opens the given stage directly with `stage_utils.open_stage`;
- does not call `add_reference_to_stage` for left/right wrappers;
- defaults stage units to `0.01`;
- defaults articulation roots to:
  - `/puppet_left_vx300s/root_joint`
  - `/puppet_right_vx300s/root_joint`

This is required because the clean runtime stage is a whole-stage composition. Going back through defaultPrim references would reintroduce the old zero-collider composition path.

## Validation

Command:

```bash
codex-evidence --name phase34-clean-stage-qpos-batch-smoke -- \
  .venv_issac/bin/python \
  aloha_isaac_replay/scripts/validate_aloha1_native_arm_qpos_replay_batch.py \
  --stage-usd local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_clean_runtime.usda \
  --hdf5-root local_rlt_data/raw_from_103/rollouts/key_regions/twist_off_the_bottle_cap/2026-06-17/rl \
  --episode-limit 2 \
  --max-frames-per-episode 20 \
  --output-dir reports/aloha1_isaac_adaptation/phase34_clean_stage_qpos_batch_smoke_20260718
```

Evidence:

```text
.codex/artifacts/20260718-015532_phase34-clean-stage-qpos-batch-smoke
```

Report:

```text
reports/aloha1_isaac_adaptation/phase34_clean_stage_qpos_batch_smoke_20260718/batch_replay_metrics.json
reports/aloha1_isaac_adaptation/phase34_clean_stage_qpos_batch_smoke_20260718/batch_replay_metrics.md
```

Result:

| Check | Result |
| --- | --- |
| Validator status | PASS |
| Episodes tested | 2 |
| Frames tested | 40 |
| Stage units | 0.01 |
| Left articulation root | `/puppet_left_vx300s/root_joint` |
| Right articulation root | `/puppet_right_vx300s/root_joint` |
| Max readback error | 0.0 |
| Unresolved reference warnings | 0 |

## Interpretation

The clean runtime stage can now pass the deterministic qpos set/readback gate through the same validation script family used in earlier phases.

This does not yet prove dynamic tracking or contact behavior. It does prove that:

1. the validator can target the clean stage directly;
2. the clean stage exposes usable left/right articulations;
3. deterministic qpos set/readback remains exact;
4. the old unresolved visual-reference warnings are gone when validating the final generated stage.

## Next step

Port the dynamic tracking and single-joint response validators to the same `--stage-usd` entry point. Then rerun them on:

```text
local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_clean_runtime.usda
```

Only after those gates pass should bottle/table/pipe contact simulation be trusted.

