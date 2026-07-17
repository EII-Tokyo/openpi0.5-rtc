# Phase 33: Clean runtime asset package

## Question

Can we create a separate local ALOHA1 runtime USD package that:

1. does not modify the original importer output;
2. removes the six known broken visual reference arcs;
3. preserves collision, rigid body, mass, and articulation composition;
4. opens as a clean Isaac runtime stage without unresolved visual-reference warnings?

## Method

New generator:

```text
aloha_isaac_replay/scripts/build_aloha1_clean_runtime_asset.py
```

Command used:

```bash
codex-evidence --name phase33-build-clean-runtime-asset -- \
  .venv_issac/bin/python \
  aloha_isaac_replay/scripts/build_aloha1_clean_runtime_asset.py \
  --output-dir local_eval_assets/aloha1_clean_runtime_20260718 \
  --overwrite
```

The generator:

- copies the original imported ALOHA1 configuration USD files into a local generated package;
- patches only copied base layers;
- clears the six broken visual reference arcs:
  - three left fixed-link visual references;
  - three right fixed-link visual references;
- generates left/right clean side wrappers;
- generates a dual-arm runtime stage.

Generated local asset package:

```text
local_eval_assets/aloha1_clean_runtime_20260718/
```

This directory is ignored by Git and is not committed.

## Build result

Report:

```text
local_eval_assets/aloha1_clean_runtime_20260718/clean_runtime_asset_report.json
local_eval_assets/aloha1_clean_runtime_20260718/clean_runtime_asset_report.md
```

Generated runtime stage:

```text
local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_clean_runtime.usda
```

Summary from the build report:

| Check | Result |
| --- | --- |
| Copied original importer configuration | PASS |
| Original importer files modified | NO |
| Broken visual references removed from copied base layers | PASS |
| Static missing local reference target count | 0 |
| Collision API count | 22 |
| Rigid body API count | 28 |
| Mass API count | 28 |
| Left articulation init during build | PASS |
| Right articulation init during build | PASS |

The build log still contains six unresolved-reference warnings. Those warnings happen while the generator opens the copied base layers before removing their broken references. Therefore the build log is not the final clean-runtime validation.

## Runtime-only validation

After generation, the final stage was opened separately without re-copying or re-patching:

```bash
codex-evidence --name phase33-validate-generated-clean-runtime -- \
  .venv_issac/bin/python <runtime-only validation snippet>
```

Evidence:

```text
.codex/artifacts/20260718-015307_phase33-validate-generated-clean-runtime
```

Result:

| Check | Result |
| --- | --- |
| Runtime stage opens | PASS |
| Left articulation init | PASS, 9 DOF, 14 bodies |
| Right articulation init | PASS, 9 DOF, 14 bodies |
| Unresolved reference warning count | 0 |

This validates the generated clean runtime stage, not just the generator's internal report.

## Interpretation

This is the first ALOHA1 Isaac runtime asset in this sequence that satisfies all three requirements together:

1. collision/mass/rigid-body composition exists;
2. both arms initialize as Isaac articulations;
3. runtime opening of the final generated stage does not emit unresolved visual-reference warnings.

This makes it a better base for future controller, contact, bottle, table, and pipe tests than the older `/World/left` / `/World/right` defaultPrim reference path, which had zero composed collision prims.

## Important limitation

The generated package is local and ignored by Git:

```text
local_eval_assets/aloha1_clean_runtime_20260718/
```

Only the generator script is versioned. This avoids committing multi-megabyte generated binary USD files until the asset layout is stable enough to promote intentionally.

## Next step

Future runtime and controller scripts should accept:

```text
local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_clean_runtime.usda
```

or regenerate it from:

```text
aloha_isaac_replay/scripts/build_aloha1_clean_runtime_asset.py
```

Then rerun the qpos replay and dynamic tracking gates against this clean runtime stage rather than the older zero-collider reference composition.

