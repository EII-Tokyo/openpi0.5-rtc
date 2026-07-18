# Phase 61 User-Measured Workcell Regeneration

## Question

After Phase 60, the measured table dimensions were corrected to:

```text
length = 1.220 m
width  = 0.625 m
```

Phase 61 asks:

Does the generated user-measured Isaac workcell USD actually contain those updated dimensions and the derived pipe marker positions?

## Command Artifact

```text
.codex/artifacts/20260718-144653_phase61-rebuild-user-measured-workcell-stage
```

Generated USD:

```text
local_eval_assets/aloha_isaac_user_measured/aloha_workcell_user_measured.usda
```

This generated USD is a local runtime asset, not a committed source file.

## Verified USDA Evidence

Text inspection of the generated USDA found:

```text
/World/Table
xformOp:scale = (1.22, 0.625, 0.04)
xformOp:translate = (0, 0, -0.02)
```

The derived pipe geometry also reflects the corrected table size:

```text
pipe visual center = (-0.11095, 0.4075, 0.14815)
pipe end marker    = (-0.1919, 0.4075, 0.2263)
pipe support center = (-0.03, 0.4075, 0.025)
measurement A      = (-0.03, 0.3125, 0)
base offset marker = (-0.03, 0.36, 0)
```

## Interpretation

The user-measured workcell generator is now aligned with the latest measured table dimensions.

This fixes a previous inconsistency:

| Source | Old value | New value |
| --- | ---: | ---: |
| `workcell_user_measured.yaml` table size | `1.10 x 0.60 m` | `1.22 x 0.625 m` |
| README table description | `1.10 x 0.60 m` | `1.22 x 0.625 m` |
| pipe A point | `(0.03, 0.30, 0)` | `(-0.03, 0.3125, 0)` |

## Limitation

This validates the generated workcell geometry file, not full dynamic contact.

The actual Phase 60 dynamic table candidate still used object-bottom placement inside the validator. A final table contact gate still needs a fixed table prim in the same runtime stage used by the HDF5 replay.

## Next Gate

Phase 62 should connect these two tracks:

1. load or sublayer a fixed measured table prim into the left-arm replay runtime stage;
2. run the already-grasped HDF5 replay under gravity;
3. classify table-object, table-finger, and unexpected contacts;
4. reject any fixed table pose where the fingertip proxy collides with the table.

