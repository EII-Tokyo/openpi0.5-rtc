# Phase 108 Diagnostic Table Contact Gate

## Question

Can the Phase107 full-scene BottleUSD HDF5 drive-target replay tolerate the current Phase63 fixed-table candidate?

This phase deliberately uses the diagnostic support-plane config:

```text
examples/aloha_isaac/config/phase63_fixed_table_candidate.yaml
```

It does not use `--require-calibrated-table-frame`, because Phase63 is explicitly not calibrated:

```text
T_table_left_base.status = not_calibrated
T_table_right_base.status = not_calibrated
```

## Runner

Added:

```text
aloha_isaac_replay/scripts/run_phase108_bottleusd_hdf5_diagnostic_table_gate.py
```

Command:

```bash
codex-evidence --name aloha-phase108-bottleusd-hdf5-diagnostic-table-runner -- \
  .venv/bin/python aloha_isaac_replay/scripts/run_phase108_bottleusd_hdf5_diagnostic_table_gate.py
```

Initial probe artifact:

```text
.codex/artifacts/20260719-005651_aloha-phase108-bottleusd-hdf5-diagnostic-table-probe
```

Structured report:

```text
reports/aloha1_isaac_adaptation/phase108_bottleusd_hdf5_diagnostic_table_probe_20260719/gripper_passive_contact_metrics.json
```

## Result

| Check | Result |
| --- | --- |
| status | `PASS` |
| contact trace status | `PASS_BILATERAL_CONTACT_CANDIDATE` |
| target contact persistence steps | `27` |
| non-target object categories | `workcell_or_environment` |
| non-target object gate | `PASS_NON_TARGET_CONTACTS_ALLOWED` |
| controller tracking gate | `PASS_POST_STEP_TRACKING_WITHIN_THRESHOLD` |
| max controlled error | `0.01287` |
| worst DOF | `left_shoulder` |

The replay remains numerically stable and tracks the HDF5 target within the existing threshold.

## Important Finding

The diagnostic Phase63 support plane itself did not contact the bottle:

```text
diagnostic_contact_summaries["/World/phase58_static_support_plane"].contact_pair_count = 0
```

The observed non-target object contacts came from the already imported `/scene/worldBody` workcell geometry, especially:

```text
/scene/worldBody/__22/collisions/__22/__22/extrusion_1220
```

This path is not the Phase63 support-plane prim. It is a collider already present in the imported scene.

## Exact Prim Audit

Added a generic composed-stage prim audit script:

```text
aloha_isaac_replay/scripts/inspect_stage_prims.py
```

Exact audit artifact:

```text
.codex/artifacts/20260719-010002_aloha-phase108-specific-contact-prim-audit-script
```

Structured audit:

```text
reports/aloha1_isaac_adaptation/phase108_specific_contact_prim_audit_20260719/stage_prim_audit.json
```

Key result:

| Prim | Meaning from bbox/schema |
| --- | --- |
| `/scene/worldBody/__22/collisions/__22/__22/extrusion_1220` | a mesh collider, about `1.22 m` long and `0.02 m x 0.02 m` thick |
| `/scene/worldBody/__22` | a rigid-body parent for that extrusion |
| `/scene/worldBody/table` | a smaller rigid-body table-like prim, about `0.744 m x 0.274 m x 0.02 m` |
| `/World/phase58_static_support_plane` | not present in the static stage audit because it is created at validator runtime |

This means the current contact gate is already seeing physical workcell collision, but the semantics are not yet clean:

```text
BottleUSD contact with extrusion_1220 != calibrated full tabletop support
```

## Interpretation

Phase108 proves a narrow engineering fact:

1. adding the Phase63 diagnostic table config does not break the Phase107 replay;
2. controller tracking remains good;
3. BottleUSD stays bounded under gravity;
4. the validator can classify workcell/environment contact separately from gripper target contact.

Phase108 also exposes a stronger blocker:

The current imported workcell already contains active colliders that can touch BottleUSD, but those colliders are not yet mapped into task-level semantics such as:

```text
tabletop
front rail
camera rack extrusion
pipe support
```

Without that mapping, a passing contact gate could still be physically misleading.

## What This Does Not Prove

Phase108 does not prove:

1. the tabletop is calibrated to the real ALOHA table;
2. the bottle is supported by a tabletop rather than a rail/extrusion;
3. pipe collision is present or correctly placed;
4. insertion into the pipe is physically valid;
5. the Phase63 diagnostic table candidate is a final workcell model.

## Decision

Keep Phase108 as a diagnostic contact-provenance gate, not as the final table gate.

The next phase should build a workcell-collider semantic map:

```text
/scene/worldBody/table          -> candidate tabletop or small plate
/scene/worldBody/__22/...       -> 1.22 m extrusion, likely rail/frame member
other /scene/worldBody children -> classify before allowing in final gates
```

Then the final replay gate can distinguish:

```text
allowed tabletop support
allowed pipe contact
allowed fixture contact
unexpected rail/frame collision
unexpected robot-body collision
```
