# Phase 109 Workcell Semantic Audit

## Question

Phase108 showed that the BottleUSD HDF5 replay contacts `/scene/worldBody/...` geometry, but that category is too broad.

Phase109 asks:

Can we identify which workcell collider actually touches the bottle, and whether it should be treated as tabletop support?

## Tooling

Added:

```text
aloha_isaac_replay/scripts/inspect_workcell_semantics.py
```

Command:

```bash
codex-evidence --name aloha-phase109-workcell-semantic-audit -- \
  .venv_issac/bin/python aloha_isaac_replay/scripts/inspect_workcell_semantics.py \
  --stage-usd local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose/aloha2_menagerie_scene_deep_black_real_start_pose_proxy_runtime.usda \
  --contact-report reports/aloha1_isaac_adaptation/phase108_bottleusd_hdf5_diagnostic_table_gate_20260719/gripper_passive_contact_metrics.json \
  --output-dir reports/aloha1_isaac_adaptation/phase109_workcell_semantic_audit_20260719
```

Artifact:

```text
.codex/artifacts/20260719-010258_aloha-phase109-workcell-semantic-audit
```

Structured report:

```text
reports/aloha1_isaac_adaptation/phase109_workcell_semantic_audit_20260719/workcell_semantic_audit.json
```

## Result

The audit found these semantic counts under `/scene/worldBody`:

| Semantic guess | Count |
| --- | ---: |
| `long_thin_frame_member` | `55` |
| `small_fixture_or_bracket` | `52` |
| `unknown_workcell_geometry` | `53` |
| `unknown_no_bbox` | `35` |
| `table_named_prim` | `6` |
| `tabletop_or_plate_candidate` | `2` |
| `floor` | `6` |

The contact-relevant object path was:

```text
/scene/worldBody/__22/collisions/__22/__22/extrusion_1220
```

Its parent `/scene/worldBody/__22` has:

| Field | Value |
| --- | --- |
| semantic guess | `long_thin_frame_member` |
| bbox center | `[0.0, 0.36899998784065247, 0.009999999776482716]` |
| bbox size | `[1.2200000286102297, 0.019999999552965275, 0.01999999955296544]` |
| applied schemas | `PhysicsRigidBodyAPI`, `PhysxRigidBodyAPI` |

This is a long, thin 1.22 m frame or rail member.

It is not the named table prim.

The named table prim exists:

```text
/scene/worldBody/table
```

but its bbox is only approximately:

```text
[0.7442, 0.2738, 0.0200] m
```

That is not the full measured table size.

## Interpretation

Phase109 changes the meaning of the current contact gates.

Before this audit, `workcell_or_environment` was too broad:

```text
BottleUSD touched something in the imported workcell.
```

After this audit, the specific statement is:

```text
BottleUSD touched a 1.22 m long, 2 cm thick frame/rail collider.
```

That is not enough to claim realistic tabletop support.

## Why This Matters

For bottle insertion, these contacts have different meanings:

| Contact type | Meaning |
| --- | --- |
| bottle with calibrated tabletop | expected support before grasp or during setdown |
| bottle with pipe inner wall | task-relevant insertion contact |
| bottle with pipe edge | task-relevant near-miss or scraping |
| bottle with frame/rail extrusion | likely false physics contact unless the real bottle is actually touching that rail |
| bottle with gripper base or arm link | invalid grasp/contact setup |

If all of these are collapsed into `workcell_or_environment`, a gate can pass while validating the wrong physical interaction.

## Decision

Phase109 is now the semantic blocker before final table/pipe replay:

```text
workcell_or_environment must be split into specific allowed contact classes.
```

The next implementation step should introduce an allow/deny policy such as:

```text
allowed_tabletop_support
allowed_pipe_contact
allowed_fixture_contact
denied_frame_rail_collision
denied_robot_body_collision
unknown_workcell_collision
```

Until this exists, final insertion replay should not treat `workcell_or_environment` as globally acceptable.

## Next Gate

Phase110 should add a contact policy file that maps workcell collider paths to semantic contact classes.

The first conservative policy should classify:

```text
/scene/worldBody/__22/** -> denied_frame_rail_collision
/scene/worldBody/table/** -> candidate_table_prim_not_full_measured_table
```

Then rerun Phase108 with this policy and verify that the current replay fails for the right reason:

```text
FAIL_DENIED_FRAME_RAIL_COLLISION
```

That negative control is necessary before trusting any future PASS.
