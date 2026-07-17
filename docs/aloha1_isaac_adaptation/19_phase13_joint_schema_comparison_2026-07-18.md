# Phase 13 - Joint Schema Comparison - 2026-07-18

## Purpose

Phase 11 and Phase 12 showed that the Phase 9 left-arm mapping can match end-effector position moderately well, but cannot keep end-effector orientation consistent.

Phase 13 checks whether the failure is consistent with a deeper joint-schema mismatch between:

- trusted ALOHA1 VX300S URDF facts; and
- the Trossen `stationary_ai` USD joint schema used by the scaffold.

## Evidence

- Script: `aloha_isaac_replay/scripts/compare_aloha1_trossen_joint_schema.py`
- Full bounded run artifact: `.codex/artifacts/20260718-001053_phase13-joint-schema-comparison-v2`
- JSON report: `reports/aloha1_isaac_adaptation/phase13_joint_schema_20260718/joint_schema_comparison.json`
- Markdown report: `reports/aloha1_isaac_adaptation/phase13_joint_schema_20260718/joint_schema_comparison.md`

## Scope

This is a read-only static schema diagnostic:

- no real robot command;
- no stage save;
- no controller execution;
- no gripper/contact validation.

The official NVIDIA Isaac MCP was consulted before changing the Isaac/USD diagnostic code. The relevant rule from Isaac/USD is that USD joint axes and rotations are local-frame quantities, so raw axis tokens cannot be treated as world-frame directions.

## Gates

```text
real_robot_touched: PASS_FALSE
stage_saved: PASS_FALSE
isaac_runtime_started: PASS
aloha_urdf_loaded: PASS
trossen_usd_loaded: PASS
scaffold_usd_loaded: PASS
semantic_trossen_rows_present: PASS
controller: BLOCKED_NOT_ATTEMPTED
```

## Key Result

The ALOHA1 URDF parsed correctly:

```text
ALOHA1 parsed joints: 7
Trossen raw focus joints: 13
Scaffold focus joints: 13
```

The expected Trossen left arm rows are present, but their extracted axis semantics do not line up as a simple one-to-one ALOHA1 joint replacement.

| semantic | ALOHA1 axis in parent frame | Trossen joint | Trossen extracted body-frame axis |
|---|---|---|---|
| waist | `[0, 0, 1]` | `follower_left_joint_0` | `[0, 0, 1]` |
| shoulder | `[0, 1, 0]` | `follower_left_joint_1` | `[0, 1, 0]` |
| elbow | `[0, 1, 0]` | `follower_left_joint_2` | approximately `[0, -1, 0]` |
| forearm_roll | `[1, 0, 0]` | `follower_left_joint_3` | approximately `[0, -1, 0]` |
| wrist_angle | `[0, 1, 0]` | `follower_left_joint_4` | approximately `[0, 0, -1]` |
| wrist_rotate | `[1, 0, 0]` | `follower_left_joint_5` | `[1, 0, 0]` |

## Interpretation

This does not prove the exact corrected mapping by itself, because URDF and USD axes are both local-frame quantities and the connected body frames may differ.

However, it is strong evidence that the current Phase 9 mapping was too weakly constrained:

1. It optimized end-effector position only.
2. It allowed a mapping that can make the path shape look acceptable.
3. It did not require wrist and forearm orientation semantics to match.
4. The resulting controller candidate therefore fails orientation consistency by about 40 degrees p95.

The most suspicious joints are:

- `forearm_roll`;
- `wrist_angle`;
- `elbow` sign/offset;
- terminal link frame selection after the wrist chain.

## Decision

Do not proceed to controller work from the Phase 9 mapping.

The next valid phase is an orientation-aware mapping search. It must include both:

- end-effector position residual; and
- end-effector orientation residual after a fixed base/frame alignment.

That search should prioritize the left forearm and wrist chain rather than continuing to tune base/shoulder joints.

## Status

```text
BLOCKED_REQUIRES_ORIENTATION_AWARE_MAPPING_SEARCH
```
