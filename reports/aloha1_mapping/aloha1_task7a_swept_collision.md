# ALOHA1 Task 7A swept-collision validation

- Status: `FAIL`
- Stage SHA-256: `d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf`
- Cases: `48`
- Failed cases: `4`
- Determinism: `PASS`
- solve_articulation_contact_last: `true`
- Unique failed trajectories: `2`
- Contact-envelope-only pairs: `1`

## Deterministic failures

| Case | Target (rad) | Final readback (rad) | Physical pairs |
|---|---:|---:|---:|
| `follower_left:shoulder:positive` | `1.194503376` | `0.287970573` | `2` |
| `follower_right:shoulder:positive` | `1.194503376` | `0.287974000` | `2` |

Both positive-shoulder trajectories are stopped near `0.288 rad` when both supplier-CAD finger colliders physically contact `user_confirmed_table`. The same two failures reproduce in both fresh repeats.

## Interpretation boundary

This run preserves the authored collision filters and self-collision settings. PASS proves no unexpected reported contact along the tested trajectories under those settings; it does not prove disabled collision pairs are geometrically separated.

No source Stage, collider, drive, mimic, timestep, solver iteration, or final/default asset was modified.
