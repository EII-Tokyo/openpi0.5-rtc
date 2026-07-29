# ALOHA1 Task 7A swept-collision validation

- Status: `PASS`
- Stage SHA-256: `d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf`
- Cases: `48`
- Failed cases: `0`
- Partial cases: `0`
- Determinism: `PASS`
- solve_articulation_contact_last: `true`
- Contact-limited workcell trajectories: `2`
- Contact-envelope-only pairs: `1`
- User-confirmed allowed finger/table contacts: `4`

## Workcell reachability boundary

| Case | Target (rad) | Final readback (rad) | Status |
|---|---:|---:|---|
| `follower_left:shoulder:positive` | `1.194503376` | `0.287970573` | `PASS` |
| `follower_right:shoulder:positive` | `1.194503376` | `0.287974000` | `PASS` |

The user confirmed that finger/table contact is allowed physical workcell behavior. These trajectories record a contact-limited workcell reachability boundary, not a control-direction or collider failure. Other robot/environment contacts remain forbidden unless separately classified with evidence.

## Interpretation boundary

This run preserves the authored collision filters and self-collision settings. PASS proves no unexpected reported contact along the tested trajectories under those settings; it does not prove disabled collision pairs are geometrically separated.

No source Stage, collider, drive, mimic, timestep, solver iteration, or final/default asset was modified.
