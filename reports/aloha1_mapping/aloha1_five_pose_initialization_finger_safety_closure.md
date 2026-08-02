# ALOHA1 five-pose initialization and finger-safety closure

- Task 7: `PARTIAL`
- Historical grasp outcome: `PASS`
- Attempt10 finger safety: `PASS`
- Attempt10 collision screenshot review: `PASS`
- Negative controls: `PASS`
- Source-limit semantics: `VERIFIED_USD_LIMIT_DEFECT`
- Physical pair collision: `NOT_AUTHORED_INCONCLUSIVE`
- Final/default promotion: `NOT_PROMOTED`
- Task 8: `NOT_RUN`

| Sample | Runtime/safety | Hold drop (m) | Clearance (m) |
|---|---:|---:|---:|
| `sample_01` | `PASS` | 0.000193937320122628 | 0.20018182360669862 |
| `sample_02` | `PASS` | 0.006202900385199744 | 0.20120438132460197 |
| `sample_03` | `PASS` | 0.0007431492529592909 | 0.20042036757849827 |
| `sample_04` | `PASS` | 0.0016494691683948959 | 0.20158486329596476 |
| `sample_05` | `PASS` | 0.000842787943897666 | 0.20056605252113302 |

The five previously user-confirmed videos were not rerun. They remain evidence of grasp outcome. Attempt10 adds ten fresh-process runtime records, per-frame finger-limit/overlap guards and 240 hash-bound raw/annotated collision images.

Task 7 remains `PARTIAL` because passing a diagnostic session layer is not permission to promote it into final/default assets, and the independently tracked PhysicsRules candidates remain unpromoted. This is an asset-promotion boundary, not a grasp failure.

## Remaining real blockers

- `FINGER_SOURCE_LIMIT_SESSION_LAYER_NOT_PROMOTED`
- `HELPER_MASS_COM_INERTIA_SEMANTICS_NOT_PRESERVED_IN_TOPOLOGY_CANDIDATE`
- `COLLIDER_SPLIT_AND_TOPOLOGY_CANDIDATE_NOT_PROMOTED_OR_GRASP_REGRESSED`
