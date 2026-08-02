# ALOHA1 Task 7 PhysicsRules root-cause closure

- Status: `PARTIAL`
- Task 7: `PARTIAL`
- Task 8: `NOT_RUN`
- Frozen Stage SHA-256: `327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9`
- Validator fresh processes: `20`
- Runtime fresh processes: `20`
- Final/default asset modified: `false`

## Measured outcome

The isolated combined candidate reduces the original 20 standalone-follower PhysicsRules blockers to one unsuppressed `MimicAPICheck` per follower. Both followers pass two fresh 120-frame runtime probes with identical per-side signatures.

The straightforward helper-body removal is rejected. It reproducibly creates 57 non-adjacent collider-clash findings per follower. Four visually legible raw/annotated images identify the affected helper chain and collision region. Those images were not captured from a legal runtime finger readback and therefore do not validate supplier-CAD finger installation or finger-pair collision response; their absolute paths and hashes are in the JSON report.

The frame-preserving topology candidate avoids that clash regression, but removes `0.00300000014 kg` of source-authored helper mass per follower. Those values are source placeholders, not physically calibrated measurements. The candidate therefore cannot be promoted until its mass/COM/inertia semantics are preserved and the changed collider composition passes the accepted grasp regression.

## Remaining real blockers

- `HELPER_MASS_COM_INERTIA_SEMANTICS_NOT_PRESERVED_IN_TOPOLOGY_CANDIDATE`
- `COLLIDER_SPLIT_AND_TOPOLOGY_CANDIDATE_NOT_PROMOTED_OR_GRASP_REGRESSED`
