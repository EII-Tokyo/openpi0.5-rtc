# ALOHA Viper follower_right Stage audit

- Status: `PASS`
- Candidate Stages: `31`
- Eligible current supplier-CAD Stages: `1`
- CAD availability: `VERIFIED_SINGLE_REUSABLE_ROBOT_PRODUCT`
- Next action: `VALIDATE_EXISTING_FOLLOWER_RIGHT_STAGE`
- Protected inputs unchanged: `True`
- Scope: read-only, local Isaac Sim 5.1 USD composition evidence.
- Task 8: `NOT_RUN`

## Result

At least one independent follower_right Stage contains both handed supplier-v2 mesh source hashes.

## Classification counts

- `ELIGIBLE_CURRENT_SUPPLIER_CAD`: 1
- `HISTORICAL_GYM_ALOHA_NOT_CURRENT_SUPPLIER_CAD`: 2
- `REJECTED_ALOHA2_OR_LEGACY_REBUILD`: 7
- `REJECTED_GENERIC_FINGER`: 3
- `REJECTED_NOT_INDEPENDENT_FOLLOWER_RIGHT_ARTICULATION`: 6
- `REJECTED_PHANTOM_RIGHT_BRANCH`: 7
- `REJECTED_STAGE_OPEN_FAILED`: 1
- `UNKNOWN_FINGER_PROVENANCE`: 4

## HARD_BLOCKER

- `HARD_BLOCKER_APPROVED_STAGE_MISSING_FOLLOWER_RIGHT`
- `HARD_BLOCKER_FOLLOWER_RIGHT_WORKCELL_INSTALL_TRANSFORM`

The approved review Stage remains immutable and contains no `/workcell/vx300s_right`; that absence is scoped only to the approved left review Stage. No rejected or historical asset was promoted. The workcell placement transform remains a separate HARD_BLOCKER from robot-local asset generation.
