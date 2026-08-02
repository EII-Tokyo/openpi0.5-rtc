# ALOHA1 finger limit and pair-collision semantics

- Status: `PASS`
- Limit semantics: `VERIFIED_USD_LIMIT_DEFECT`
- Pair-collision support: `INCONCLUSIVE`
- Candidate created: `True`
- Task 8: `NOT_RUN`

## Source versus live limits

- URDF left: `{'lower': 0.021, 'upper': 0.057}`
- Live left: `{'lower': 0.020999999716877937, 'upper': 0.05700000002980232}`
- URDF right: `{'lower': -0.057, 'upper': -0.021}`
- Live right: `{'lower': -0.06419999897480011, 'upper': -0.013799999840557575}`

The source URDF remains the admissible runtime interval. A wider imported/composed right-finger interval is an asset defect, not a license to command beyond the source limit.

## Pair-collision boundary

Local 5.1 exposes articulation-wide self-collision and a filtered-pairs API that disables selected pairs. This read-only probe did not enable all internal self-collisions merely to obtain one positive finger pair, so no pair-collision candidate is claimed.

## Isolated candidate

`/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/finger_limit_pair_collision_candidate/1.0/aloha1_finger_source_limit_candidate.usda` is NOT_PROMOTED.
