# ALOHA1 Task 8 collider LOD candidate

- Status: `PASS_STATIC_GEOMETRY_CERTIFICATE`
- Candidate: `DIAGNOSTIC_ONLY_NOT_PROMOTED`
- Modified link suffix: `upper_arm_link` on both followers
- Authored convex pieces: `8` → `2`
- Retained existing source piece: `piece_000`
- Maximum containment residual: `7.6327832943e-17 m`
- Derived numerical tolerance: `2.47821822086e-06 m`
- New or reshaped collider geometry: `none`
- Gripper/finger/Bottle500/table collider changes: `none`
- Runtime cooking: `PASS_TWO_FRESH_PROCESSES`
- Static regression: `PASS_EQUIVALENT_TO_BASELINE_WITH_PREEXISTING_ABSOLUTE_GATE_FAILURE`
- Swept regression: `PASS_809_WAYPOINTS_TWO_FRESH_PROCESSES`
- Bottle500 smoke: `PASS_LIGHTWEIGHT_BOTTLE500`
- Final/default promotion: `false`

The selected candidate keeps the already-authored `piece_000` convex hull and only deactivates three source pieces proven to lie inside it. Two fresh cooking runs, static/swept comparison and one representative grasp smoke were completed.

The earlier full single-hull hypothesis remains `REJECTED_GEOMETRIC_OVERAPPROXIMATION` because its sampled outward deviation was `0.053817584 m`.

The candidate remains diagnostic because the benchmark found no stable, non-overlapping performance improvement.
