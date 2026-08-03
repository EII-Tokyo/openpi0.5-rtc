# ALOHA1 Task 8 collider LOD comparison

- Final conclusion: `NO_MEASURABLE_IMPROVEMENT`
- Candidate promotion: `false`
- Authored upper-arm convex pieces: `8` → `2`
- Fresh cooking: `PASS_TWO_FRESH_PROCESSES`
- Static equivalence: `PASS_EQUIVALENT_TO_BASELINE_WITH_PREEXISTING_ABSOLUTE_GATE_FAILURE`
- 809-waypoint swept regression: `PASS_809_WAYPOINTS_TWO_FRESH_PROCESSES`
- Fidelity Bottle500 smoke: `PASS`
- Throughput Bottle500 smoke: `PASS`

Both smoke runs retained bilateral force-carrying solver contact and passed the 20 cm / 2 s hold gate. The exact `separation <= 0` count changed because the left minimum separation crossed zero by micrometres; signed values and impulses are preserved in JSON. This is not described as contact loss.

The candidate is geometrically valid but has no repeatable measurable performance benefit, so it remains diagnostic and is not promoted.
