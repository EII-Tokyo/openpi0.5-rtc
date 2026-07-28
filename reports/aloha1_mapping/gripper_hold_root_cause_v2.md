# ALOHA1 Gripper Hold Root Cause V2

- Status: `PARTIAL`
- Root cause: `inconclusive`
- Contributing causes: `none`

## Subsystem results

- contact_semantics: `VERIFIED_PHYSICAL_CONTACT`
- normal_force: `INSUFFICIENT`
- material: `SUFFICIENT`
- friction: `INCONCLUSIVE`
- static_hold: `FAIL`
- mimic_accuracy: `PRIOR_AB_EXPLICIT_AND_MIMIC_TRAJECTORIES_IDENTICAL`
- solver: `INCONCLUSIVE`
- determinism: `PASS`

## Interpretation boundary

The fixed-bottle preload measurements prove that stable bilateral
normal-force delivery is insufficient for the temporary 20 g,
mu=0.7 diagnostic threshold. They do not distinguish insufficient
commanded preload from maxForce saturation: the available measured
joint effort is solver force, not applied drive force.

Unresolved observations:

- `drive_vs_max_force_not_observable`
- `kinematic_to_dynamic_release_transient`

The 40/40 dynamic-release failures are deterministic numerical
ejection/release transients followed by contact loss and free fall.
Contact persistence is not treated as a physical hold pass.
Convex Decomposition improved fit but did not solve static hold.
Task 8 remains `NOT_RUN`; the final collider is unchanged.
