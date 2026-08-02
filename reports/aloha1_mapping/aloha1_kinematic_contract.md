# ALOHA1 kinematic contract

- Status: **PASS**
- Explicit joint order: `['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']`
- ID 6/7 conflict gate: **PASS_RESOLVED_WITH_CONFLICT_RETAINED**
- Left/right robot-local identity: **PASS**
- Maximum FK translation residual: `4.48405047021e-16 m`
- Maximum FK rotation residual: `2.20758714493e-16 rad`
- Maximum Jacobian residual: `1.95809590764e-10`

The official Trossen POE model and an independent URDF-chain implementation were compared at home and four deterministic legal joint samples. Isaac IK was not called. Left and right are identical robot-local products, not mirrored; this report makes no claim about their workcell installation transforms.

Tolerances are derived from published decimal precision and the finite-difference error expression recorded in the JSON report; no behavior-fitted tolerance is used.
