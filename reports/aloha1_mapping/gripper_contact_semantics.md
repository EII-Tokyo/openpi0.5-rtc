# ALOHA1 Gripper Contact Semantics

- Status: `PASS`
- CONTACT_SEMANTICS_STATUS: `VERIFIED_PHYSICAL_CONTACT`
- Runtime: Isaac Sim 5.1.0.0 / Kit 107.3.3 / PhysX 107.3.26.
- Positive separation is classified only after comparison with an independent closest-point/AABB distance.

## Per-profile evidence

| Approximation | Robot | Side | First separation (m) | Independent distance (m) | Min separation (m) | Classification |
|---|---|---|---:|---:|---:|---|
| convexHull | follower_left | left | 0.01082709338515997 | 0.010778795985279907 | -4.1574226088414434e-06 | VERIFIED_PHYSICAL_CONTACT |
| convexHull | follower_left | right | 0.010875079780817032 | 0.011108541479667534 | 2.4996720640046988e-06 | VERIFIED_PHYSICAL_CONTACT |
| convexHull | follower_right | left | 0.01082709338515997 | 0.010778795985279907 | -4.1574226088414434e-06 | VERIFIED_PHYSICAL_CONTACT |
| convexHull | follower_right | right | 0.010875079780817032 | 0.011108541479667534 | 2.4996720640046988e-06 | VERIFIED_PHYSICAL_CONTACT |
| convexDecomposition | follower_left | left | 0.010882825590670109 | 0.011202171850452775 | -4.150578661210602e-06 | VERIFIED_PHYSICAL_CONTACT |
| convexDecomposition | follower_left | right | 0.010698344558477402 | 0.01130513234064566 | 3.091995722570573e-06 | VERIFIED_PHYSICAL_CONTACT |
| convexDecomposition | follower_right | left | 0.010882825590670109 | 0.011202171850452775 | -4.150578661210602e-06 | VERIFIED_PHYSICAL_CONTACT |
| convexDecomposition | follower_right | right | 0.010698344558477402 | 0.01130513234064566 | 3.091995722570573e-06 | VERIFIED_PHYSICAL_CONTACT |

The fixed-bottle contact-persistence signal is not a static-hold pass.
Task 8 remains `NOT_RUN`; the final collider is unchanged.
