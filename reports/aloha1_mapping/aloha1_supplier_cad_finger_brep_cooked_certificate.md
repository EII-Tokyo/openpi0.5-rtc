# ALOHA1 supplier-CAD B-Rep / cooked finger certificate

- Status: **PASS_DETERMINISTIC_MEASUREMENT_FAIL_EXACT_SURFACE_GATE**
- Exact surface: **ALL_PROFILES_CROSS_INWARD_CAD_SURFACE**
- Decomposition: **DECOMPOSITION_MIXED_OR_WORSE**
- Asset decision: **REJECTED_EXACT_CAD_CONTACT_GATE**
- Runtime grasp/hold claim: `NOT_MADE`
- Final/default collider modified: `false`

| Side | Approximation | Pieces | Exact B-Rep samples | Cooked coverage | Max inward crossing (mm) | Max undercoverage (mm) | Exact gate |
|---|---|---:|---:|---:|---:|---:|---|
| left | `convexHull` | 1 | 2029 | 0.896008 | 0.681205 | 0.000024 | FAIL_CROSSES_INWARD_CAD_SURFACE |
| left | `convexDecomposition` | 32 | 2029 | 0.618531 | 0.548108 | 8.964634 | FAIL_CROSSES_INWARD_CAD_SURFACE |
| right | `convexHull` | 1 | 2029 | 0.929029 | 0.681205 | 0.000008 | FAIL_CROSSES_INWARD_CAD_SURFACE |
| right | `convexDecomposition` | 32 | 2029 | 0.898472 | 1.349716 | 9.247752 | FAIL_CROSSES_INWARD_CAD_SURFACE |

The contact points are evaluated directly on the audited, trimmed OCCT B-Rep faces in two fresh FreeCAD 1.1.1 / OCCT 7.8.1 processes. No OBJ tessellation supplies these points. The exact crossing gate uses only a derived numerical floor: the maximum of the OCCT membership tolerance and eight float32 ULPs at the largest sample coordinate.

A failed exact-surface gate proves that the approximation is not an exact contact surface. It does not by itself define how much error is acceptable for the bottle task; that task-local approximation tolerance remains a HARD_BLOCKER and was not fitted from successful grasp videos.
