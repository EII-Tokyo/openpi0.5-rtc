# ALOHA1 supplier-CAD finger cooked contact certificate

- Cooking status: **PASS_COOKING_DETERMINISTIC**
- Geometry classification: **DECOMPOSITION_MIXED_OR_WORSE**
- Asset decision: **DIAGNOSTIC_ONLY_NOT_PROMOTED**
- Runtime hold claim: `NOT_MADE`
- Final/default collider modified: `false`

| Side | Approximation | Pieces | Exact source coverage | Maximum contact deviation (m) | Geometry gate |
|---|---|---:|---:|---:|---|
| left | `convexHull` | 1 | 0.91129 | 0.000798538592 | FAIL_EXCEEDS_TESSELLATION_ERROR_BUDGET |
| left | `convexDecomposition` | 32 | 0.685484 | 0.000561188714 | FAIL_EXCEEDS_TESSELLATION_ERROR_BUDGET |
| right | `convexHull` | 1 | 0.951613 | 0.000798539557 | FAIL_EXCEEDS_TESSELLATION_ERROR_BUDGET |
| right | `convexDecomposition` | 32 | 0.951613 | 0.00132829852 | FAIL_EXCEEDS_TESSELLATION_ERROR_BUDGET |

The maximum deviation combines outward normal-ray envelope for source points covered by the cooked union with nearest cooked-surface distance for uncovered source points. It is compared only with the pre-existing 0.20 mm FreeCAD tessellation budget; no threshold was fitted from grasp success. This is a geometry certificate, not a grasp/hold promotion.
