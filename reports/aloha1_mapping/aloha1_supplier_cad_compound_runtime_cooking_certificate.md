# ALOHA1 supplier-CAD compound runtime cooking certificate

- Status: **PASS_RUNTIME_COOKED_CONTACT_REGION_GEOMETRY_NOT_PROMOTED**
- Fresh-process determinism: **PASS_DETERMINISTIC_FRESH_PROCESS_COOKING**
- Asset decision: **DIAGNOSTIC_ONLY_NOT_PROMOTED**
- Final/default collider modified: `false`
- Timeline/video: `NOT_APPLICABLE_STATIC_COOKING_ONLY`

| side | source pieces | cooked pieces | exact-ray coverage | tolerance-adjusted coverage | max outward crossing (nm) | max quantization distance (nm) | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| left | 34 | 34 | 0.778546713 | 1.000000000 | 0.291119 | 2.995208 | PASS |
| right | 34 | 34 | 1.000000000 | 1.000000000 | 0.636510 | 0.000000 | PASS |

The exact-ray coverage ratio is intentionally retained. PhysX stores the cooked vertices at float32 precision, so source points displaced by nanometres can lie just outside an exact half-space test. The adjusted result accepts only points whose nearest cooked surface distance and normal projection are both below the previously derived `MAX(OCCT membership tolerance, 8 float32 ULP)` floor.

The first rejected report is preserved as `REJECTED_CERTIFICATE_EXACT_RAY_FALSE_NEGATIVE`; it did not prove a geometry failure. This certificate covers only the central, CAD-derived contact rectangle. Full-face coverage, articulation integration, contact dynamics and asset promotion remain outside this gate.
