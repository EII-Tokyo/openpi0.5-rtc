# ALOHA1 Bottle500 swept contact-band collider certificate

- Audit status: **PASS_DETERMINISTIC_REJECTION**
- Task contact-band status: **FAIL_CENTRAL_TANGENCY_OUTSIDE_COMPOUND_PATCH**
- Candidate decision: **REJECTED_TASK_CONTACT_BAND_NOT_PROMOTED**
- Final/default collider modified: `false`
- Grasp success used to set tolerance: `false`

| side | signed plane residual (mm) | finite-patch miss (mm) | cooked normal error (deg) | outward crossing (mm) | finite patch |
|---|---:|---:|---:|---:|---|
| left | 0.000000000 | 1.613742 | 0.000001708 | 0.000000291 | FAIL |
| right | 0.000000000 | 1.613740 | 0.000003078 | 0.000000637 | FAIL |

The analytic Bottle500 tangent point is on the authoritative infinite supplier-CAD inner plane on both sides. It is nevertheless outside the finite 10.02 mm compound contact rectangle by about 1.61 mm. Plane alignment alone was therefore an insufficient acceptance test.

The known conservative numerical sum is `0.200480 mm`; the smallest finite-patch miss is `1.613740 mm` (`8.049x` larger). The local 107.3 runtime does not expose the effective contactOffset as USD readback, so the complete contact-envelope budget remains PARTIAL. That missing readback cannot promote a geometry patch which does not contain the task's central analytic tangency.

This is a deterministic rejection certificate, not a collider repair. The 68-piece candidate remains diagnostic-only and unpromoted.
