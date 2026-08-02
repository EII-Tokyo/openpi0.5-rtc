# ALOHA1 supplier-CAD compound contact candidate

- Status: **PASS_OFFLINE_CONTACT_REGION_GEOMETRY**
- Contact-region gate: **PASS**
- Full-face scope: **PARTIAL_CONTACT_REGION_ONLY**
- Asset decision: **DIAGNOSTIC_ONLY_NOT_PROMOTED**
- Final/default collider modified: `false`

| side | width (mm) | height (mm) | depth (mm) | pieces | contact coverage | max inward crossing (mm) | full-face max undercoverage (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| left | 10.023935 | 14.996426 | 3.004017 | 34 | 1.000000 | 0.000000000 | 8.964634 |
| right | 10.023940 | 14.996437 | 3.004019 | 34 | 1.000000 | 0.000000000 | 9.247752 |

The body pieces are the deterministic default Isaac decomposition clipped by the audited CAD contact plane. The added contact primitive is the maximum centered parameter rectangle obtained from exact OCCT face containment, extruded to the maximum uniform depth whose Boolean outside volume stays within the derived OCCT tolerance. No dimension was fitted from grasp success.

This offline result certifies only the central CAD-derived contact rectangle. It does not certify the complete finger face, runtime cooking, contact stability, or final asset promotion.
