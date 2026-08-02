# ALOHA1 gripper geometry contract

- Status: **PARTIAL**
- Linkage formula: **PASS** over `1001` samples
- URDF carriage-center interval: `[0.042, 0.114] m`
- Trossen exact-product claim: `[0.042, 0.116] m`
- Aperture definition: **HARD_BLOCKER_DEFINITION_CONFLICT**

The pinned driver linkage is monotonic and yields exactly opposed left/right finger coordinates. The 114 mm URDF carriage-center endpoint is not changed to 116 mm to match the product-page claim. The definitions must be reconciled against the supplier CAD inner surfaces before formal candidate authoring.
