# ALOHA1 gripper geometry contract

- Status: **PASS**
- Linkage formula: **PASS** over `1001` samples
- URDF carriage-center interval: `[0.042, 0.114] m`
- Trossen exact-product claim: `[0.042, 0.116] m`
- Aperture definition: **PASS_WITH_DOCUMENTED_OFFICIAL_SOURCE_CONFLICT**

The pinned driver linkage is monotonic and yields exactly opposed left/right finger coordinates. The 114 mm URDF carriage-center endpoint is not changed to 116 mm to match the product-page claim. CAD carriage datums agree with 114 mm, while the tilted distal contact-surface gap is position-dependent. The 2 mm product-page conflict remains explicit and no fitted endpoint is used.
