# ALOHA1 collider geometry contract

- Status: **PARTIAL**
- Deterministic tessellation: **PASS**
- Existing swept collision gate: **PASS**
- Unresolved CAD/link records: `0`
- Unresolved suffixes: `[]`
- Formal candidate gate: **BLOCKED**
- Exact supplier-finger B-Rep gate: **ALL_PROFILES_CROSS_INWARD_CAD_SURFACE**
- Default decomposition comparison: **DECOMPOSITION_MIXED_OR_WORSE**
- Compound central contact region: **PASS_RUNTIME_COOKED_CONTACT_REGION_GEOMETRY_NOT_PROMOTED**
- Compound geometry-only USD: **PASS_GEOMETRY_ONLY_DIAGNOSTIC_USD**
- Bottle500 finite task contact band: **FAIL_CENTRAL_TANGENCY_OUTSIDE_COMPOUND_PATCH**

The supplier STEP remains authoritative for the geometry it exposes, and its fused gripper/invalid wrist boundaries are preserved. Byte-identical pinned Interbotix meshes supply the link-level identities. Every physical link now has a numerical convex-hull surface/volume certificate. Two fresh FreeCAD processes sampled the exact trimmed finger-pad B-Rep faces, and two fresh Isaac 5.1 processes cooked both single hull and default decomposition. All four profiles cross the inward CAD face beyond the derived numerical floor; decomposition is mixed/worse across handed sides. A CAD-derived compound candidate then removes inward-plane crossing within a finite central contact rectangle in two fresh finger-link-local PhysX cooking runs. Its 68-piece geometry-only USD is deterministic. A signed Bottle500 task certificate now proves that the analytic tangency is on the infinite CAD plane but about 1.61 mm outside that finite patch on both fingers, so the candidate is rejected rather than promoted. Articulation integration remains incomplete. Promotion remains blocked because a corrected effective task contact surface and remaining physical mappings are not fully proven; successful grasp videos were not used to fit a tolerance. Existing static/swept tests remain rejection evidence. No collider is accepted because a grasp happened to pass, and no final/default asset was changed.
