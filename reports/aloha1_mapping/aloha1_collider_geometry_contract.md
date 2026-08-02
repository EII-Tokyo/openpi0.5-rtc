# ALOHA1 collider geometry contract

- Status: **PARTIAL**
- Deterministic tessellation: **PASS**
- Existing swept collision gate: **PASS**
- Unresolved CAD/link records: `0`
- Unresolved suffixes: `[]`
- Formal candidate gate: **BLOCKED**
- Exact supplier-finger B-Rep gate: **ALL_PROFILES_CROSS_INWARD_CAD_SURFACE**
- Default decomposition comparison: **DECOMPOSITION_MIXED_OR_WORSE**

The supplier STEP remains authoritative for the geometry it exposes, and its fused gripper/invalid wrist boundaries are preserved. Byte-identical pinned Interbotix meshes supply the link-level identities. Every physical link now has a numerical convex-hull surface/volume certificate. Two fresh FreeCAD processes sampled the exact trimmed finger-pad B-Rep faces, and two fresh Isaac 5.1 processes cooked both single hull and default decomposition. All four profiles cross the inward CAD face beyond the derived numerical floor; decomposition is mixed/worse across handed sides. Promotion remains blocked because the task-local acceptable approximation error is not defined; successful grasp videos were not used to fit a tolerance. Existing static/swept tests remain rejection evidence. No collider is accepted because a grasp happened to pass, and no final/default asset was changed.
