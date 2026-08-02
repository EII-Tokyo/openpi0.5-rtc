# ALOHA1 official model-first closure

- Status: **PARTIAL_MODEL_PROOF**
- Compound central contact region: **PASS_RUNTIME_COOKED_CONTACT_REGION_GEOMETRY_NOT_PROMOTED**
- Coordinate frame: `FINGER_LINK_LOCAL_METRES`
- Full effective contact surface: **PARTIAL_CONTACT_REGION_ONLY**
- Geometry-only USD: **PASS_GEOMETRY_ONLY_DIAGNOSTIC_USD**
- Articulation integration: **NOT_RUN**
- Contact dynamics: **NOT_RUN**
- Official model candidate: **NOT_BUILT_BLOCKED**
- Task 8: **AUTHORIZED_PAUSED_AT_MODEL_PROOF_GATE**
- Final/default asset modified: `False`

## Verified boundary

The supplier-CAD central finger contact region is now proven in finger-link-local metres through two fresh Isaac 5.1 cooking processes, and the geometry-only USD is byte-identical across two fresh builds. This does not prove the complete effective finger contact surface, articulation integration, contact dynamics, calibrated drives, or material/solver mappings. The diagnostic candidate is therefore not promoted and the final/default asset remains unchanged.

This was a static collision-cooking and geometry-authoring check. No timeline was started, so a video would not provide additional physical evidence. The rejected exact-ray attempt and its annotated numerical-quantization screenshot remain part of the evidence trail.

## Remaining formal blockers

- `HARD_BLOCKER_CONTINUOUS_TORQUE_SPEED_CURRENT_THERMAL_CURVE`: measured continuous torque-speed-current thermal envelope beyond the official 12 V 20%-of-stall estimates
- `HARD_BLOCKER_PHYSX_DRIVE_PHYSICAL_DERIVATION`: physical mapping from exact actuator/controller/transmission to PhysX stiffness, damping and maxForce
- `HARD_BLOCKER_COLLIDER_ACCEPTANCE_ERROR_BUDGET`: official or task-derived numerical acceptance tolerance for the complete per-link convex-hull surface/volume certificate
- `HARD_BLOCKER_EXACT_CONTACT_MATERIAL_PROPERTIES`: exact static/dynamic friction, restitution and combine rules for finger-bottle-table material pairs
- `HARD_BLOCKER_NUMERICAL_ERROR_BUDGET_NOT_YET_DERIVED`: documented numerical error budget selecting timestep and solver iterations for this model

The supplier-derived diagnostic geometry remains local-only while redistribution rights are unknown.
