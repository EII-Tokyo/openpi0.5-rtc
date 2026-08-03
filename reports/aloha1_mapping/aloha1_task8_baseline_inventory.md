# ALOHA1 Task 8 baseline inventory

Status: `PASS`

The user-authorized boundary is `Task 7 = PARTIAL_ACCEPTED_FOR_TASK8`; this inventory does not promote or modify a final/default asset.

## Frozen inputs

- Stage: `/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_z_up_meters_diagnostic.usda`
- Stage SHA-256: `327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9`
- finger-limit layer: `/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/finger_limit_pair_collision_candidate/1.0/configuration/finger_source_limits.usda`
- finger-limit SHA-256: `2547e6fb374c213b5c6c54f200c7ced37605ab0e1a11735d0a32c0a231fd260f`

## Inventory

- composed prims (including instance proxies): 647
- meshes: 129 (56 visual, 73 collision)
- points / faces: 1441184 / 522537
- materials: 26
- instanceable prims: 84
- payload prims: 3
- repeated visual geometry groups: 19
- repeated collision geometry groups: 29

## Ranked opportunities

- `deduplicate_repeated_visual_geometry`: `ISOLATED_CANDIDATE`; risk `MEDIUM_HYDRA_REGRESSION_KNOWN`
- `deduplicate_materials`: `ISOLATED_CANDIDATE_AFTER_BINDING_AUDIT`; risk `LOW_PHYSICS_MEDIUM_VISUAL`
- `deduplicate_collision_geometry`: `DEFER_UNTIL_VISUAL_CANDIDATE_EVALUATED`; risk `HIGH_PHYSICS_REGRESSION`
- `add_payloads`: `NO_ACTION_ALREADY_PRESENT`; risk `LOW_PHYSICS_COMPOSITION_ONLY`

The first candidate is limited to repeated visual geometry. Collision deduplication remains deferred because it changes physics composition. Existing payload/instanceable authoring and the local Hydra protoPath failure are explicit constraints.
