# ALOHA follower finger collision runtime diagnosis

- Status: `PASS`
- Classification: `FINGER_COLLISION_PIPELINE_VERIFIED`
- Stage: `/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/table_support_alignment/1.0/aloha1_table_support_aligned_workcell.usda`
- Stage SHA-256: `2b3f76365ed67532f478d995ae859a88b5639975ac07cb7ac8a53ac679e8205c`
- Left physical contacts: `8`
- Right physical contacts: `8`
- Maximum bottle displacement during replay: `0.004185624 m`

## Collider screenshot semantics

- Every required phase has a normal image and a collision-overlay image from the same paused physics frame.
- The Isaac 5.1 setting `/persistent/physics/visualizationDisplayColliders` is read back and set to `2` for overlay captures.
- Green render evidence is the exact authored CollisionAPI mesh synchronized to PhysX body poses.
- Green geometry is not a cooked PhysX convex-hull readback.
- Blue is `left_finger`; orange is `right_finger`.

## Boundary

This report validates the finger/Bottle500 collision pipeline only. It does not promote the collider,
does not replace the final asset, and does not by itself prove a five-position grasp acceptance run.
Task 8 remains `NOT_RUN`.
