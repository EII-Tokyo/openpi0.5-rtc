# ALOHA1 Bottle500 velocity consistency diagnosis

- Status: `PARTIAL`
- Velocity semantics: `INCONCLUSIVE`
- Isaac Sim / Kit / PhysX: `5.1.0.0 / 107.3.3 / 107.3.26`
- Task 8: `NOT_RUN`

| Variant | Runtime | Alignment | Signature |
|---|---|---|---|
| BASELINE | PASS | MISMATCH | `90da200849182aa4cadbd37c64a6b36983758fe0e23a3741acb09f1e6740a88d` |
| INITIALIZE_KINEMATIC_BODIES | FAIL | NOT_OBSERVABLE_NO_HOLD_SAMPLES | `e90a524f29f4e2be93c4d257281857d671b026947c5dee4cdf4b775994c1a984` |
| RECREATE_AFTER_DYNAMIC | PASS | MISMATCH | `90da200849182aa4cadbd37c64a6b36983758fe0e23a3741acb09f1e6740a88d` |

The baseline and dynamic-view recreation have identical grasp signatures and identical velocity mismatch. Calling `initialize_kinematic_bodies()` at the tested post-reset point makes the first tensor sample invalid and fails the run. COM and rigid-prim-origin comparisons were both evaluated; point choice alone does not explain the mismatch.

Tensor velocity is therefore retained as an explicitly unresolved diagnostic channel. Contact pairs, pose, support clearance, drop, and deterministic video evidence remain the authoritative hold gate; this report does not silently reinterpret tensor velocity.
