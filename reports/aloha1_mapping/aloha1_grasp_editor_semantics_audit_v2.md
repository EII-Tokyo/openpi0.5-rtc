# ALOHA 1 Grasp Editor semantics audit V2

- Status: `PASS`
- Classification: `GRASP_EDITOR_EXPORT_PASS_DIAGNOSTIC_COUPLING`
- Passing coupling path: `official_symmetric_adapter`
- Runtime residual: `2.390146255493164e-05 m`
- Bilateral physical contact: `True`
- Transform closure: `PASS`
- Screenshot review: `PASS`
- Diagnostic IK allowed: `True`
- Final asset promotion authorized: `False`
- Task 8: `NOT_RUN`

The raw and derived YAML expose only `left_finger`, matching the one physical gripper actuation coordinate. The right finger remains a source-backed runtime observer derived as `-q`.

The vertical bottle screenshots are robot-local Grasp Editor authoring evidence, not horizontal task-placement evidence.
