# ALOHA1 Task 7A official-rule triage

- Triage status: `PARTIAL`
- Literal NVIDIA status: `FAIL`
- Official findings suppressed: `false`
- Runtime: Isaac Sim `5.1.0.0`, Kit `107.3.3`, PhysX `107.3.26`
- Source findings: `37`
- Blocking findings: `22`
- Inconclusive findings: `0`

| Classification | Count |
|---|---:|
| ISAAC_5_1_VALIDATOR_SCHEMA_CONFLICT | 2 |
| LAYER_PACKAGING_DEFECT | 28 |
| MISSING_SOURCE_EVIDENCE | 6 |
| NON_APPLICABLE_FALSE_POSITIVE | 1 |

## Boundary

The gripper JointStateAPI exists twice in the frozen workcell home layer; the official finding is produced when the child robot asset is validated without that workcell layer.

The read-only mimic probe loaded the installed 5.1 MimicAPICheck and confirmed positive limits on the active finger, negative limits on the opposite local finger axis, and positive gearing. The installed rule compares those raw local-axis intervals and its positive-gearing error message labels the self upper limit as a lower limit. This is recorded as a version-specific validator/schema conflict; the two literal errors are not suppressed.

Mass-only helper links remain missing-source-evidence blockers. No collider, density, mass, or inertia was invented. The literal NVIDIA failures remain visible.
