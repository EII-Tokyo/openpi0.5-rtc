# ALOHA1 Task 7 PhysicsRules root-cause matrix

- Status: `PARTIAL`
- Validator fresh Isaac processes: `20`
- Runtime fresh Isaac processes: `20`
- Frozen Stage SHA-256: `327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9`
- Final/default assets modified: `false`
- Task 8: `NOT_RUN`

| Candidate | Validator result | Decision |
|---|---|---|
| `joint_state_zero` | left `5`, right `5` blockers | `TARGETED_FIX_VERIFIED_RUNTIME_EQUIVALENT` |
| `baseline_gripper_fixed_group_split` | left `9`, right `9` blockers | `TARGETED_FIX_VERIFIED_RUNTIME_STABLE_GRASP_REGRESSION_REQUIRED` |
| `virtual_helpers_without_rigid_body` | left `64`, right `64` blockers | `REJECTED_REPEATABLE_REGRESSION` |
| `virtual_helper_topology_collapse` | left `6`, right `6` blockers | `TARGETED_TOPOLOGY_FIX_VERIFIED_PHYSICS_EQUIVALENCE_BLOCKED` |
| `combined_topology_joint_state` | left `1`, right `1` blockers | `VALIDATOR_REDUCED_TO_KNOWN_MIMIC_CONFLICT_PHYSICS_EQUIVALENCE_BLOCKED` |

The helper-body removal candidate is rejected: it removes the six original helper missing-collider findings but creates 57 deterministic `NonAdjacentCollisionMeshesDoNotClash` findings per follower. Two fresh processes per follower reproduce the same signature.

Raw and annotated failure evidence was visually reviewed after one rejected capture/annotation attempt. Absolute paths and hashes are stored in the JSON report.

The joint-state-zero candidate is runtime-equivalent to the frozen baseline in two fresh processes per follower. The fixed-group split is deterministic and stable but changes active collider paths, so accepted-grasp regression remains required before promotion.

The frame-preserving topology collapse removes the six helper findings without creating the 57 clash errors. The combined candidate leaves only the known Asset Validation 1.1.0 mimic-formula conflict. However, collapse also removes 0.00300000014 kg of source-authored, physically uncalibrated helper mass per follower. It remains diagnostic and non-promotable until mass, COM and inertia semantics are preserved or explicitly authorized for removal.
