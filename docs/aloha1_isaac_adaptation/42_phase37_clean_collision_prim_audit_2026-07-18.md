# Phase 37: Clean collision prim audit

## Question

Phase 36 showed that the clean runtime stage passes single-joint dynamic response only after disabling all robot collision prims.

This phase asks:

What exactly are those collision prims, and are they attached to the ALOHA1 articulations as link collision geometry?

## Code change

Added:

```text
aloha_isaac_replay/scripts/inspect_aloha1_clean_collision_prims.py
```

The script opens a clean runtime stage read-only and records:

- stage units and up axis;
- articulation roots;
- every `PhysicsCollisionAPI` prim;
- whether each collision prim is under an articulation root;
- whether it has `PhysicsRigidBodyAPI`;
- whether it has a rigid-body ancestor;
- whether its world bounding box is valid.

## Validation

Command:

```bash
codex-evidence --name phase37-clean-collision-prim-audit-v2 -- \
  .venv_issac/bin/python \
  aloha_isaac_replay/scripts/inspect_aloha1_clean_collision_prims.py \
  --stage-usd local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_clean_runtime.usda \
  --output-dir reports/aloha1_isaac_adaptation/phase37_clean_collision_prim_audit_20260718
```

Evidence:

```text
.codex/artifacts/20260718-020927_phase37-clean-collision-prim-audit-v2
```

Report:

```text
reports/aloha1_isaac_adaptation/phase37_clean_collision_prim_audit_20260718/clean_collision_prim_audit.json
reports/aloha1_isaac_adaptation/phase37_clean_collision_prim_audit_20260718/clean_collision_prim_audit.md
```

## Result

| Check | Result |
| --- | ---: |
| Collision prim count | 22 |
| Under articulation root | 0 |
| Has `PhysicsRigidBodyAPI` | 0 |
| Has rigid-body ancestor | 0 |
| Valid world bounding box | 0 |
| Suspicious static collision count | 22 |

The collision paths are root-level objects such as:

```text
/colliders/puppet_left_base_link/base/node_STL_BINARY_
/colliders/puppet_left_shoulder_link/shoulder/node_STL_BINARY_
/colliders/puppet_left_upper_arm_link/upper_arm/node_STL_BINARY_
...
/colliders/puppet_right_gripper_link/gripper/node_STL_BINARY_
```

They are not under:

```text
/puppet_left_vx300s/root_joint
/puppet_right_vx300s/root_joint
```

and they have no rigid-body ancestor.

## Interpretation

This explains the Phase 36 result.

The current clean runtime stage contains robot-shaped collision meshes, but they are composed as root-level collision prims rather than link-owned collision geometry in the articulation tree.

From PhysX's point of view, those prims are suspicious static collision objects. When the articulation moves in free space, it can collide with these static robot-shaped meshes and destabilize the drive. That is why:

- collision enabled: single-joint dynamic gate fails;
- all collisions disabled: same gate passes cleanly.

The invalid bounding boxes are another warning. The audit cannot currently use these prims as trustworthy collision geometry for contact simulation.

## Decision

Do not use the current imported `/colliders` layer for bottle, table, or pipe contact.

The clean runtime stage is still useful for:

- articulation initialization;
- qpos set/readback;
- collision-disabled dynamic controller gates;
- future collision repair.

But contact simulation remains blocked until collision geometry is re-authored or correctly attached to the articulated links.

## Next step

Build a collision strategy in this order:

1. keep the clean stage collision-disabled for controller and replay validation;
2. create a separate collision repair layer;
3. attach simplified collision proxies to the actual link hierarchy, not root-level `/colliders`;
4. validate each group with the same single-joint dynamic gate;
5. only then add bottle/table/pipe contact.

