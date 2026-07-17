# Phase 39: Link visual proxy candidate audit

## Question

Phase 38 created a stable collision-disabled controller runtime stage.

The next contact-simulation blocker is collision repair. Before generating new collision geometry, this phase asks:

Can the current clean ALOHA1 stage provide link-owned visual meshes or bounding boxes that are safe sources for simplified collision proxies?

## Code change

Added:

```text
aloha_isaac_replay/scripts/inspect_aloha1_link_visual_proxy_candidates.py
```

The script opens a clean runtime stage read-only and records each `PhysicsRigidBodyAPI` prim:

- side and robot root;
- whether the prim has a valid composed world bounding box;
- mesh descendants under the rigid body;
- collision descendants under the rigid body;
- whether it is a bbox-only proxy candidate;
- whether it is a higher-confidence mesh-owned proxy candidate.

## Validation

Command:

```bash
codex-evidence --name phase39-link-visual-proxy-candidates-v3 -- \
  .venv_issac/bin/python \
  aloha_isaac_replay/scripts/inspect_aloha1_link_visual_proxy_candidates.py \
  --stage-usd local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_clean_runtime.usda \
  --output-dir reports/aloha1_isaac_adaptation/phase39_link_visual_proxy_candidates_20260718
```

Evidence:

```text
.codex/artifacts/20260718-021944_phase39-link-visual-proxy-candidates-v3
```

Report:

```text
reports/aloha1_isaac_adaptation/phase39_link_visual_proxy_candidates_20260718/link_visual_proxy_candidates.json
reports/aloha1_isaac_adaptation/phase39_link_visual_proxy_candidates_20260718/link_visual_proxy_candidates.md
```

## Result

| Check | Result |
| --- | ---: |
| Rigid body rows | 28 |
| Rows with valid bbox | 22 |
| Rows with mesh descendants | 0 |
| Rows with link-owned collision descendants | 0 |
| Bbox-only proxy candidates | 22 |
| Mesh-owned proxy candidates | 0 |

The 22 valid bbox rows include the major links on both sides:

```text
base_link
shoulder_link
upper_arm_link
upper_forearm_link
lower_forearm_link
wrist_link
gripper_link
gripper_prop_link
gripper_bar_link
left_finger_link
right_finger_link
```

The known invalid rows are:

```text
ee_arm_link
fingers_link
ee_gripper_link
```

on both left and right.

## Important correction

The Isaac importer applies `ArticulationRootAPI` on:

```text
/puppet_left_vx300s/root_joint
/puppet_right_vx300s/root_joint
```

The rigid bodies are not children of those joint prims. They are siblings under:

```text
/puppet_left_vx300s
/puppet_right_vx300s
```

So link ownership checks must use the side robot root paths, not the ArticulationRootAPI paths as tree ancestors.

## Interpretation

This stage is not ready for mesh-owned collision repair by simple traversal, because there are no direct Mesh descendants under the rigid-body prims.

However, the stage does expose valid composed bounding boxes for 22 rigid bodies. This supports a conservative next step:

- create small link-owned box proxies from each valid rigid-body bounding box;
- keep those proxies under the actual link hierarchy;
- enable them incrementally;
- run the Phase 38 single-joint response gate after each group.

This is lower confidence than mesh-owned colliders, but it is safer than reusing the current root-level `/colliders` layer.

## Decision

Proceed with bbox-only collision proxies as an experimental collision repair layer.

Do not claim contact or grasp realism from this alone. The first acceptance gate is still free-space controller stability with the proxies enabled.

## Next step

Generate a separate collision-proxy stage/layer that:

1. disables the root-level `/colliders`;
2. adds simplified box collision proxies under the 22 valid rigid-body links;
3. excludes the six invalid bbox links;
4. validates left `waist` and right `shoulder` with the same Phase 38 controller gate;
5. only then expands toward gripper, table, pipe, and bottle contact.
