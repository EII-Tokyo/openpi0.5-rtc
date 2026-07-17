# Collision Repair Research For ALOHA1 Isaac

## Question

Two failures needed broader investigation:

1. Why do root-level `/colliders` break dynamic control?
2. Why does a full-link bbox proxy collision layer become unstable?

## Confirmed Local Facts

- Disabling 22 root-level `/colliders` makes clean-stage single-joint dynamic control pass.
- Full-link bbox proxy collision stages are unstable across tested scales.
- Gripper-only bbox proxy stage passes arm single-joint smoke, gripper DOF smoke, and direct finger proxy gap validation.

## Source-Backed Mechanism

OpenUSD and NVIDIA physics documentation define collider ownership through the rigid body hierarchy. A collision geometry that is under a rigid body is treated as part of that body. A collision geometry without a rigid body ancestor is effectively a static collider.

Therefore, robot link colliders placed under a root-level `/colliders` tree are not guaranteed to follow the intended articulation link. In practice they can behave like invisible static obstacles or incorrectly owned collision shapes. The articulation drive then has to fight false contacts.

For full-link bbox proxies, the problem is different. The proxies are link-owned, but the boxes are too coarse. They fill empty space around joints and neighboring links. This creates false self-contact and excessive contact constraints. The drive system, joint constraints, and false contact constraints compete, so the dynamic response becomes scale-dependent and unstable.

## Engineering Consequence

The short-term stable path is:

1. Disable root-level imported `/colliders`.
2. Keep the clean articulation and drive layer.
3. Add only minimal, link-owned collision proxy geometry where contact is actually needed.
4. Start with gripper-only proxies before adding arm-body or full-task contact.

The long-term path is not full-link bbox. It is a proper per-link collision layer: manually authored primitives, convex decomposition, or carefully generated collision meshes, with collision filtering for adjacent links.

## Repair Routes

### Route A: Short-Term Smoke Runtime

Use the current controller runtime:

- root-level `/colliders` disabled;
- no full-arm collision proxies;
- optional gripper-only collision proxies.

Acceptance:

- single-joint dynamic response passes;
- gripper DOF or finger DOF response passes;
- no false collision is required for free-space replay.

Failure signals:

- joint response changes when unrelated collision prims are enabled;
- contact count rises during free-space motion;
- a static collider appears near moving robot links.

### Route B: Local Gripper Contact Runtime

Keep only collision near the gripper and task object:

- finger boxes/capsules or reduced convex proxies;
- passive bottle/box/cylinder with simple collision;
- tuned friction and contact offsets.

Acceptance:

- passive object contact does not explode;
- object motion is bounded;
- gripper can open and close around the object;
- no NaN or sustained jitter.

Failure signals:

- object is pushed by empty bbox space;
- fingers visually miss but collision still occurs;
- contact jitter persists after settling.

### Route C: Training-Grade Collision Asset

Create a dedicated simulation collision layer:

- per-link collision geometry under each rigid body;
- no root-level robot colliders;
- convex decomposition for complex link geometry;
- primitives for simple structural links;
- SDF or carefully simplified mesh only for contact-critical non-convex shapes;
- collision filtering for adjacent links.

Acceptance:

- long headless rollouts remain stable;
- random small perturbations do not cause self-collision explosions;
- contact debug visualization matches expected physical surfaces;
- replay/controller metrics remain stable with collision enabled.

Failure signals:

- stability only appears after unrealistically lowering drive gains;
- contact buffer warnings or solver divergence appear;
- full rollout behavior is non-repeatable.

## References

- NVIDIA Omni Physics rigid bodies:
  <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/rigid_bodies.html>
- NVIDIA Omni Physics colliders:
  <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/collision.html>
- NVIDIA Omni Physics articulations:
  <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/articulations.html>
- NVIDIA articulation stability guide:
  <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/articulation_stability_guide.html>
- NVIDIA gripper tuning example:
  <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/gripper_tuning_example.html>
- Isaac Sim URDF importer and collision options:
  <https://docs.isaacsim.omniverse.nvidia.com/6.0.0/importer_exporter/ext_isaacsim_asset_importer_urdf.html>
- OpenUSD `UsdPhysicsCollisionAPI`:
  <https://openusd.org/dev/api/class_usd_physics_collision_a_p_i.html>
- Trossen AI Isaac:
  <https://github.com/TrossenRobotics/trossen_ai_isaac>
- Trossen arm description:
  <https://docs.trossenrobotics.com/interbotix_xsarms_docs/ros2_packages/arm_descriptions.html>
- CoACD:
  <https://github.com/SarahWeiii/CoACD>
- V-HACD:
  <https://github.com/Unity-Technologies/VHACD>
- obj2mjcf:
  <https://github.com/kevinzakka/obj2mjcf>
- Factory contact-rich assembly paper:
  <https://arxiv.org/abs/2205.03532>
