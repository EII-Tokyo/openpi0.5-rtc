# Phase 52 Bottle500 USD Contact Gate

## Question

Phase 51 proved that a hand-authored bottle proxy can pass a clean local left-gripper contact gate. Phase 52 asks the next question:

Can the actual `Bottle500` Isaac USD asset, with its authored collision meshes, replace the simplified proxy in the same local ALOHA1 gripper contact gate?

This matters because a proxy passing contact is useful but not enough. The simulation path must also prove that the asset intended for the bottle task has usable PhysX collision geometry.

## Isaac Semantics Checked

Before changing the validator, the NVIDIA Isaac Sim MCP documentation was consulted for:

- USD and Omniverse composition;
- PhysX rigid body and collider semantics;
- imported asset collision behavior.

The relevant Isaac/PhysX rule is:

- visual mesh and collision mesh are separate;
- collision behavior comes from `UsdPhysics.CollisionAPI`, not from what is visible;
- multiple child colliders under one rigid body can act as one compound rigid body.

This matches the current Bottle500 asset structure:

```text
assets/bottle_500ml/isaac/bottle_500ml_sim.usd
```

The asset contains visible bottle geometry plus many hidden collision mesh pieces under the bottle root.

## Code Change

Updated:

```text
aloha_isaac_replay/scripts/validate_aloha1_gripper_passive_contact.py
```

New support:

```text
--object-shape bottle_usd
--object-usd assets/bottle_500ml/isaac/bottle_500ml_sim.usd
--object-usd-prim-path /Bottle500
```

The validator now:

1. references the Bottle500 USD into the local contact test stage;
2. rotates Bottle500 local `+Z` long axis onto the requested world axis;
3. computes the composed Bottle500 bbox;
4. shifts the root so the actual composed bbox center, not just the semantic origin, is centered between the fingertip proxies;
5. applies contact/rest offsets to all child collision prims under the referenced bottle object;
6. runs the same contact-pair gate used for the previous proxy tests.

## Test Setup

Stage:

```text
local_eval_assets/aloha1_clean_runtime_20260718/aloha1_left_fingertip_pad_proxy_offset_runtime.usda
```

Why left-only:

The dual-arm runtime stage can still produce non-target contact pairs from the opposite side. Phase 51 showed that this can pollute local gripper contact gates. The left-only stage isolates the local left gripper.

Object placement:

- finger gap axis: `y`;
- open finger surface gap: about `0.0712 m`;
- Bottle500 diameter: about `0.068 m`;
- bottle long axis placed along world `X`, so the gripper contacts the bottle radial diameter through `Y`.

This is the intended geometry for a side grasp of a horizontal bottle.

## Command Artifact

```text
.codex/artifacts/20260718-141418_phase52-left-only-bottle-usd-contact
```

Structured report:

```text
reports/aloha1_isaac_adaptation/phase52_left_only_bottle_usd_contact_20260718/gripper_passive_contact_metrics.json
```

## Result

| Check | Result |
| --- | --- |
| status | `PASS` |
| contact trace status | `PASS_BILATERAL_CONTACT_CANDIDATE` |
| object shape | `bottle_usd` |
| object USD | `assets/bottle_500ml/isaac/bottle_500ml_sim.usd` |
| both expected fingers contacted object | true |
| both expected fingers had `CONTACT_FOUND` events | true |
| wrong contact pairs | 0 |
| object settle displacement | `5.9169e-08 m` |
| object close displacement | `0.001446 m` |
| max object displacement | `0.001446 m` |
| no explosion | true |
| contact motion policy | `not_required_for_bilateral_closure` |

The object collision offset pass touched the Bottle500 child collision meshes, for example:

```text
/World/phase43_passive_contact_cube/Collisions/COL_Body_00/COL_Body_00Mesh
/World/phase43_passive_contact_cube/Collisions/COL_Bottom/COL_BottomMesh
/World/phase43_passive_contact_cube/Collisions/COL_Neck_02/COL_Neck_02Mesh
```

This proves the test is not merely contacting a placeholder cube.

## Interpretation

This is now stronger than the Phase 51 proxy gate.

It proves:

1. the Bottle500 USD can be composed into the Isaac test stage;
2. its child collision meshes are visible to the collision-offset and contact-tracing code;
3. the left fingertip proxies can contact the Bottle500 collision asset from both sides;
4. there are no non-target contact pairs in the isolated left-only stage;
5. the bottle remains numerically stable during the bilateral closure.

It does not yet prove:

1. full-arm approach is correct;
2. inverse kinematics is correct;
3. the gripper can lift the bottle;
4. the bottle-pipe insertion task is physically calibrated;
5. dual-arm base transforms are correct;
6. real HDF5 replay can reproduce a successful insertion in simulation.

## Decision

Treat this as the current clean Bottle500 local gripper contact gate:

```text
reports/aloha1_isaac_adaptation/phase52_left_only_bottle_usd_contact_20260718/gripper_passive_contact_metrics.json
```

The previous `bottle_proxy` gate remains useful as a minimal debug object, but the Bottle500 USD gate should be the preferred gate before moving into real task replay.

## Next Gate

The next gate should replay a short real left-gripper close segment from HDF5 while the Bottle500 object is present.

Minimum requirements:

1. use a real recorded qpos segment, not a hand-authored finger-only target;
2. keep the test left-only until dual-arm base transforms are validated;
3. keep Bottle500 as `bottle_usd`;
4. require both expected fingertip contacts;
5. require wrong contact pairs to remain zero;
6. require finite bounded object motion.

Only after that should the work return to full dual-arm scene contact.
