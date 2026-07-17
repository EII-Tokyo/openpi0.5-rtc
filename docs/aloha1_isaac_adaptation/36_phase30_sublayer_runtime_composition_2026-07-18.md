# Phase 30: Sublayer runtime composition

## Question

Phase 27 showed that the current runtime reference path composes zero collision prims under `/World`.

Phase 28 then showed a more subtle fact: the ALOHA1 source layers are not empty or collision-free. The native source stages contain rigid bodies, mass APIs, and collision APIs. The likely problem is composition scope: the imported USD puts `/visuals` and `/colliders` at the stage root, while the wrapper `defaultPrim` points only to the robot subtree.

This phase asks a narrower diagnostic question:

If the ALOHA1 left/right wrappers are loaded as whole-stage sublayers instead of defaultPrim references, do the root-level collider scopes enter the runtime stage while the robot articulations still initialize?

## Method

New diagnostic script:

```text
aloha_isaac_replay/scripts/validate_aloha1_sublayer_runtime_composition.py
```

It creates a temporary diagnostic stage:

```text
reports/aloha1_isaac_adaptation/phase30_sublayer_runtime_composition_20260718/aloha1_dual_sublayer_diagnostic.usda
```

The stage:

- sets stage units to centimeters, matching the native imported ALOHA1 layers;
- sets up-axis to Y, matching the native imported ALOHA1 layers;
- sublayers both ALOHA1 wrappers as whole stages;
- adds a minimal `/World/physicsScene`;
- counts physics collision, rigid body, and mass APIs;
- opens the stage through Isaac Sim 5.1 headless runtime;
- initializes `SingleArticulation` for both arms at:
  - `/puppet_left_vx300s/root_joint`
  - `/puppet_right_vx300s/root_joint`

This is a diagnostic gate. It is not yet the final scene format.

## Result

Validation artifact:

```text
.codex/artifacts/20260718-014152_phase30-sublayer-runtime-composition-v2
```

Generated report:

```text
reports/aloha1_isaac_adaptation/phase30_sublayer_runtime_composition_20260718/sublayer_runtime_composition.json
reports/aloha1_isaac_adaptation/phase30_sublayer_runtime_composition_20260718/sublayer_runtime_composition.md
```

Summary:

| Check | Result |
| --- | --- |
| Whole-stage collision API count | 22 |
| Whole-stage rigid body API count | 28 |
| Whole-stage mass API count | 28 |
| Articulation roots | `/puppet_right_vx300s/root_joint`, `/puppet_left_vx300s/root_joint` |
| Left `SingleArticulation` init | PASS, 9 DOF, 14 bodies |
| Right `SingleArticulation` init | PASS, 9 DOF, 14 bodies |

Both runtime articulations reported the expected DOF names:

```text
waist
shoulder
elbow
forearm_roll
wrist_angle
wrist_rotate
gripper
left_finger
right_finger
```

## Interpretation

This strongly supports the Phase 29 hypothesis.

The ALOHA1 wrapper source does contain useful physics composition, but the current defaultPrim-reference runtime path loses root-level sibling scopes such as `/colliders`. Loading the left/right wrappers as whole-stage sublayers preserves those scopes, and Isaac Sim can still initialize both ALOHA1 articulations.

That means the next implementation step should not tune controllers against the old zero-collider runtime scene. First fix the runtime composition strategy so that visual and collision layers are present together.

## Remaining issue

The sublayer diagnostic still produced unresolved `/visuals/...` reference warnings. Therefore the result is not yet a clean final asset:

- collision and articulation initialization are validated;
- visual reference composition still needs a proper asset-layout fix;
- final scene generation should avoid relying on accidental root-level namespace composition.

## Next step

Build a runtime-ready ALOHA1 composition layer that:

1. preserves the full imported robot, visual, and collision namespace;
2. places left and right arms under stable scene paths;
3. keeps articulation root paths stable for controllers;
4. resolves `/visuals` and `/colliders` references without warnings;
5. passes the same collision/articulation runtime checks as this Phase 30 diagnostic.

