# Phase 31/32: Visual reference repair attempts

## Question

Phase 30 proved that whole-stage sublayer composition restores collision and mass composition for the ALOHA1 imported USD.

The remaining issue is the Isaac runtime warning:

```text
Unresolved reference prim path ... </visuals/puppet_left_ee_arm_link>
Unresolved reference prim path ... </visuals/puppet_left_fingers_link>
Unresolved reference prim path ... </visuals/puppet_left_ee_gripper_link>
```

and the corresponding three right-arm paths.

This phase tested whether the warning can be repaired without re-exporting or rewriting the imported ALOHA1 base layer.

## Phase 31: parent-stage placeholder targets

Script:

```text
aloha_isaac_replay/scripts/validate_aloha1_clean_visual_targets.py
```

Attempt:

- create a runtime diagnostic stage;
- sublayer the left/right ALOHA1 wrappers;
- define the six missing `/visuals/...` targets as empty `Xform` prims in the parent diagnostic stage;
- verify collision composition and articulation initialization.

Result:

| Check | Result |
| --- | --- |
| Static missing local reference target count | 0 |
| Collision API count | 22 |
| Rigid body API count | 28 |
| Mass API count | 28 |
| Left articulation init | PASS |
| Right articulation init | PASS |
| Runtime unresolved reference warnings | still present, 6 lines in the evidence log |

Evidence:

```text
.codex/artifacts/20260718-014603_phase31-clean-visual-targets
```

Interpretation:

The parent-stage placeholder targets can make a composed-stage static scan look clean, but this does not make the Isaac runtime USD log clean. The local reference authored inside the imported base layer is not reliably repaired by adding target prims only at the parent runtime stage.

## Phase 32: side-wrapper reference override

Script:

```text
aloha_isaac_replay/scripts/validate_aloha1_clean_side_wrappers.py
```

Attempt:

- generate temporary left/right clean side wrappers under `reports/.../generated_wrappers`;
- sublayer the original ALOHA1 physics layers into those wrappers;
- try to override the six broken visual source prims with an explicit empty reference list;
- build a runtime diagnostic stage from those clean side wrappers;
- verify collision composition and articulation initialization.

Result:

| Check | Result |
| --- | --- |
| Static missing local reference target count | 6 |
| Collision API count | 22 |
| Rigid body API count | 28 |
| Mass API count | 28 |
| Left articulation init | PASS |
| Right articulation init | PASS |
| Runtime unresolved reference warnings | still present |

Evidence:

```text
.codex/artifacts/20260718-014909_phase32-clean-side-wrappers-v3
```

Interpretation:

The stronger side-wrapper override is still insufficient. The broken reference arcs remain effectively authored by the imported base layer. A thin overlay wrapper is not enough to create a runtime-clean asset.

## Engineering conclusion

The good news:

- ALOHA1 collision, mass, rigid body, and articulation data are recoverable.
- Both arms initialize as Isaac runtime articulations.
- The unresolved visual warnings are localized to six visual reference arcs.

The important blocker:

- The clean solution is not another parent-stage overlay.
- The clean solution should rebuild or rewrite the imported ALOHA1 composition layer so those six broken visual reference arcs are removed or replaced at the source layer level.

## Next step

Create a generated ALOHA1 clean asset package that is separate from the original imported files:

```text
assets/isaac/aloha1_clean_runtime/
```

The package should:

1. copy or regenerate the imported ALOHA1 USD composition into a controlled local package;
2. remove only the six broken visual reference arcs;
3. preserve all valid visual meshes, collision APIs, mass APIs, joints, drives, and articulation roots;
4. open in Isaac Sim 5.1 without unresolved reference warnings;
5. pass the Phase 30 collision/articulation runtime checks;
6. then replace the older zero-collider runtime entry point for future controller and contact tests.

