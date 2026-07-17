# Phase 18 Runtime Articulation Validation

## Question

Phase 17 proved that a direct physics-layer wrapper composes Mesh, CollisionAPI, RigidBodyAPI, joints, and articulation roots.

The next question is:

Can Isaac Sim runtime initialize the repaired ALOHA1 wrappers as `SingleArticulation` objects and expose usable DOF names?

## Method

I added:

```bash
python3 aloha_isaac_replay/scripts/validate_aloha1_physics_wrapper_runtime.py
```

The script:

1. starts Isaac Sim headlessly;
2. references the Phase 17 left/right wrappers into `/World/left` and `/World/right`;
3. dynamically finds articulation roots instead of relying on hard-coded paths;
4. initializes `SingleArticulation` for each arm;
5. writes PASS/FAIL JSON before and after each major step.

## Evidence

Generated report:

- JSON: `reports/aloha1_isaac_adaptation/phase18_runtime_articulation_20260718/physics_wrapper_runtime_articulation.json`
- Markdown: `reports/aloha1_isaac_adaptation/phase18_runtime_articulation_20260718/physics_wrapper_runtime_articulation.md`

Verification artifact:

- `.codex/artifacts/20260718-003216_phase18-runtime-articulation-validation-v3`

## Result

| Side | Runtime articulation root | DOFs | Bodies | Gate |
| --- | --- | ---: | ---: | --- |
| left | `/World/left/root_joint` | 9 | 14 | PASS |
| right | `/World/right/root_joint` | 9 | 14 | PASS |

Both sides expose the same DOF names:

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

This is the first strong positive result for the ALOHA1-native Isaac route:

- the repaired wrapper is not just statically visible;
- Isaac runtime can initialize it as an articulation;
- the exposed DOF names match the ALOHA1 semantic joint names used by the real robot code;
- each arm has 14 bodies and 9 DOFs.

This does not mean the model is fully finished. It still needs:

- unresolved visual reference cleanup;
- deterministic production asset placement outside `reports/`;
- drive, damping, force, and max velocity checks;
- qpos limit and sign validation against real ALOHA1 data;
- short replay smoke tests;
- workcell integration with table, pipe, bottle, and cameras.

## Decision

Proceed with ALOHA1-native production asset generation based on the physics-layer wrapper approach.

Stop spending effort on forcing ALOHA1 qpos into Trossen `stationary_ai` joint chains. Trossen remains the framework reference, but the ALOHA1 robot asset should now be built from ALOHA1 URDF/importer physics layers.

## Next Gates

Phase 19 should create a proper production asset directory and validate:

1. no unresolved reference warnings, or at least a bounded list of remaining warnings;
2. clean left/right wrappers under `assets/isaac/`;
3. runtime DOF order and limits;
4. initial qpos inside limits;
5. one short recorded qpos replay without controller remapping;
6. no robot explosion or physics instability on a short simulation step.
