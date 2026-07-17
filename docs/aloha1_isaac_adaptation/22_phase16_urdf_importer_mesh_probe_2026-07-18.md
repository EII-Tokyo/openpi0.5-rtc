# Phase 16 URDF Importer Mesh Probe

## Question

Phase 15 showed that the composed ALOHA1 generated USD variants report zero Mesh prims and zero CollisionAPI prims. The next question is:

Is this caused by invalid URDF mesh paths, a missing Isaac importer option, or a broken USD composition/reference layer?

## Method

I added:

```bash
python3 aloha_isaac_replay/scripts/probe_aloha1_urdf_importer_mesh_options.py
```

The probe starts Isaac Sim 5.1 headlessly and imports:

```text
assets/isaac/original_stationary_aloha/generated/puppet_left_vx300s_resolved.urdf
```

It compares two importer configs:

1. The current project import config.
2. A candidate config that sets `collision_from_visuals=True` and probes several guessed visual/collision flags.

It then inspects both the composed wrapper USD and the generated `configuration/*` layer files.

## Evidence

Generated report:

- JSON: `reports/aloha1_isaac_adaptation/phase16_urdf_importer_probe_20260718/urdf_importer_mesh_probe.json`
- Markdown: `reports/aloha1_isaac_adaptation/phase16_urdf_importer_probe_20260718/urdf_importer_mesh_probe.md`

Verification artifact:

- `.codex/artifacts/20260718-002445_phase16-urdf-importer-mesh-probe-v2`

## Result

### Direct composed stage

| Import config | Mesh prims | Collision prims | Rigid bodies | Joints |
| --- | ---: | ---: | ---: | ---: |
| current config | 0 | 0 | 14 | 14 |
| candidate mesh config | 0 | 0 | 14 | 14 |

### Generated layer files

| Layer | Mesh prims | Collision prims | Rigid bodies | Joints |
| --- | ---: | ---: | ---: | ---: |
| composed wrapper USD | 0 | 0 | 14 | 14 |
| `*_base.usd` | 32 | 0 | 0 | 0 |
| `*_physics.usd` | 32 | 11 | 14 | 14 |
| `*_robot.usd` | 0 | 0 | 0 | 0 |
| `*_sensor.usd` | 0 | 0 | 0 | 0 |

## Importer Config Finding

Isaac Sim 5.1 `URDFCreateImportConfig` exposes:

```text
collision_from_visuals
```

It does not expose the guessed fields:

```text
import_visuals
import_collision
parse_visuals
parse_collision
create_visuals
create_collisions
```

Setting `collision_from_visuals=True` did not make the composed wrapper stage see Mesh or CollisionAPI prims.

## Interpretation

This refines Phase 15.

The URDF mesh paths are not the primary problem. The importer does generate mesh-containing layer files:

- the base layer has visual Mesh prims;
- the physics layer has Mesh prims, CollisionAPI prims, RigidBodyAPI prims, and joints.

The blocker is composition/reference integrity: the top-level composed USD opens as an articulation skeleton with rigid bodies and joints, but Mesh and CollisionAPI prims do not compose into the visible/control stage.

The Isaac log repeatedly reports unresolved references such as:

```text
base layer visual scope -> physics layer /visuals/... prim path not found
```

So the next repair target is the USD layer structure produced by the URDF importer, not the original STL files.

## Decision

The ALOHA1-native route remains correct, but the implementation path changes:

1. Do not discard the resolved ALOHA1 URDF and mesh package.
2. Do not keep using the current top-level generated USD wrapper as the controller asset.
3. Repair or bypass the broken importer composition layer.
4. Preserve ALOHA1 URDF joint semantics as source of truth.
5. Preserve generated base/physics layer data as evidence that meshes and collision can exist.

## Next Phase

Phase 17 should test the smallest repair:

1. Open generated base and physics layers directly.
2. Identify the exact broken reference targets.
3. Create a repaired wrapper or flattened diagnostic USD in a scratch output directory.
4. Validate that the repaired composed stage has:
   - nonzero Mesh prims;
   - nonzero CollisionAPI prims;
   - nonzero RigidBodyAPI prims;
   - correct joint count;
   - valid articulation root.
5. Only after this passes, regenerate both left and right ALOHA1 assets.
