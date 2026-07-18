# Phase99 Convex Legacy Collision Cleanup

## Result

Phase99 keeps the Phase97 drive-target replay gate passing after rebuilding the `/scene` proxy runtime stage with explicit `convexHull` approximation on disabled legacy finger collision prims.

This removes the previous PhysX dynamic-body warning:

```text
triangle mesh collision ... cannot be a part of a dynamic body, falling back to convexHull approximation
```

## Why This Was Needed

The proxy runtime stage already disabled the original imported finger mesh collision prims and used `bbox_collision_proxy` as the active contact target. However, PhysX still parsed those disabled dynamic-body mesh collision prims and produced fallback warnings.

The fix is to keep those legacy collision prims disabled, but also author:

```text
UsdPhysics.MeshCollisionAPI.approximation = convexHull
```

That makes the runtime layer explicit instead of relying on PhysX fallback behavior.

## Rebuild Command

```bash
codex-evidence --name aloha-phase98-rebuild-proxy-runtime-convex-legacy-collisions -- \
  .venv_issac/bin/python aloha_isaac_replay/scripts/build_aloha1_bbox_proxy_runtime_stage.py \
  --stage-usd local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose/aloha2_menagerie_scene_deep_black_real_start_pose_with_user_table_pipe.usda \
  --output-usd local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose/aloha2_menagerie_scene_deep_black_real_start_pose_proxy_runtime.usda \
  --output-dir reports/aloha1_isaac_adaptation/phase98_scene_proxy_runtime_convex_legacy_collisions_20260719 \
  --contact-proxy-profile scene_base_link \
  --include-regex 'finger_link$' \
  --bbox-scale 0.6 \
  --min-extent 0.005 \
  --proxy-contact-offset 0.02 \
  --proxy-rest-offset 0.0 \
  --proxy-static-friction 1.0 \
  --proxy-dynamic-friction 1.0
```

Build summary:

| Metric | Value |
| --- | --- |
| selected proxy count | `4` |
| disabled selected source collision count | `36` |
| disabled collision approximation count | `36` |

Artifact:

```text
.codex/artifacts/20260719-000725_aloha-phase98-rebuild-proxy-runtime-convex-legacy-collisions
```

## Phase97 Regression After Cleanup

The Phase97 recipe was rerun after rebuilding the proxy runtime stage.

Artifact:

```text
.codex/artifacts/20260719-000750_aloha-phase99-phase97-after-convex-legacy-collisions
```

Key result:

| Metric | Value |
| --- | --- |
| exit code | `0` |
| validator status | `PASS` |
| failure reasons | `[]` |
| contact trace status | `PASS_BILATERAL_CONTACT_CANDIDATE` |
| controller tracking gate | `PASS_POST_STEP_TRACKING_WITHIN_THRESHOLD` |
| active target contact gate | `SKIPPED_ALREADY_IN_CONTACT_SETUP` |
| stderr lines | `0` |
| PhysX triangle-mesh fallback matches | `0` |

## Interpretation

This resolves one residual risk from Phase97: the validator no longer depends on PhysX silently repairing disabled legacy finger mesh collision prims at runtime.

It still does not prove full grasp success. The same Phase97 scope boundaries remain:

- the contact setup is explicitly `already-in-contact`;
- the active target contact gate is skipped for this reference run;
- table/base calibration is not validated by this run;
- the active contact/grasp milestone still needs a separate gate where contact first appears during the close phase.
