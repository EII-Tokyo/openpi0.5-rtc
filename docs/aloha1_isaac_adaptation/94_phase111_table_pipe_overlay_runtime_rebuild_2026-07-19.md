# Phase 111: Table/Pipe Overlay Runtime Rebuild

## Question

Can the user-measured table and pipe overlay be rebuilt into the Trossen/Menagerie ALOHA scene without relying on the stale `/scene/worldBody` rail/table colliders as semantic workcell support?

## Evidence

- Builder artifact: `.codex/artifacts/20260719-011451_aloha-phase111-rebuild-proxy-runtime-after-user-table-pipe`
- Stage audit artifact: `.codex/artifacts/20260719-011646_aloha-phase111-table-pipe-prim-audit-after-rebuild`
- Re-run of Phase110 after rebuild: `.codex/artifacts/20260719-011706_aloha-phase110-after-rebuilt-table-pipe-runtime`

## Result

The measured overlay was rebuilt and the world-aligned bounding boxes are now audited correctly.

Important measured overlay prims:

| Prim | Result |
| --- | --- |
| `/World/Table` | Collision enabled; bbox size about `1.22 m x 0.625 m x 0.04 m`. |
| `/World/PipePlaceholder/axis` | Collision enabled; pipe diameter about `0.005 m`. |
| `/World/PipePlaceholder/support_base_placeholder` | Collision enabled; support base proxy exists. |

Phase110 still failed after this rebuild. The useful result is that it failed for a narrower reason:

- old `/scene/worldBody/table` and `/scene/worldBody/__22` colliders still participated;
- the measured `/World/Table` exists, but the old Menagerie workcell colliders were still active.

## Implementation Lesson

The first bbox audit was misleading because it used `ComputeWorldBound(...).GetBox()`. For scaled or rotated USD geometry, the correct world-aligned size for geometry audit is:

```python
cache.ComputeWorldBound(prim).ComputeAlignedBox()
```

This change matters because policy decisions such as "is this collider the real table or a frame rail" depend on world dimensions, not local unaligned ranges.

## Consequence

Phase111 did not prove the workcell contact gate. It proved:

1. the measured overlay can be present in the runtime stage;
2. the audit can now report correct world-aligned bbox sizes;
3. stale `/scene/worldBody` colliders must be explicitly disabled or policy-gated before a strict contact gate can be meaningful.

