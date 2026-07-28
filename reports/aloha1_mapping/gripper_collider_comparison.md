# ALOHA 1 Gripper Collider Comparison

Status: **PASS**

This is a diagnostic comparison. Convex decomposition is supported by the local NVIDIA PhysX schema, but it is not assumed to be the correct final collider and does not produce an exact collider.

## Local Isaac 5.1 readback

- URDF Importer: `2.4.30`
- `ImportConfig.convex_decomp` initial readback: `False`
- Approximation tokens tested: `convexHull`, `convexDecomposition`

| Attribute | Local schema default | Authored by diagnostic layer |
| --- | ---: | --- |
| `errorPercentage` | `10.0` | `False` |
| `hullVertexLimit` | `64` | `False` |
| `maxConvexHulls` | `32` | `False` |
| `minThickness` | `0.0010000000474974513` | `False` |
| `shrinkWrap` | `False` | `False` |
| `voxelResolution` | `500000` | `False` |

## Cooked geometry

| Profile | Robot | Side | Pieces | Sum convex volume (m³) | GPU warning count |
| --- | --- | --- | ---: | ---: | ---: |
| convex_decomposition | follower_left | left | 32 | 3.701840437350787e-05 | 0 |
| convex_decomposition | follower_left | right | 32 | 3.701840437350787e-05 | 0 |
| convex_decomposition | follower_right | left | 32 | 3.701840437350787e-05 | 0 |
| convex_decomposition | follower_right | right | 32 | 3.701840437350787e-05 | 0 |
| convex_hull | follower_left | left | 1 | 6.031490884162812e-05 | 0 |
| convex_hull | follower_left | right | 1 | 6.031490884162812e-05 | 0 |
| convex_hull | follower_right | left | 1 | 6.031490884162812e-05 | 0 |
| convex_hull | follower_right | right | 1 | 6.031490884162812e-05 | 0 |

The source STL is not watertight, so its mesh volume is not presented as calibrated solid volume. Cooked-piece volume, AABB, and deterministic surface sampling are retained in the JSON report.

## Numeric interpretation

- Hull/decomposition cooked-volume ratio: `1.6293222212676552`.
- Decomposition piece count: `32` (local default maximum `32`).
- Hull source-distance p95: `0.006895076072061914 m`; decomposition: `0.0022830804260642935 m`.

The numeric evidence supports that the single hull bridges STL concavities. It does not by itself prove that the bridged region is the calibrated inner fingertip contact surface.

## Frozen runtime A/B

- Final status: `NO_MEANINGFUL_EFFECT`.
- Root cause: `neither_resolved`.
- Hull drop: `0.05196145176887512 m`; decomposition drop: `0.04741010069847107 m`; unchanged gate: `0.01 m`.
- Contact points per trial: Hull `2071.0`, decomposition `20581.0`.
- Mean runtime ratio decomposition/hull: `1.8251801184596237`.
