# Correct ALOHA Finger Collider Geometry Audit

Status: **PASS**

This audit uses only the user-confirmed custom ALOHA finger meshes and the local Isaac Sim 5.1 / Kit 107.3.3 / PhysX 107.3.26 runtime.

| Profile | Robot | Side | Approximation readback | Cooked pieces | Sum convex volume (m³) |
| --- | --- | --- | --- | ---: | ---: |
| `convex_hull` | `follower_left` | `left` | `convexHull` | 1 | 6.384204789525298e-05 |
| `convex_hull` | `follower_left` | `right` | `convexHull` | 1 | 6.384204789525297e-05 |
| `convex_hull` | `follower_right` | `left` | `convexHull` | 1 | 6.384204789525298e-05 |
| `convex_hull` | `follower_right` | `right` | `convexHull` | 1 | 6.384204789525297e-05 |
| `convex_decomposition` | `follower_left` | `left` | `convexDecomposition` | 32 | 3.3386459101488415e-05 |
| `convex_decomposition` | `follower_left` | `right` | `convexDecomposition` | 32 | 3.4297885109060535e-05 |
| `convex_decomposition` | `follower_right` | `left` | `convexDecomposition` | 32 | 3.3386459101488415e-05 |
| `convex_decomposition` | `follower_right` | `right` | `convexDecomposition` | 32 | 3.4297885109060535e-05 |

The PNGs are numerical cooked-collider visualizations. They are supplemental to, not substitutes for, the Isaac runtime contact and hold screenshots.

- Screenshot root: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-correct-finger-task5/screenshots`
- Full screenshot manifest: `/home/eii/project/openpi0.5-rtc-reward-learning/reports/aloha1_mapping/gripper_correct_finger_all_screenshot_manifest.json`
- Default collider: **unchanged**
- Task 8: **NOT_RUN**
