# ALOHA1 Task 8 benchmark comparison

Status: `NO_MEASURABLE_IMPROVEMENT`

Three fresh Isaac Sim 5.1 processes per profile used the local official `isaacsim.benchmark.services` frame and memory recorders. Lower is better; only nonoverlapping run ranges are classified as directional evidence.

| Metric | Baseline mean | Candidate mean | Delta | Classification |
|---|---:|---:|---:|---|
| stage_load_ms | 174.112 | 171.103 | -1.729% | `INCONCLUSIVE_OVERLAPPING_RANGE` |
| app_frame_ms | 3.55 | 3.68333 | 3.756% | `INCONCLUSIVE_OVERLAPPING_RANGE` |
| physics_frame_ms | 0.913333 | 1.01 | 10.584% | `WORSENS_NONOVERLAPPING_RANGE` |
| rss_gb | 4.64133 | 4.69933 | 1.250% | `INCONCLUSIVE_OVERLAPPING_RANGE` |
| gpu_dedicated_gb | 1.206 | 1.20467 | -0.111% | `INCONCLUSIVE_OVERLAPPING_RANGE` |

The candidate is not promoted and no grasp smoke is run because it has no nonoverlapping improvement while physics frame time is reproducibly worse. No screenshot or video is applicable to this sub-millisecond performance-only negative result; no visible render, collision or grasp failure occurred.
