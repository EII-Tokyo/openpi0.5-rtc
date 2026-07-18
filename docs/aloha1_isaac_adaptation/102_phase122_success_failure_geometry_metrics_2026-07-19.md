# Phase 122: Success/Failure Geometry Metrics

## Question

Phase121 showed that final bottle-mouth entry and axis alone do not cleanly separate success and failure. Phase122 asks:

```text
Do richer trajectory metrics separate success and failure better?
```

## Method

Run:

```bash
.venv/bin/python aloha_isaac_replay/scripts/compare_phase122_success_failure_geometry_metrics.py
```

Inputs:

- success replays from Phase120;
- failure replays from Phase121;
- the Phase120 success empirical entry and axis as the reference.

Outputs:

- metrics JSON: `reports/aloha1_isaac_adaptation/phase122_success_failure_geometry_metrics_20260719/success_failure_geometry_metrics.json`
- plot: `reports/aloha1_isaac_adaptation/phase122_success_failure_geometry_metrics_20260719/success_failure_geometry_metrics.png`

Metrics:

| Metric | Meaning |
| --- | --- |
| `path_length_m` | total mouth path length |
| `net_displacement_m` | start-to-end mouth displacement |
| `tail_lateral_mean_m` | final-tail lateral distance to the success empirical axis |
| `tail_progress_m` | final-tail progress along the success empirical axis relative to the start |

## Result

| Metric | Success mean | Failure mean |
| --- | ---: | ---: |
| path length | about `0.1618 m` | about `0.0978 m` |
| net displacement | about `0.1498 m` | about `0.0845 m` |
| tail lateral mean | about `0.0111 m` | about `0.0247 m` |
| tail progress | about `0.0221 m` | about `0.0141 m` |

## Interpretation

Richer trajectory metrics are more promising than final pose alone:

- success has larger path length and net displacement;
- success has lower final-tail lateral error on average;
- success has somewhat larger progress along the empirical axis.

But the result is not yet strong enough for an automatic reward:

- only three success and three failure samples were tested;
- one failure sample has low lateral error, so lateral error alone is not reliable;
- failure spread is high, which means failures include multiple modes.

## Engineering Conclusion

For Isaac-based reward or critic diagnostics, do not use a single final-position threshold. A better candidate is a multi-term score:

```text
score = insertion_progress - lateral_error - instability_penalty
```

This is only a conceptual scoring direction. It should not be wired into training until:

1. workcell calibration is fixed;
2. the same metrics are computed on more labeled success/failure segments;
3. thresholds are chosen from distributions, not by hand.

## Next Work

1. Increase success/failure sample counts.
2. Plot metric distributions for all available 2026-07-08 labeled segments.
3. Add temporal monotonicity metrics.
4. After workcell calibration, define pipe-axis projected insertion depth.
5. Re-evaluate whether geometry can become a dense reward signal.
