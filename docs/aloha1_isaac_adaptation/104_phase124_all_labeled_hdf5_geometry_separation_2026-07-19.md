# Phase 124: All Labeled HDF5 Geometry Separation

## Question

Phase123 showed that ten success and ten failure clips preserve some success/failure geometry differences. Phase124 expands the diagnostic to all currently selected labeled HDF5 key regions on `2026-07-08`:

```text
52 success clips versus 77 failure clips
```

The goal is not to create a final reward yet. The goal is to test whether the geometry features remain correlated with human labels at a larger sample size.

## Inputs

Success replay cluster:

```text
reports/aloha1_isaac_adaptation/phase120_success_hdf5_empirical_pipe_cluster_20260719_all52/phase120_cluster_summary.json
```

Failure replay cluster:

```text
reports/aloha1_isaac_adaptation/phase121_failure_hdf5_empirical_pipe_cluster_20260719_all77/phase120_cluster_summary.json
```

Geometry comparison:

```text
reports/aloha1_isaac_adaptation/phase122_success_failure_geometry_metrics_20260719_all52_all77/success_failure_geometry_metrics.json
reports/aloha1_isaac_adaptation/phase122_success_failure_geometry_metrics_20260719_all52_all77/success_failure_geometry_metric_separation.csv
reports/aloha1_isaac_adaptation/phase122_success_failure_geometry_metrics_20260719_all52_all77/success_failure_geometry_metrics.png
```

Full command logs:

```text
.codex/artifacts/20260719-021050_aloha-phase120-success-cluster-all52
.codex/artifacts/20260719-021838_aloha-phase121-failure-cluster-all77
.codex/artifacts/20260719-023016_aloha-phase122-geometry-metrics-all52-all77
```

## Replay Gate Summary

| Group | Usable fits | Replay gate pass | Fit used despite replay gate warning |
| --- | ---: | ---: | ---: |
| success | 52 | 38 | 14 |
| failure | 77 | 48 | 29 |

The replay gate warnings are mostly controller-tracking threshold warnings. They are still important, but they did not prevent extracting bottle-mouth trajectories for this diagnostic.

## Final Entry Cluster

| Group | Entry mean | Entry RMS spread |
| --- | --- | ---: |
| success | `[-0.1551, 0.3123, 0.3570] m` | `0.0205 m` |
| failure | `[-0.1802, 0.3138, 0.3643] m` | `0.0215 m` |

The entry means differ by only a few centimeters and both groups have roughly `2 cm` spread.

This reinforces the earlier conclusion:

```text
final bottle-mouth position is not enough
```

## Metric Separation

AUC means the probability that a randomly sampled success clip is ranked better than a randomly sampled failure clip by that metric, using the expected direction.

| Metric | Expected success direction | Success mean | Failure mean | AUC |
| --- | --- | ---: | ---: | ---: |
| `path_length_m` | higher | `0.1575` | `0.1283` | `0.829` |
| `net_displacement_m` | higher | `0.1378` | `0.1114` | `0.814` |
| `tail_lateral_mean_m` | lower | `0.0196` | `0.0289` | `0.723` |
| `tail_lateral_max_m` | lower | `0.0337` | `0.0389` | `0.622` |
| `tail_progress_m` | higher | `0.0133` | `0.0146` | `0.498` |

## Interpretation

The larger sample confirms three things:

1. `path_length_m` and `net_displacement_m` are consistently useful. Their AUC stays above `0.80`.
2. `tail_lateral_mean_m` is moderately useful. It captures alignment quality better than a single final point.
3. `tail_progress_m` is not useful. It is almost random on the full sample.

This means the useful signal is not simply "the bottle moved forward." The useful signal is closer to:

```text
the mouth followed a sufficiently long, stable, success-like approach trajectory
```

## Why This Matters For Isaac RL

If Isaac Sim is used to create an automatic score, the score should not be based on one terminal threshold. It should start from trajectory features that already show label correlation:

```text
candidate_score =
  path_or_displacement_term
  - average_lateral_error_term
  - instability_term
```

This is still a candidate scoring family, not a final reward. It needs contact and calibration before training.

## Current Limitations

- The current replay is kinematic diagnostic replay, not full contact-rich insertion simulation.
- The empirical success axis is derived from HDF5 replayed mouth motion, not from a calibrated physical pipe pose.
- Replay gate warnings remain common.
- Labels are binary human labels, so near-miss and bad-failure modes are mixed inside the failure group.
- Geometry features explain part of the label, but not all of it.

## Engineering Decision

Proceed with geometry-feature evaluation as a diagnostic layer.

Do not yet use these features as the sole critic reward.

The next safe implementation step is:

```text
Add a replay-evaluator report that computes these metrics for any selected HDF5 set and compares them against labels.
```

The next unsafe step would be:

```text
Immediately wiring these metrics into RL training as dense reward.
```

That would be premature because contact, pipe calibration, and failure-mode separation are not yet validated.

