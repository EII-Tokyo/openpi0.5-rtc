# Phase 123: Limit-10 Success/Failure Geometry Check

## Question

Phase120 to Phase122 used three success and three failure HDF5 segments. That was enough to expose useful signals, but too small to trust.

Phase123 expands the same diagnostic to ten success and ten failure HDF5 segments:

```text
Do the Phase122 geometry differences remain visible when the sample count increases from 3 to 10 per group?
```

## Inputs

Success cluster:

```text
reports/aloha1_isaac_adaptation/phase120_success_hdf5_empirical_pipe_cluster_20260719_limit10/phase120_cluster_summary.json
```

Failure cluster:

```text
reports/aloha1_isaac_adaptation/phase121_failure_hdf5_empirical_pipe_cluster_20260719_limit10/phase120_cluster_summary.json
```

Geometry comparison:

```text
reports/aloha1_isaac_adaptation/phase122_success_failure_geometry_metrics_20260719_limit10/success_failure_geometry_metrics.json
```

Full command logs:

```text
.codex/artifacts/20260719-020052_aloha-phase120-success-cluster-limit10
.codex/artifacts/20260719-020227_aloha-phase121-failure-cluster-limit10
.codex/artifacts/20260719-020520_aloha-phase122-geometry-metrics-limit10
```

## Cluster Result

| Group | Usable HDF5 count | Replay gate pass | Fit used despite replay gate warning | Empirical entry mean |
| --- | ---: | ---: | ---: | --- |
| success | 10 | 6 | 4 | `[-0.1671, 0.3120, 0.3628] m` |
| failure | 10 | 7 | 3 | `[-0.1869, 0.3166, 0.3681] m` |

The success and failure empirical final entries are close. Their mean distance is only about `0.0209 m`.

This confirms the Phase121 warning: final bottle-mouth pose alone is not a reliable success/failure separator.

## Geometry Metric Result

| Metric | Success mean | Failure mean | Direction |
| --- | ---: | ---: | --- |
| `path_length_m` | `0.1610 m` | `0.1220 m` | success larger |
| `net_displacement_m` | `0.1414 m` | `0.1079 m` | success larger |
| `tail_lateral_mean_m` | `0.0130 m` | `0.0235 m` | success smaller |
| `tail_progress_m` | `0.0135 m` | `0.0138 m` | nearly tied |

The most stable differences are:

1. Successful segments move the bottle mouth farther overall.
2. Successful segments have lower final-tail lateral error relative to the empirical success axis.

The weak difference is:

```text
tail_progress_m
```

Progress alone does not separate success and failure in this sample.

## Interpretation

The limit-10 result is more useful than the limit-3 result because the same qualitative trend still appears after adding more HDF5s:

- success has longer mouth trajectory;
- success has larger net displacement;
- success has lower final-tail lateral error;
- final pose and raw progress are not enough.

This supports using trajectory-level diagnostics rather than a single terminal pose threshold.

But it is still not a training-ready dense reward. The current diagnostic is based on replayed kinematic mouth trajectories and an empirical success axis. It does not yet include:

- contact with the pipe;
- bottle deformation or slip;
- true calibrated pipe pose;
- force/current/contact evidence;
- camera reprojection validation.

## Engineering Conclusion

The current Isaac replay path is now useful for calibration and reward-feature discovery, not yet for final RL reward generation.

The most promising feature family is:

```text
success_like_motion = long_enough_motion + small_tail_lateral_error + stable_tail_direction
```

The least reliable feature family is:

```text
success_like_motion = final_pose_only
```

or:

```text
success_like_motion = progress_only
```

## Next Gate

Before using these signals for actor or critic training, the next Isaac gate should be:

1. replay more labeled HDF5s, preferably all available same-task success/failure clips;
2. add tail-direction stability and velocity smoothness metrics;
3. compare metrics against human labels;
4. calibrate the pipe/table transform using real measured geometry or image evidence;
5. only then define a candidate dense reward.

The immediate next implementation task should not be full RL. It should be a stronger replay evaluator:

```text
HDF5 -> bottle mouth trajectory -> geometric metrics -> label correlation report
```

