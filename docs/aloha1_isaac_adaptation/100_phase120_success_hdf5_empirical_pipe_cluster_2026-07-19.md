# Phase 120: Empirical Pipe Cluster From Three Success HDF5 Segments

## Question

Phase119 fit an empirical pipe candidate from one diagnostic replay. A single trajectory can be misleading. Phase120 repeats the same fit on multiple locally annotated success HDF5 segments:

```text
Do reward=1 success key regions imply a consistent pipe-entry cluster?
```

## Method

Run:

```bash
.venv/bin/python aloha_isaac_replay/scripts/run_phase120_success_hdf5_empirical_pipe_cluster.py --limit 3
```

The runner:

1. reads `/home/eii/data/openpi0.5-rtc-reward-learning/segment_db/segments.sqlite3`;
2. selects recent `reward=1` key regions under `2026-07-08`;
3. runs the Phase117 held-bottle Isaac replay for each HDF5 from frame `0`;
4. fits an empirical pipe candidate from the final 20 percent of each bottle-mouth trajectory;
5. aggregates the fitted entries and axes.

Full output is under:

```text
reports/aloha1_isaac_adaptation/phase120_success_hdf5_empirical_pipe_cluster_20260719/
```

The runner keeps replay gate status as a quality label. A replay can fail a controller gate but still provide a CSV for geometry fitting. This happened once here: the failed segment exceeded the post-step tracking threshold by only about `0.0003 rad`.

## Selected HDF5 Segments

| Key region | Replay gate | Fit |
| --- | --- | --- |
| `e30de75b9779488590d233d97edb0482` | PASS | PASS |
| `86380e1e17204780a8c5d5d291c7131a` | PASS | PASS |
| `fd269c0b9bf6488f86f133a7d7ba537e` | FAILED_GATE | PASS |

The failed replay gate reason was:

```text
post_step_controller_tracking_exceeded_threshold
```

The maximum post-step controlled error was about `0.02030 rad`, slightly above the `0.02 rad` gate.

## Result

Aggregate empirical entry:

```text
[-0.1785, 0.3122, 0.3646]
```

Aggregate empirical axis:

```text
[0.5042, 0.1970, 0.8408]
```

Cluster spread:

| Metric | Value |
| --- | ---: |
| usable fits | `3` |
| replay gate pass count | `2` |
| replay gate failed but fit count | `1` |
| entry std x | about `0.0023 m` |
| entry std y | about `0.0031 m` |
| entry std z | about `0.0056 m` |
| entry RMS spread | about `0.0068 m` |

Compared with the current measured pipe entry:

```text
measured entry = [-0.1919, 0.4075, 0.2263]
```

the empirical cluster mean differs by:

```text
[+0.0133, -0.0953, +0.1383] m
```

The Euclidean entry difference is about `0.1685 m`.

The empirical axis and measured axis differ by about `77.2 deg`.

## Interpretation

This is the strongest geometry diagnostic so far:

- three success-labeled HDF5 segments produce a tight empirical entry cluster;
- the cluster spread is only about 7 mm;
- the cluster is far from the hand-measured pipe transform by about 17 cm;
- the axis disagreement is large.

Therefore the dominant problem is probably not random replay noise. The current Isaac workcell transform is not yet in the same frame as the successful insertion replay.

The most likely remaining causes are:

1. table coordinate convention is wrong;
2. pipe base/entry direction was encoded against the wrong axis;
3. ALOHA base-to-table transform is still approximate;
4. the HDF5 replay frame and measured table frame do not share a calibrated origin.

## Boundary

This is still diagnostic held-object replay:

```text
It proves geometric inconsistency.
It does not prove dynamic grasp realism.
```

Do not overwrite `workcell_user_measured.yaml` from this cluster alone. The next step should be a calibration step that explains why the empirical cluster and hand measurements differ.

## Next Work

1. Increase the cluster to more success segments if runtime permits.
2. Use the cluster mean as a target when calibrating table-to-left-base and pipe transforms.
3. Check whether the same cluster appears for failures; failures should not cluster at the same insertion entry.
4. Once the frame transform is fixed, rerun Phase117/120 and require empirical entries to land near the measured pipe entry.
