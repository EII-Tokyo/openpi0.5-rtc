# Phase 118: Pipe Axis Hypothesis Probe

## Question

Phase117 produced a stable diagnostic held-bottle replay and exported bottle-mouth positions and directions. The first trajectory plot showed that the replayed bottle mouth is far from the currently measured pipe axis.

Before changing the Isaac workcell, Phase118 asks a narrower question:

```text
Is the mismatch mainly a pipe direction mistake, a pipe/table/robot frame offset mistake, or both?
```

## Method

Run:

```bash
.venv/bin/python aloha_isaac_replay/scripts/probe_phase118_pipe_axis_hypotheses.py
```

Inputs:

- Phase117 CSV: `reports/aloha1_isaac_adaptation/phase117_diagnostic_held_bottle_replay_20260719/gripper_passive_contact_timeseries.csv`
- measured workcell config: `examples/aloha_isaac/config/workcell_user_measured.yaml`

Outputs:

- summary JSON: `reports/aloha1_isaac_adaptation/phase118_pipe_axis_hypothesis_probe_20260719/pipe_axis_hypothesis_summary.json`
- top-view plot: `reports/aloha1_isaac_adaptation/phase118_pipe_axis_hypothesis_probe_20260719/pipe_axis_hypotheses_top_view.png`

The script keeps the measured pipe base point fixed and tests four planar direction hypotheses with the same measured length and tilt:

| Hypothesis | Meaning |
| --- | --- |
| `x_negative_current` | current config |
| `x_positive_opposite` | opposite along the x direction |
| `y_negative_toward_table` | toward the table interior |
| `y_positive_outward` | outward from the table |

For each hypothesis it reports:

- final bottle-mouth to pipe-axis distance;
- minimum bottle-mouth to pipe-axis distance;
- unsigned angle between the final bottle-mouth axis and the candidate pipe axis.

## Result

| Hypothesis | Final distance to axis | Axis angle |
| --- | ---: | ---: |
| `x_positive_opposite` | about `0.2035 m` | about `13.0 deg` |
| `y_positive_outward` | about `0.2212 m` | about `49.2 deg` |
| `y_negative_toward_table` | about `0.1157 m` | about `70.2 deg` |
| `x_negative_current` | about `0.1612 m` | about `89.0 deg` |

Interpretation:

- The current pipe direction is almost orthogonal to the replayed bottle-mouth axis.
- Reversing the x direction makes the axis much more plausible, but the pipe is still too far from the replayed bottle-mouth path.
- Moving along y toward the table gives a smaller distance, but its direction is still poor.

Therefore the mismatch is not a single-parameter fix. It is likely a combination of:

1. pipe direction semantics;
2. pipe base position in the Isaac table frame;
3. robot base to table frame alignment.

## What This Does Not Prove

Phase118 does not prove the correct real pipe transform. It only shows that the current transform is not compatible with the replayed held-bottle trajectory.

The test still depends on the diagnostic held-object assumption from Phase117:

```text
The bottle is moved kinematically with the gripper.
This is not dynamic grasp/contact validation.
```

## Next Work

The next reliable step is to fit an empirical pipe-entry candidate from real successful insertion replay:

1. use only successful HDF5 segments;
2. compute the final bottle-mouth cluster;
3. compute the final bottle-mouth axis cluster;
4. compare that empirical candidate to the hand-measured pipe transform;
5. update the workcell only after the discrepancy is explained.

Do not silently change `workcell_user_measured.yaml` from the Phase118 result alone.
