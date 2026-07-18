# Phase 119: Empirical Pipe Candidate From Replay

## Question

Phase118 showed that the current measured pipe axis is not compatible with the Phase117 carried-bottle trajectory. Phase119 asks:

```text
If the final part of this replay really represents insertion, where would the pipe entry and pipe axis be in Isaac world coordinates?
```

This is a diagnostic fit, not a calibration replacement.

## Method

Run:

```bash
.venv/bin/python aloha_isaac_replay/scripts/fit_phase119_empirical_pipe_candidate.py
```

Inputs:

- Phase117 CSV: `reports/aloha1_isaac_adaptation/phase117_diagnostic_held_bottle_replay_20260719/gripper_passive_contact_timeseries.csv`
- measured workcell config: `examples/aloha_isaac/config/workcell_user_measured.yaml`

Outputs:

- summary JSON: `reports/aloha1_isaac_adaptation/phase119_empirical_pipe_candidate_20260719/empirical_pipe_candidate_summary.json`
- plot: `reports/aloha1_isaac_adaptation/phase119_empirical_pipe_candidate_20260719/empirical_pipe_candidate.png`

The script:

1. takes the final 20 percent of the bottle-mouth trajectory;
2. averages the final bottle-mouth positions as an empirical pipe-entry candidate;
3. averages the final bottle-mouth local axis as an empirical base-to-entry pipe direction candidate;
4. projects backward by the measured pipe length to get an empirical pipe-base candidate;
5. compares that candidate with the hand-measured workcell pipe.

## Result

| Quantity | Value |
| --- | --- |
| final-tail samples | `29` |
| empirical entry | `[-0.0801, 0.3331, 0.3134]` |
| empirical axis | `[0.6544, 0.2216, 0.7229]` |
| empirical base | `[-0.2274, 0.2832, 0.1507]` |
| measured base | `[-0.0300, 0.4075, 0.0700]` |
| measured entry | `[-0.1919, 0.4075, 0.2263]` |
| candidate minus measured entry | `[+0.1117, -0.0744, +0.0871] m` |
| candidate minus measured base | `[-0.1974, -0.1243, +0.0807] m` |
| candidate/measured axis angle | about `88.2 deg` |
| final-tail RMS spread | about `0.0143 m` |

## Interpretation

The empirical candidate is not close to the hand-measured pipe transform. The mismatch is too large to be explained by small contact noise:

- entry x differs by about 11 cm;
- entry y differs by about 7 cm;
- entry z differs by about 9 cm;
- axis differs by about 88 degrees.

This means at least one major frame assumption is still wrong:

1. the measured pipe direction was encoded with the wrong table-axis convention;
2. the table frame and robot base frame are not aligned to the replay frame;
3. the diagnostic held-bottle replay is not yet using the same object frame that corresponds to the real bottle in video;
4. the selected HDF5 segment may not actually end at the physical pipe entry.

## Boundary

Do not copy the empirical candidate into `workcell_user_measured.yaml` automatically. It is a measurement target:

```text
The empirical candidate tells us what the replay implies.
It does not tell us what the physical workcell truly is.
```

## Next Work

1. Run the same empirical fit on several confirmed successful insertion segments.
2. Check whether the empirical entries cluster tightly.
3. If they do, use that cluster to debug table/robot/pipe frame alignment.
4. If they do not, the problem is likely the held-object frame, the grasp transform, or the segment selection.
5. Only after this clustering check should the measured workcell config be changed.
