# Phase 121: Failure HDF5 Cluster Counterexample

## Question

Phase120 showed that three `reward=1` success segments produce a tight empirical bottle-mouth entry cluster. A useful diagnostic must also pass the counterexample:

```text
Do reward=0 failure segments stay away from the same empirical entry cluster?
```

If failures cluster at the same place, final mouth position and axis alone are not enough to decide success.

## Method

Run the same runner with `reward=0` and a separate output directory:

```bash
.venv/bin/python aloha_isaac_replay/scripts/run_phase120_success_hdf5_empirical_pipe_cluster.py \
  --reward 0 \
  --limit 3 \
  --output-dir reports/aloha1_isaac_adaptation/phase121_failure_hdf5_empirical_pipe_cluster_20260719
```

Selected failure HDF5 segments:

| Key region | Replay gate | Fit |
| --- | --- | --- |
| `cd167a72170848bbad39e3c110f15ec7` | PASS | PASS |
| `692fbb41241b4465a47df5120ca9ed5b` | PASS | PASS |
| `1dbbc3d72dc445f183c4c0b50334bad5` | PASS | PASS |

## Result

Success cluster from Phase120:

```text
entry mean = [-0.1785, 0.3122, 0.3646]
axis mean  = [0.5042, 0.1970, 0.8408]
entry RMS spread = 0.0068 m
```

Failure cluster:

```text
entry mean = [-0.1944, 0.3239, 0.3787]
axis mean  = [0.4790, 0.2684, 0.8358]
entry RMS spread = 0.0151 m
```

Failure minus success:

```text
entry delta = [-0.0159, +0.0117, +0.0141] m
entry distance = 0.0242 m
axis angle = 4.35 deg
```

## Interpretation

This is a negative result for a simple geometry-only success detector:

- success segments cluster tightly;
- failure segments also cluster near the same region;
- success/failure mean entry distance is only about 2.4 cm;
- success/failure mean axis differs by only about 4.3 degrees.

Therefore final bottle-mouth position and axis are useful for checking workcell frame consistency, but they are not sufficient to separate success and failure.

This matches the real task intuition:

```text
Near-miss failure can look geometrically close to success.
Insertion success depends on contact, depth, alignment tolerance, and whether the bottle mouth actually enters the pipe.
```

## Consequence

Do not use the current empirical entry cluster as an automatic reward. It should be used for calibration only.

For reward/critic supervision, the next signal should include at least one of:

1. insertion depth along the pipe axis;
2. distance to pipe entry plus direction alignment;
3. temporal monotonicity toward the pipe, not just final pose;
4. contact/penetration or geometry overlap in a calibrated simulator;
5. visual confirmation from `cam_low` or `cam_right_wrist`.

## Next Work

1. Fix workcell calibration first, because current measured pipe is far from both success and failure clusters.
2. After calibration, define richer geometric metrics:
   - final entry distance;
   - axis angle;
   - projected insertion depth;
   - lateral error;
   - end-of-trajectory stability.
3. Compare these metrics on more success and failure segments.
4. Only then decide whether Isaac geometry can provide an automatic dense reward.
