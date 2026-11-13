# FastUMI midtraining and posttraining preparation

Branch: `train/pi05-fastumi-mid-post`.
Status: metadata inspected and a proposed public episode split generated. This
is not a runnable training recipe yet. No training or bulk video download has
started. Phase budgets, initialization, source scope, and sampling weights await
user decisions.

## Proposed sources and split

Public source: <https://huggingface.co/datasets/IPEC-COMMUNITY/FastUMI_100k_lerobot>,
pinned to `4cf7cca8fdc15d6f71ee0055553ce8a5e78aa9a0`.
The existing one-arm policy fits its 12 single-arm tasks: 41,999 episodes and
14,418,308 frames. The repository also contains dual-arm data, which this
proposal excludes pending confirmation. The advertised 100k is not the count
of eligible single-arm episodes in this pinned revision.

Fan sources are exactly the three immutable revisions in `run3_datasets.json`:
403 episodes and 33,368 frames. Include every fan episode in both phases.

`mid_post_public_split.proposed.json` records a draft selection of 403 public
episodes, sampled uniformly without replacement using seed 42. A seeded SHA256
ranking makes selection independent of filesystem enumeration and NumPy versions.
Episode identity includes the task path because episode indices restart per task.

| Phase | Fan episodes | Public episodes | Public frames |
| --- | ---: | ---: | ---: |
| Midtrain | 403 | 41,596 | 14,278,432 |
| Posttrain | 403 | 403 | 139,876 |

The public partitions are disjoint and exhaustive. Reserve the entire episode,
including all frames and action chunks. Public posttraining episodes must be
excluded from midtraining samples **and normalization statistics**. This is a
phase split, not a validation split: posttraining trains on its reserved episodes.
The split establishes disjoint episode IDs; it does not prove absence of content
duplicates across different episode IDs. Check trajectory duplicates during the
full preparation audit before accepting the split.

## Sampling and phase transition

- Proposed midtraining mixture: 10% fan samples, 90% public samples. Uniform
  pooling instead would expose the fan task only about 0.23% of the time.
  This ratio awaits the user's choice.
- Required posttraining mixture: 50% fan samples, 50% reserved public samples.
  With batch size 32, use 16 examples from each group per batch, sampling with
  replacement. Within each group, sample frames uniformly. Equal episode counts
  alone do not give a 50/50 mixture because public episodes are longer.
- Proposed initialization: fresh pi05 base for midtraining, then load the chosen
  midtraining checkpoint for posttraining with a fresh optimizer and schedule.
  Starting from run 3 is an alternative pending the user's choice.
- Fit normalization on the approved midtraining sampling distribution only.
  Carry the same stats into posttraining so the inherited model retains the same
  input/output coordinates. Do not include reserved public episodes while fitting
  midtraining stats, and do not silently reuse run 3 fan-only statistics.
- Keep the existing relative pose representation, gripper-only state, wrist
  camera, horizon 16 at 20 Hz, and color-only image augmentation.
- Keep `put the fan in the box` for fan episodes. Public examples must retain
  their own task text in both phases. The existing run 3 transform overrides all
  task text and therefore cannot be reused unchanged for this mixture.
- W&B must use separate phase runs; S3 must use separate phase prefixes. Finalize
  these names and backup schedules together with the phase step budgets.

## Verified schema differences

`mid_post_source_audit.json` records metadata and a pose sample from episode 0
of each public task. These are sample checks, not a full dataset audit.

| Property | Fan datasets | Public single-arm datasets |
| --- | --- | --- |
| RGB key | `observation.images.front` | `observation.images.camera_rgb_image` |
| Pose | 7D XYZ/RPY/gripper | 7D XYZ/RPY/gripper |
| Action alignment | `action[t] == state[t]` | Sampled episodes: `action[t] == state[t+1]`, final frame clamped |
| Action chunk offset | +1 | 0 |
| Language | Fixed fan instruction | Dataset task instruction |

Both adapters must produce target poses at t+1 through t+16 relative to the
observation-time TCP. Public poses start at the episode origin in inspected
samples; they must still be converted to the current observation frame, without
subtracting their motion twice. Verify Euler convention and gripper orientation
with decoded images and trajectories before accepting the adapter.

## Reproduce the proposed split

This downloads only metadata, pins the revision, validates counts/schema/task
labels, and refuses to overwrite a different existing split:

```bash
.venv/bin/python examples/fastumi/prepare_two_phase.py \
  --posttrain-public-episodes=403 \
  --seed=42 \
  --output=examples/fastumi/mid_post_public_split.proposed.json

.venv/bin/python -m pytest -q examples/fastumi/two_phase_test.py
```

Before training, implement and test source-specific camera/prompt/timing adapters,
episode filtering, exact group-balanced sampling, phase checkpoint handoff, and
normalization provenance. Audit all selected pose files and the video/task
alignment. Check remote storage before downloading: the complete public repo is
approximately 1.4 TB, and conversion can retain both source and converted copies.
Do not download the full repository onto the local evaluation disk.
