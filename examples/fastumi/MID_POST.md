# FastUMI midtraining and posttraining preparation

Branch: `train/pi05-fastumi-mid-post`.
Status: the user authorized 200k midtraining steps followed by 100k posttraining
steps from the original pi05 base, with single-arm data only and the proposed
sampling ratios. The remote pipeline downloads/audits all sources, computes
midtraining-only normalization, runs prelaunch checks, and starts both phases
in sequence. Training remains gated on successful data preparation.

## Approved sources and split

Public source: <https://huggingface.co/datasets/IPEC-COMMUNITY/FastUMI_100k_lerobot>,
pinned to `4cf7cca8fdc15d6f71ee0055553ce8a5e78aa9a0`.
The existing one-arm policy fits its 12 single-arm tasks: 41,999 episodes and
14,418,308 frames. The repository also contains dual-arm data, which this
run excludes by user confirmation. The advertised 100k is not the count
of eligible single-arm episodes in this pinned revision.

Fan sources are exactly the three immutable revisions in `run3_datasets.json`:
403 episodes and 33,368 frames. Include every fan episode in both phases.

`mid_post_public_split.proposed.json` records the approved selection of 403 public
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
  The user approved this ratio.
- Required posttraining mixture: 50% fan samples, 50% reserved public samples.
  With batch size 32, use 16 examples from each group per batch, sampling with
  replacement. Within each group, sample frames uniformly. Equal episode counts
  alone do not give a 50/50 mixture because public episodes are longer.
- Proposed initialization: fresh pi05 base for midtraining, then load the chosen
  midtraining checkpoint for posttraining with a fresh optimizer and schedule.
  The user confirmed initialization from the original base, not run 3.
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

The implemented reader uses audited original episode files and compact numeric
arrays, avoiding a second copy of videos during conversion. Preparation audits
every selected pose file, task index, video frame count, and exact trajectory/video
duplicates across public phases. Real decoded samples and the full model transform
are checked before launch. Public videos are downloaded only on the GPU server.

## Authorized budget

The user reports run 3's 50k checkpoint outperformed 40k in robot evaluation.
The approved budget is 200k midtraining steps and 100k posttraining
steps at global batch 32. This is an experimental budget, not an estimated
optimum. At 50% fan sampling, 100k posttraining steps provide 1.6 million fan
examples, matching the exposure of run 3's 50k fan-only steps. Midtraining adds
640k fan examples and 5.76 million public examples (about 0.40 public frame
passes). Full public-frame coverage is not guaranteed.

Run 3 took approximately 11h47m including startup and checkpoint saves for 50k
steps. At the same speed, these phases take approximately 47h and 24h, totaling
71h. Reserve 3–4 days of training time for the larger dataset's possible loader
overhead, plus initial download/audit/normalization time. Re-estimate using
measured throughput once the larger loader runs. Evaluate posttraining
checkpoints at 50k, 75k, and 100k on the robot; loss alone does not determine
which checkpoint is best.

## Learning rates and execution

| Phase | Steps | Warmup | Peak LR | Final LR |
| --- | ---: | ---: | ---: | ---: |
| Midtrain | 200,000 | 1,000 | 2.5e-5 | 2.5e-6 |
| Posttrain | 100,000 | 1,000 | 1e-5 | 1e-6 |

Each phase uses its own cosine schedule and fresh AdamW state. Posttraining loads
the final midtraining EMA weights at step 199999. Both phases use full finetuning,
global batch 32, FSDP over two H100 GPUs, eight data workers, clip norm 1, EMA .999.
The reduced posttraining peak is an experimental choice, not a claimed optimum.

Normalization uses 1,048,576 seeded samples from the 10/90 midtraining mixture,
without decoding images or accessing reserved public episodes. Its provenance
records the source-index hash, sample count, seed, and resulting statistics hash.
Posttraining reuses these statistics exactly.

Remote checkout: `/home/ubuntu/openpi-fastumi-run4`. Environment:
`/home/ubuntu/fastumi-run4-env.sh`. Persistent sessions:
`fastumi-run4-prepare`, `fastumi-run4-pipeline`, and per-phase upload watchers.
Copies of the launch scripts are checked in as `run4_*.sh`; `run4_plan.json`
records the phase configuration and destinations.

W&B project: `eii-inc/openpi`, separate experiment names `umi_pi05_run4_midtrain`
and `umi_pi05_run4_posttrain`. S3 destinations use these same names under
`s3://openpi-tokyo/checkpoints/`. Full local checkpoints are saved every 5k steps;
retain midtraining 50k multiples and posttraining 25k multiples plus the latest.
Upload retained milestones for inference, and each final checkpoint with full
optimizer state. This limits new retained training checkpoints to roughly 350 GB.

A local detached process renews 36-hour temporary S3 credentials every six hours
for up to seven days, stopping when both phases and backups finish. It logs to
`/tmp/fastumi-run4-credentials.log`; PID is in
`/tmp/fastumi-run4-credentials.pid`. The local workstation must remain running
for credential refresh. If it goes offline longer than credential validity,
training continues remotely and the S3 watcher retries until credentials return.
No permanent AWS keys are copied to the GPU server.
