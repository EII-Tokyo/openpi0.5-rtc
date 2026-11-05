# Four-camera training with microbatch JIT

Training snapshot from the two-H200 server on 2026-11-05.

The Python loop calls one JIT for each microbatch and averages the gradients.
All microbatches read the same model parameters. AdamW, the step counter and EMA
update once per accumulation group. Disposable accumulators are donated during
accumulation; the training state is donated only for the optimizer update.
There is no stacked accumulation axis or list of per-microbatch gradient trees.

Active recipe:
`eii_with_rinse_cam4_fullft_all_98repo_h200x2_448_allcam_mem1s4hz_mosaic2x2_pool256`.
It uses 98 unique datasets (112 weighted entries), BS128, accumulation 2,
FSDP across two GPUs and 16 loader workers. Each of the four cameras supplies
four historical frames at [-1, -.75, -.5, -.25] seconds as a 2x2 448px mosaic,
plus the current 448px frame. Spatial pooling retains 256 image tokens per
camera; with 200 text positions, the prefix has 1224 tokens.
The previous high-camera-only layouts retain their existing defaults.

The included normalization statistics are the statistics used by this recipe.
Weights and datasets must be provisioned separately. Existing repository
`assets/` exclusions still apply to other assets and checkpoints.

## CPU regression checks

From the repository root, with the project environment installed:

```sh
JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
.venv/bin/python scripts/streaming_accumulation_test.py
```

The five checks cover accumulation 1/2/4, two successive updates, RNG behavior,
frozen parameters, AdamW, EMA, sharded layouts and exhaustion partway through a
group. The reference fixture is the unmodified `train_step` extracted from the
previous trainer (Git base a521612). Only test loading strips annotations.
No weights, dataset or W&B access is needed for these checks.

## Full-model numerical comparison

This optional test uses the real image pipeline and base weights. Run on idle
GPUs with sufficient memory (validated with two H200 NVL GPUs at BS128).
It does not start a training service, write checkpoints or contact W&B.
Configure the same dataset/cache environment as training, then run from the
repository root:

```sh
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0,1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
.venv/bin/python benchmarks/microbatch_jit/compare_full_model.py \
  --params-path /path/to/pi05_base/params
```

The two implementations reuse identical real observations, actions, starting
weights and random keys. The reference exposes its accumulated gradients as an
extra output. To fit both results in device memory, the candidate retains only
its starting weights initially and temporarily shares the reference's unused
optimizer/EMA inputs during gradient computation. Independent initial optimizer
and EMA buffers are restored before its update. This calls the production
microbatch JIT with the production input/output layouts. Comparisons reduce
errors on GPU; only scalar statistics are copied to the host.

Measured for one complete BS128 x 2 update:

| Output | Maximum absolute difference |
| --- | ---: |
| Loss | 0 (both 0.05467789247632027) |
| Accumulated gradients | 1.1175870895385742e-08 |
| Updated parameters | 0 |
| Optimizer state | 5.529727786779404e-10 |
| EMA | 0 |

See `numerical_result_h200.json`. Small intermediate floating-point differences
remain; these results do not guarantee bitwise-identical future training steps.

## Loader trial

Both runs used the microbatch-JIT trainer, the full recipe, BS128 x 2 and the
same two H200 GPUs. The first compilation update is excluded.

| Measurement | 4 workers | 16 workers |
| --- | ---: | ---: |
| Mean of updates 1-20 | 24.6145 s | 24.46115 s |
| Mean of updates 11-20 | 25.0581 s | 24.0944 s |
| Recorded loader wait, updates 11-20 | 2.7996 s | 1.7557 s |

The overall difference in this small sequential trial was only 0.6%. The
16-worker run had 33-36 second slow updates and a service memory peak of about
103.6 GiB. It remained active without OOM or restart, with at least 144 GiB
available host memory in the captured observations. Worker count alone did
not establish a clear overall win. See `worker16_comparison.json`.

`end_to_end_update_time` is the throughput measurement. Loader retrieval can
overlap device execution, so `data_wait_time` must not be added to that value.
`streaming_update_wall_time` includes retrieval and is not comparable directly
to the previous scan trainer's compute-only `train_step_time`.
