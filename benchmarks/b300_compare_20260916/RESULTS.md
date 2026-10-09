# B300 inference benchmark — 2026-09-16

## Scope and comparability

Same rinse cam4 April-only model, checkpoint 3000, four 448×448 cameras, prefix shape (1, 4296, 2048), 50-action horizon, 10 denoising steps. RTC replanning interval 25 and execution horizon 10. Fixed synthetic inputs; no robot or network round trip. Checkpoint downloaded directly from S3: 19 weights/assets files, 12,443,883,411 bytes, SHA256 verified; no train state.

B300 SXM6 AC, 275040 MiB, 1100 W limit, driver 595.84. JAX 0.5.3 failed to compile for compute capability 10.3, even a small BF16 matrix multiplication. Valid B300 measurements use JAX 0.7.2, Flax 0.11.2, Orbax 0.11.24, CUDA 13.0 libraries. The other GPUs below used JAX 0.5.3. These are cross-environment comparisons, NOT isolated hardware speedups. The B200 SSH endpoint refused connections, so a same-version control could not be obtained. See COMPATIBILITY.md and requirements.txt.

Only two B300-local compatibility changes: unwrap Orbax metadata.item_metadata when present; replace removed nnx.Dict with an nnx.Module holding the same llm/img attributes. No source changes to 103 production. Compiler warned that no B300-specific SoL config was available and used defaults. No output-quality evaluation was performed.

## GPU-resident model calls

Separate process per batch; 3 warmups, 30 synchronized timed repetitions per mode. Means in milliseconds per batch, including host dispatch/synchronization but excluding input preprocessing and output postprocessing. All outputs finite; no OOM.

| Batch | Plain mean | Plain P95 | RTC mean | RTC P95 | Plain samples/s | RTC samples/s |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 66.56 | 67.25 | 107.43 | 107.97 | 15.02 | 9.31 |
| 2 | 135.61 | 140.77 | 186.56 | 187.63 | 14.75 | 10.72 |
| 4 | 253.26 | 253.65 | 329.17 | 330.25 | 15.79 | 12.15 |
| 8 | 475.65 | 477.92 | 598.70 | 599.27 | 16.82 | 13.36 |

## Batch-1 comparison

| GPU / software | Model plain | Model RTC | Full policy plain | Full policy RTC |
|---|---:|---:|---:|---:|
| RTX 5090 / JAX 0.5.3 | 189.96 | 227.35 | 219.41 | 259.82 |
| H100 / JAX 0.5.3 | 103.04 | 149.75 | 160.01 | 203.86 |
| B200 / JAX 0.5.3 | 92.14 | 136.19 | 115.33 | 159.76 |
| B300 / JAX 0.7.2 | 66.56 | 107.43 | 120.60 | 163.85 |

Full policy: 100 measured calls after compilation/warmup, including preprocessing, transfer and postprocessing, excluding networking and robot execution. H100 numbers are from the matched preprocessing-dependency rerun.

B300 full plain: median 117.54 ms, P95 140.34 ms, model-call mean 67.05 ms, remaining mean 53.55 ms. Full RTC: median 159.10 ms, P95 198.79 ms, model-call mean 108.02 ms, remaining mean 55.83 ms. B200 remaining time was about 22 ms. Therefore faster model execution did not translate to faster end-to-end policy calls on this host. The remaining-time difference is observed, not conclusively attributed to a particular CPU or transfer bottleneck.

Compared with prior B200 measurements, B300 model-call latency is 27.8% lower plain and 21.1% lower RTC; full policy is 4.6% and 2.6% higher respectively. Software, host CPU and driver differences prevent attributing these numbers solely to GPU architecture.

## Memory and artifacts

Full policy warmed GPU usage: 17126 MiB (16.72 GiB). JAX live allocation peak 8,293,342,976 bytes; pool peak 17,177,772,032 bytes. Pool reservation is not the same as live tensor memory. Batch-8 pool peak was 34,357,641,216 bytes; live allocation peak 16,705,453,056 bytes. No OOM at batches 1/2/4/8.

Raw records: results.jsonl, policy.json, batch*.log, batch*_gpu.csv, policy.log, environment.json, requirements.txt. Telemetry spans loading, compilation, warmup and measurement; it must not be treated as measurement-only utilization. Remote artifacts: /workspace/benchmarks/b300_compare/jax072_compat2. Test processes exit after completion; the instance is not shut down.
