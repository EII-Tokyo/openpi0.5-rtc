# module_jit vs nnx.jit — 103, 2026-09-16

RTX 5090; JAX 0.5.3; Flax 0.10.2. Same four-camera 448 model, checkpoint 3000, batch 1, prefix 4296, ten denoising steps. Synthetic inputs resident on GPU, seed 42, explicit JAX key 123. Three warmups and 30 blocking timed calls per mode. Plain inference followed by RTC in each isolated process. No robot connections. Original production code unchanged; wrapper monkeypatched only inside benchmark process.

| Wrapper | JAX cumulative allocation peak (GiB) | Live after RTC (GiB) | Pool after RTC (GiB) | Process nvidia-smi (MiB) | Plain mean ms | RTC mean ms |
|---|---:|---:|---:|---:|---:|---:|
| Existing module_jit | 7.68578 | 6.46108 | 16.06055 | 17040 | 193.487 | 226.234 |
| nnx.jit, explicit model argument | 13.11201 | 6.61450 | 16.00586 | 16986 | 229.765 | 271.140 |
| nnx.jit(bound method) | unavailable | unavailable | unavailable | unavailable | failed | not reached |

Cumulative peak includes initialization, compilation/autotuning and warmup, not an isolated steady-state inference peak. Explicit nnx.jit peak exceeded baseline by 5.42623 GiB. Pool growth obscures this difference in nvidia-smi. Single run per wrapper, clocks not locked.

Bound-method case exited 137 during first compilation. Kernel log at 2026-09-16 08:08:13 UTC confirms global host OOM killing PID 388552; anonymous RSS 56,169,812 KiB (~53.57 GiB), shared RSS 1,531,904 KiB. No CUDA OOM established. It did not complete a warmup; no valid steady GPU-memory comparison exists. Explicit-model followup used a 40 GiB container host-memory/swap limit for safety.

RTC outputs shape (1,50,32), finite, but not equal at rtol=atol=1e-5: maximum absolute difference 0.034156263; mean absolute difference 0.002108553; RMS difference 0.004010532; baseline RMS 0.15654597. Numerical difference cause not isolated; no claim of validated drop-in equivalence or robot behavior.

CPU counter test in installed Flax: module_jit(Counter.tick) returned 1,1 and original count stayed 0; nnx.jit(Counter.tick)(model) returned 1,2 and original count became 2. This demonstrates discarded vs propagated state updates.

Installed flax/nnx/transforms/compilation.py JitFn returns pure_args_out, pure_kwargs_out, pure_out; jit_wrapper reconstructs/synchronizes these using an update context. Existing module_jit returns only the method result, no updated model state. Bound model is not an explicit argument to that graph transformation; captured parameter constants are a plausible reason for compilation RAM growth, not a fully isolated causal finding.

Test containers removed on exit. No production compose or model code modified. Existing aloha2-collect was observed stopped at 08:06:55 UTC with exit 143 and OOMKilled=false, before test host OOM; benchmark did not issue commands to it.
