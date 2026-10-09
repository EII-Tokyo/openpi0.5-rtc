# GPU allocator experiment — 103, 2026-09-17

RTX 5090, JAX 0.5.3, existing module_jit, four cameras at 448, checkpoint 3000, batch 1, 10 denoising steps. Independent processes; three warmups and 60 timed/blocking calls each for plain and RTC. GPU-resident synthetic inputs. No runtime/ROS started and no production configuration changed. All test containers exited; existing aloha2-collect left running.

| Mode | Process GPU memory after RTC (MiB) | JAX pool (GiB) | Plain mean (ms) | RTC mean (ms) | Outcome |
|---|---:|---:|---:|---:|---|
| default, fraction=.9, preallocate=false | 17010 | 16.0293 | 192.663 | 230.136 | passed |
| default, fraction=.32, preallocate=false | 10862 | 10.023 | 191.537 | 228.947 | passed |
| platform, preallocate=false | 7038 | unavailable | 192.848 | 230.957 | passed |
| default, fraction=.28, preallocate=false | unavailable | unavailable | unavailable | unavailable | OOM in initial plain warmup, requested 1303022792 bytes |

Process GPU memory is nvidia-smi per-compute-process residency after inference, not transient peak. Platform memory_stats() returns None: no peak or pool measurement available, do not interpret this as zero peak usage. BFC cumulative peaks include initialization/compilation/autotuning/warmup. Same GPU memory fraction may not work on other cards, batch sizes, wrappers or shapes. One process run per mode; no clock locking. Values near 1% latency difference are not evidence of meaningful speedup.

Output check: all successful outputs finite, but RTC outputs are not bitwise identical across fresh compilations. Against baseline, max absolute difference is 0.0284129977 for fraction=.32 and 0.0329929590 for platform. Fixed NumPy seed42 and JAX key123 used; exact post-transform input hashes not collected in this experiment. Cause not isolated; do not claim strict numerical equivalence or validated real-robot behavior.

Recommended followup candidate: retain BFC with .32 fraction (~10 GiB pool), which gives a margin over measured live peaks. Platform is a lower-residency experiment but has no broadly guaranteed latency advantage. No production rollout performed.

## Reproduce (103)

From /home/eii/openpi0.5-rtc, use docker compose -f docker-compose.yml -f benchmarks/jit_compare_limits.yml run --rm --no-deps -T, followed by per-case environment overrides, then openpi_server /.venv/bin/python /app/benchmarks/benchmark_allocator.py LABEL.

- Baseline: -e XLA_PYTHON_CLIENT_ALLOCATOR=default -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
- Cap32: same but fraction=0.32
- Platform: -e XLA_PYTHON_CLIENT_ALLOCATOR=platform -e XLA_PYTHON_CLIENT_PREALLOCATE=false
- Cap28 (known OOM): default allocator, preallocate=false, fraction=0.28

The driver requires benchmarks/benchmark_batch_profile.py and writes into benchmarks/allocator_20260917 (create directory first). Host RAM limit: 40 GiB. JAX docs: https://docs.jax.dev/en/latest/gpu_memory_allocation.html
