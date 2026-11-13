# Archived temporal-attention comparison

These manual experiments compare gated temporal attention with `nn.cond`,
including bf16/fp32 output and gradient differences. They are research tools,
not imports used by training or inference. Keep the historical `siglip_*.py`
variants: the comparison scripts refer to them as baselines.

The `*_comparison.json` files retain the numeric summaries, and gradient
`manifest.json` files retain parameter paths, shapes and dtypes. On 2026-11-13,
13,167,023,616 bytes of generated gradient `.npy` arrays were removed after these
summaries were produced. The arrays are ignored by Git. Regenerate them with
`bench.py --full-grads` before rerunning `compare_full.py`; the retained manifests
alone are not complete gradient dumps.

These scripts expect a matching historical source checkout at `/app`, a writable
output mount at `/out`, and a free GPU. Some `test_*.py` files execute immediately
when imported. Run them manually only, never as part of a live robot deployment
or a general recursive pytest invocation. No GPU benchmarks were run during the
2026-11-13 source cleanup.

Eight duplicate per-dtype `siglip_*.py` snapshots were consolidated into the three
unique source variants at this directory root, verified by SHA256 before removal.
