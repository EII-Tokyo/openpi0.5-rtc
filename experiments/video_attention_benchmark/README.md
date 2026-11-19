# Optional cuDNN vision attention

`Pi0Config(vision_attention_backend="cudnn")` enables fused spatial and causal
temporal attention for action inference. The default is `"native"`. There is no
1024-token cutoff: both 224- and 448-pixel images are supported by the tested
JAX 0.5.3 / Flax 0.10.2 CUDA environment. Boolean masks are forwarded to JAX's
cuDNN SDPA implementation. Unsupported shapes, dtypes or CUDA combinations fail
visibly; there is no silent fallback. The fused path requires BF16 or FP16.

Loss computation, including validation loss, explicitly disables fusion. Image
training also uses native attention. Parameter names and shapes are unchanged;
existing checkpoints load without conversion. Training launchers and named
training configurations are unchanged.

Flax's query pre-scaling is retained, but fused BF16 reductions change numerical
rounding. This is **not bitwise equivalent**. On the measured 448-pixel fixtures,
the maximum normalized-action difference was 0.00390625; on the distinct
224-pixel checkpoint it reached 0.01309204. These are numerical checks, not robot
success-rate validation. Validate task behavior before deployment.

## Reproduce timings

From this repository, with an idle CUDA GPU and its existing environment:

```bash
PYTHONPATH=src python experiments/video_attention_benchmark/benchmark.py \
  --config-name CONFIG --checkpoint-dir CHECKPOINT \
  --backend cudnn --frames 16 --output /tmp/cudnn.json
```

Run the same command with `--backend native` and a different output path for
comparison. The script writes timings, normalized action arrays (`.npy`) and
compiled HLO (`.hlo.txt`); it checks for cuDNN FMHA in the fused executable.
Use a distinct output path for each case. Changing the model config here is
limited to this offline inference harness, not training.

The recorded protocol uses one GPU, batch 1, four camera streams, 448x448
images, 256 pooled tokens per camera (1024 total), 200 text tokens, horizon 50
and 10 denoising steps. Inputs are synthetic normalized arrays already on the
GPU. Timings block on device completion and exclude loading, JIT, image
preprocessing, host transfers and networking. Native and fused runs use the
same checkpoint and random inputs. `h100_latency_summary.csv` contains the
completed H100 SXM curve; the RTX 5090 column is a native reference.

At 16 frames per camera, H100 native/fused medians were 278.624/209.523 ms.
RTX 5090 native/fused medians were 583.909/459.827 ms. A full-model smoke test
through the final config switch measured 459.283 ms on RTX 5090, with FMHA
present in HLO. These results use our pi0.5 model; the MEM paper's Figure 3
uses pi0.6, so similar latency does not establish an exact paper reproduction.

## Validation

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python -m pytest -q \
  src/openpi/models/cudnn_attention_test.py \
  src/openpi/models/siglip_pooling_test.py
```

The 19 passing CPU cases cover scanned/unscanned parameter compatibility,
native training values and gradients, loss-path opt-out, mask forwarding,
query scaling, 224-token-grid routing and unsupported-option errors. They do
not emulate GPU fused arithmetic. The separate full-model GPU smoke test
loads the existing checkpoint and verifies actual FMHA compilation.
