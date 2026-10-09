"""Model-only JIT comparison. Run each wrapper in a fresh process; no ROS/runtime.

Uses the project's Python environment. Default checkpoint discovery is for Docker.
The bound nnx wrapper previously exhausted host RAM: always use a memory limit.
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import time

import flax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi.shared import nnx_utils
from openpi.training import config
from openpi.serving import policy_config
from openpi.models import model


def digest(tree):
    h = hashlib.sha256()
    for value in jax.tree.leaves(tree):
        array = np.asarray(value)
        h.update(str((array.shape, array.dtype)).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wrapper', choices=['module', 'nnx', 'nnx_explicit'], required=True)
    p.add_argument('--config', default='eii_data_system_with_rinse_cam4_fullft_h200_from_april_original_30repo')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output-dir', type=Path, default=Path('/app/benchmarks/jit_standalone_results'))
    p.add_argument('--repeats', type=int, default=30)
    p.add_argument('--test-layer-swap', action='store_true',
                   help='After JIT warmup/timing, replace action_out_proj with a zero layer and compare.')
    a = p.parse_args()
    if a.repeats < 1:
        p.error('--repeats must be positive')
    logging.basicConfig(level=logging.INFO, force=True)
    if a.wrapper == 'nnx':
        nnx_utils.module_jit = nnx.jit
    elif a.wrapper == 'nnx_explicit':
        def explicit_jit(method, *args, **kwargs):
            compiled = nnx.jit(method.__func__, *args, **kwargs)
            def wrapped(*inputs, **kwinputs):
                return compiled(method.__self__, *inputs, **kwinputs)
            return wrapped
        nnx_utils.module_jit = explicit_jit
    cfg = config.get_config(a.config)
    checkpoint = a.checkpoint
    if checkpoint is None:
        candidates = list((Path('/app/checkpoints') / cfg.name).glob('*/3000/params'))
        if len(candidates) != 1:
            p.error('Specify --checkpoint: expected exactly one checkpoint 3000')
        checkpoint = candidates[0].parent
    policy = policy_config.create_trained_policy(cfg, checkpoint)
    rng = np.random.default_rng(42)
    raw = {
        'state': np.zeros(14, np.float32),
        'images': {k: rng.integers(0, 256, (3, 480, 640), dtype=np.uint8)
                   for k in cfg.data.transform_pipeline.raw_image_keys},
        'task': 'twist off the bottle cap and rinse the bottle',
        'subtask': 'good action',
    }
    inputs = policy._input_transform(raw)
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
    obs = policy._observation_transform(model.Observation.from_dict(inputs))
    prev = jnp.asarray(rng.normal(size=(1, 50, 32)).astype(np.float32))
    key = jax.random.key(123)
    # Diagnostic noise hash: RTC generates its noise internally using this key.
    noise = jax.random.normal(key, (1, 50, 32))
    jax.block_until_ready((obs, prev, noise))
    result = {'wrapper': a.wrapper, 'jax': jax.__version__, 'flax': flax.__version__,
              'checkpoint': str(checkpoint), 'repeats': a.repeats,
              'input_hash': digest(obs), 'prev_hash': digest(prev),
              'noise_hash': digest(noise), 'modes': {}}
    print('INPUT_HASHES', json.dumps(result), flush=True)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in ('plain', 'rtc'):
        def run():
            if mode == 'plain':
                return policy._sample_action_chunk(key, obs, denoising_steps=10)
            return policy._sample_action_chunk_with_inference_time_rtc(
                key, prev, obs, denoising_steps=10,
                replan_start_step=25, handoff_delay_steps=10)
        for _ in range(3):
            jax.block_until_ready(run())
        first = np.asarray(jax.block_until_ready(run())).copy()
        durations = []
        for _ in range(a.repeats):
            start = time.perf_counter()
            output = jax.block_until_ready(run())
            durations.append((time.perf_counter() - start) * 1000)
        last = np.asarray(output)
        assert np.isfinite(last).all()
        np.save(a.output_dir / f'{a.wrapper}_{mode}.npy', last)
        result['modes'][mode] = {
            'mean_ms': float(np.mean(durations)),
            'p95_ms': float(np.percentile(durations, 95)),
            'repeat_max_abs_diff': float(np.max(np.abs(first - last))),
            'memory': jax.devices()[0].memory_stats(),
        }
        print('MEASURE', mode, json.dumps(result['modes'][mode]), flush=True)
        if a.test_layer_swap:
            # Keep timing/memory measurements above free of the replacement test.
            before = np.asarray(jax.block_until_ready(run())).copy()
            old_layer = policy._model.action_out_proj
            layer_def, layer_state = nnx.split(old_layer)
            zero_layer = nnx.merge(layer_def, jax.tree.map(jnp.zeros_like, layer_state))
            try:
                # Replace the actual layer, not an unrelated attribute named "state".
                policy._model.action_out_proj = zero_layer
                after = np.asarray(jax.block_until_ready(run())).copy()
                swap = {
                    'layer': 'action_out_proj',
                    'exact_equal': bool(np.array_equal(before, after)),
                    'max_abs_diff': float(np.max(np.abs(before - after))),
                    'finite': bool(np.isfinite(after).all()),
                }
                result['modes'][mode]['layer_swap'] = swap
                print('LAYER_SWAP', mode, json.dumps(swap), flush=True)
                np.save(a.output_dir / f'{a.wrapper}_{mode}_after_swap.npy', after)
                if a.wrapper == 'module':
                    np.testing.assert_array_equal(before, after)
            finally:
                policy._model.action_out_proj = old_layer
    (a.output_dir / f'{a.wrapper}.json').write_text(json.dumps(result, indent=2))
    print('RESULT_JSON', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
