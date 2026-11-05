"""Compare original scan with per-microbatch JIT on identical real inputs.

Only exposes the original accumulated gradients as one extra return value.
No changes to training files, checkpoints, data sampling or W&B runs.
"""
import argparse
import ast
import sys
import dataclasses
import gc
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import train as original
from streaming_accumulation import StreamingAccumulator
from openpi.training import config, sharding, weight_loaders

ROOT = Path(__file__).resolve().parent


def instrument_reference():
    tree = ast.parse((REPO_ROOT / 'scripts/testdata/scan_train_step.py.txt').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train_step')
    assert isinstance(fn.body[-1], ast.Return)
    fn.body[-1].value.elts.append(ast.Name(id='grads', ctx=ast.Load()))
    fn.returns = None
    fn.decorator_list = []
    scope = vars(original).copy()
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), '<reference-with-grad-output>', 'exec'), scope)
    return scope['train_step']


@jax.jit
def tree_error(a, b):
    squares, refs, diffs, exact, finite = [], [], [], [], []
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        xf, yf = x.astype(jnp.float32), y.astype(jnp.float32)
        delta = xf - yf
        squares.append(jnp.sum(delta * delta))
        refs.append(jnp.sum(xf * xf))
        diffs.append(jnp.max(jnp.abs(delta)))
        exact.append(jnp.all(x == y))
        finite.append(jnp.all(jnp.isfinite(xf)) & jnp.all(jnp.isfinite(yf)))
    if not squares:
        return jnp.array([0., 0., 1., 1.])
    return jnp.array([
        jnp.max(jnp.stack(diffs)),
        jnp.sqrt(jnp.sum(jnp.stack(squares)) / jnp.maximum(jnp.sum(jnp.stack(refs)), 1e-30)),
        jnp.all(jnp.stack(exact)), jnp.all(jnp.stack(finite)),
    ])


def error(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    a = jax.device_put(a, jax.tree.map(lambda x: x.sharding, b))
    row = np.array(tree_error(a, b))
    return dict(max_abs=float(row[0]), relative_l2=float(row[1]), exact_equal=bool(row[2]), finite=bool(row[3]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params-path', required=True)
    parser.add_argument('--repo-id', default='lyl472324464/2026-10-28_insert_01-lerobot-with-rinse')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--output-dir', type=Path, default=Path('/tmp/openpi-microjit-equivalence'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original.init_logging()
    cfg = config.get_config('eii_with_rinse_cam4_fullft_all_98repo_h200x2_448_allcam_mem1s4hz_mosaic2x2_pool256')
    cfg = dataclasses.replace(
        cfg, data=dataclasses.replace(cfg.data, repo_ids=[args.repo_id]),
        weight_loader=weight_loaders.CheckpointWeightLoader(args.params_path),
        batch_size=args.batch_size, gradient_accumulation_steps=2, num_workers=4, wandb_enabled=False,
    )
    rng, init_rng = jax.random.split(jax.random.key(cfg.seed))
    mesh = sharding.make_mesh(cfg.fsdp_devices)
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    packed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, sharding.DATA_AXIS))
    jax.config.update('jax_compilation_cache_dir', str(Path('~/.cache/jax').expanduser()))
    loader = original._data_loader.create_data_loader(cfg, sharding=data, shuffle=True)
    iterator = iter(loader)
    batches = [next(iterator) for _ in range(cfg.gradient_accumulation_steps)]
    jax.block_until_ready(batches)
    print('REAL_INPUT_SHAPES', original.training_utils.array_tree_to_info(batches[0]), flush=True)
    # Reuse the very same tensors for each implementation and both checks.
    batch = jax.tree.map(lambda *xs: jnp.stack(xs), *batches)
    initial, layout = original.init_train_state(cfg, init_rng, mesh, resume=False)
    jax.block_until_ready(initial)
    # Preserve only starting weights; postpone allocation of candidate Adam
    # moments until its gradients are ready. Keep both outputs on GPU to compare.
    candidate = dataclasses.replace(
        initial, params=jax.tree.map(lambda x: x.copy(), initial.params),
        step=initial.step.copy(),
    )
    jax.block_until_ready(candidate)
    reference = jax.jit(
        lambda rng, state, batch: instrumented(cfg, rng, state, batch),
        in_shardings=(replicated, layout, packed),
        out_shardings=(layout, replicated, layout.params.filter(cfg.trainable_filter)),
        donate_argnums=(1,),
    )
    engine = StreamingAccumulator(cfg, layout, data, replicated)
    full_engine = engine
    initialize_opt = jax.jit(
        lambda params: initial.tx.init(params.filter(cfg.trainable_filter)),
        in_shardings=(layout.params,), out_shardings=layout.opt_state,
    )
    results = dict(batch_size=cfg.batch_size, ga=2, effective_batch=256, workers=4,
                   devices=len(jax.devices()), real_input=True, rows=[])
    for step in range(1):
        started = time.perf_counter()
        with sharding.set_mesh(mesh):
            expected, expected_info, expected_grads = reference(rng, initial, batch)
            jax.block_until_ready((expected, expected_info, expected_grads))
            print('REFERENCE_DONE', step, time.perf_counter()-started, flush=True)
            # These fields are not read by the microbatch JIT. Share the
            # reference buffers temporarily so this is the exact production
            # JIT signature without allocating another Adam/EMA state yet.
            candidate = dataclasses.replace(
                candidate, opt_state=expected.opt_state, ema_params=expected.ema_params,
            )
            loss, grads = engine.initialize(candidate)
            for i, micro in enumerate(batches):
                loss, grads = engine.accumulate(rng, candidate, micro, jnp.asarray(i, jnp.int32), loss, grads)
            jax.block_until_ready((loss, grads))
            gradient_error = error(expected_grads, grads)
            candidate = dataclasses.replace(
                candidate, opt_state=initialize_opt(candidate.params),
                ema_params=jax.tree.map(lambda x: x.copy(), candidate.params) if cfg.ema_decay is not None else None,
            )
            actual, actual_info = full_engine.update(candidate, loss, grads)
            jax.block_until_ready((actual, actual_info))
            row = dict(step=step, loss_scan=float(expected_info['loss']), loss_microjit=float(actual_info['loss']),
                       grads=gradient_error, params=error(expected.params, actual.params),
                       opt_state=error(expected.opt_state, actual.opt_state),
                       ema=error(expected.ema_params, actual.ema_params),
                       metrics=error(expected_info, actual_info))
        print('COMPARISON', json.dumps(row), flush=True)
        results['rows'].append(row)
        (args.output_dir/'numerical_result.json').write_text(json.dumps(results, indent=2))
        del expected_grads, grads
        del candidate, actual
    print('FINISHED', json.dumps(results), flush=True)


if __name__ == '__main__':
    instrumented = instrument_reference()
    main()
