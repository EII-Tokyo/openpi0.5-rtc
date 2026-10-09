"""Isolated inference-only matrix; never creates a robot environment."""
import argparse
import dataclasses as dc
import json
import logging
import os
from pathlib import Path
import subprocess
import time


def gpu_mib():
    return int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()[0])


def case(a):
    import numpy as np
    import jax
    from openpi.training import config
    from openpi.serving import policy_config
    if getattr(a, 'resize_pos', False):
        from openpi.models import model as model_module
        original_restore = model_module.restore_params
        def resized_restore(*args, **kwargs):
            params = original_restore(*args, **kwargs)
            old = params['PaliGemma']['img']['pos_embedding']
            grid = int(old.shape[1] ** 0.5)
            target = a.resolution // 14
            if grid != target:
                assert grid % target == 0
                factor = grid // target
                value = np.asarray(old, dtype=np.float32).reshape(1, target, factor, target, factor, old.shape[-1])
                value = value.mean(axis=(2, 4)).reshape(1, target * target, old.shape[-1])
                params['PaliGemma']['img']['pos_embedding'] = jax.numpy.asarray(value, dtype=old.dtype)
                print('EXPERIMENTAL_POSITION_RESAMPLE', old.shape, value.shape, flush=True)
            return params
        model_module.restore_params = resized_restore

    logging.basicConfig(level=logging.INFO, force=True)
    logging.getLogger('openpi').setLevel(logging.INFO)
    cfg = config.get_config(a.config)
    original = {'resolution': cfg.model.image_resolution, 'cameras': cfg.data.transform_pipeline.raw_image_keys}
    pipe = dc.replace(cfg.data.transform_pipeline, include_low=a.cameras == 4,
                      image_resolution=(a.resolution, a.resolution), video_memory_num_frames=1)
    cfg = dc.replace(cfg, model=dc.replace(cfg.model, image_resolution=(a.resolution, a.resolution),
                     image_keys=pipe.raw_image_keys), data=dc.replace(cfg.data, transform_pipeline=pipe))
    result = {'label': a.label, 'config': a.config, 'checkpoint': a.checkpoint,
              'resolution': a.resolution, 'cameras': a.cameras, 'native': original,
              'denoising_steps': 10, 'repeats': getattr(a, 'repeats', 20),
              'experimental_position_resample': getattr(a, 'resize_pos', False),
              'jax_version': jax.__version__, 'memory_mib': {'baseline': gpu_mib()}}
    start = time.perf_counter()
    policy = policy_config.create_trained_policy(cfg, a.checkpoint,
            sample_kwargs={'denoising_steps': 10, 'replan_start_step': 25, 'handoff_delay_steps': 10})
    result['load_seconds'] = time.perf_counter() - start
    original_transform = policy._observation_transform
    def checked_transform(observation):
        observation = original_transform(observation)
        if 'actual_observation' not in result:
            result['actual_observation'] = {
                'images': {k: list(v.shape) for k, v in observation.images.items()},
                'masks': {k: np.asarray(v).tolist() for k, v in observation.image_masks.items()},
                'text_shape': list(observation.tokenized_prompt.shape),
                'valid_text_tokens': np.asarray(observation.tokenized_prompt_mask).sum(axis=-1).tolist(),
            }
            print('OBSERVATION', json.dumps(result['actual_observation']), flush=True)
        return observation
    policy._observation_transform = checked_transform

    def memory(stage):
        jax.effects_barrier()
        result['memory_mib'][stage] = gpu_mib()
        result.setdefault('jax_memory', {})[stage] = jax.devices()[0].memory_stats()
        print('STAGE', stage, result['memory_mib'][stage], flush=True)

    memory('loaded')
    rng = np.random.default_rng(42)
    obs = {'state': np.zeros(14, np.float32),
           'images': {k: rng.integers(0, 256, (3, 480, 640), dtype=np.uint8) for k in pipe.raw_image_keys},
           'task': 'twist off the bottle cap and rinse the bottle', 'subtask': 'good action'}
    previous = rng.normal(size=(50, 32)).astype(np.float32)
    prefix = np.zeros_like(previous)
    prefix[:10] = previous[25:35]
    # Same warmup order and three compiled paths as the deployed serve_policy.py.
    policy.infer(obs, prev_action=previous, chunking_mode='inference_time')
    memory('rtc_warm')
    policy.infer(obs, action_prefix=prefix, handoff_delay_steps=10, chunking_mode='training_time')
    memory('training_rtc_warm')
    policy.infer(obs, chunking_mode='sync')
    memory('all_warm')
    result['latency'] = {}
    device_times = []
    if getattr(a, 'split_timing', False):
        def wrap_sample(fn):
            def wrapped(*args, **kwargs):
                t = time.perf_counter()
                value = fn(*args, **kwargs)
                jax.block_until_ready(value)
                device_times.append((time.perf_counter() - t) * 1000)
                return value
            return wrapped
        for field in ('_sample_action_chunk', '_sample_action_chunk_with_inference_time_rtc'):
            setattr(policy, field, wrap_sample(getattr(policy, field)))
    for mode in ('plain', 'rtc'):
        times = []
        device_times.clear()
        for i in range(result['repeats'] + 5):
            t = time.perf_counter()
            out = policy.infer(obs, chunking_mode='inference_time',
                               prev_action=previous if mode == 'rtc' else None)
            # infer converts device results to numpy; explicit blocking safeguard.
            jax.block_until_ready(out)
            elapsed = (time.perf_counter() - t) * 1000
            assert np.isfinite(out['actions']).all()
            if i >= 5:
                times.append(elapsed)
        result['latency'][mode] = {'mean_ms': float(np.mean(times)), 'median_ms': float(np.median(times)),
                                   'p95_ms': float(np.percentile(times, 95)), 'min_ms': min(times), 'max_ms': max(times)}
        if device_times:
            result['latency'][mode]['synchronized_model_call_mean_ms'] = float(np.mean(device_times[5:]))
            result['latency'][mode]['other_mean_ms'] = float(np.mean(times) - np.mean(device_times[5:]))
        memory('after_' + mode)
    print('RESULT_JSON ' + json.dumps(result), flush=True)


def matrix():
    root = Path('/home/eii/openpi0.5-rtc')
    out = root / 'benchmarks/inference_matrix_20260916'
    out.mkdir(parents=True, exist_ok=True)
    choices = [
        ('demo19000', 'eii_data_system_without_rinse_cam3_fullft_h200_return_home_29repo', '19000'),
        ('no_rinse20000', 'eii_data_system_without_rinse_cam3_fullft_h200_have_cap_water_return_home_drop_empty_x1_original_45repo', '20000'),
        ('rinse3_5000', 'eii_data_system_with_rinse_cam3_fullft_h200_original_59repo', '5000'),
        ('rinse4_3000', 'eii_data_system_with_rinse_cam4_fullft_h200_from_april_original_30repo', '3000'),
    ]
    for label, cfg, step in choices:
        paths = list((root / 'checkpoints' / cfg).glob('*/' + step + '/params'))
        assert len(paths) == 1, paths
        checkpoint = '/app/' + str(paths[0].parent.relative_to(root))
        for resolution, cams in [(224, 3), (224, 4), (448, 3), (448, 4)]:
            name = f'{label}_{resolution}_cam{cams}'
            command = ['docker', 'compose', '-f', 'docker-compose.yml', 'run', '--rm', '--no-deps', '-T',
                       '--name', 'openpi-inference-benchmark', 'openpi_server', '/.venv/bin/python',
                       '/app/benchmarks/benchmark_inference_matrix.py', '--config', cfg,
                       '--checkpoint', checkpoint, '--label', name,
                       '--resolution', str(resolution), '--cameras', str(cams)]
            print('START', name, flush=True)
            with (out / (name + '.log')).open('w') as f:
                proc = subprocess.Popen(command, cwd=root, env=dict(os.environ, PWD=str(root)),
                                        stdout=f, stderr=subprocess.STDOUT)
                samples = []
                while proc.poll() is None:
                    samples.append(gpu_mib())
                    time.sleep(0.5)
            log = (out / (name + '.log')).read_text()
            results = [s[len('RESULT_JSON '):] for s in log.splitlines() if s.startswith('RESULT_JSON ')]
            row = json.loads(results[-1]) if results else {'label': name, 'error': log[-2500:]}
            row['exit_code'] = proc.returncode
            row['sampled_peak_gpu_mib'] = max(samples, default=0)
            with (out / 'results.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
            print('DONE', name, json.dumps(row.get('latency', row.get('error'))), 'peak', row['sampled_peak_gpu_mib'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--matrix', action='store_true')
    p.add_argument('--config')
    p.add_argument('--checkpoint')
    p.add_argument('--label')
    p.add_argument('--resolution', type=int)
    p.add_argument('--cameras', type=int)
    p.add_argument('--resize-pos', action='store_true')
    p.add_argument('--split-timing', action='store_true')
    p.add_argument('--repeats', type=int, default=20)
    args = p.parse_args()
    matrix() if args.matrix else case(args)
