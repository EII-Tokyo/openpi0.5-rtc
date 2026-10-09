"""GPU-resident batched inference; never connects to robot/ROS."""
import argparse
import ctypes
import json
import logging
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from openpi.training import config
from openpi.serving import policy_config
from openpi.models import model

p = argparse.ArgumentParser()
p.add_argument('--batch', type=int, default=1)
p.add_argument('--profile', action='store_true')
p.add_argument('--repeats', type=int, default=30)
a = p.parse_args()
logging.basicConfig(level=logging.INFO, force=True)
cfg = config.get_config('eii_data_system_with_rinse_cam4_fullft_h200_from_april_original_30repo')
ckpts = list((Path('/app/checkpoints') / cfg.name).glob('*/3000/params'))
assert len(ckpts) == 1
policy = policy_config.create_trained_policy(cfg, ckpts[0].parent)
rng = np.random.default_rng(42)
raw = {'state': np.zeros(14, np.float32), 'images': {k: rng.integers(0,256,(3,480,640),dtype=np.uint8)
       for k in cfg.data.transform_pipeline.raw_image_keys},
       'task': 'twist off the bottle cap and rinse the bottle', 'subtask': 'good action'}
inputs = policy._input_transform(raw)
inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
obs = policy._observation_transform(model.Observation.from_dict(inputs))
obs = jax.tree.map(lambda x: jnp.repeat(x, a.batch, axis=0), obs)
jax.block_until_ready(obs)
prev = jnp.asarray(rng.normal(size=(a.batch,50,32)).astype(np.float32))
key = jax.random.key(123)
def run(mode):
    if mode == 'plain':
        return policy._sample_action_chunk(key, obs, denoising_steps=10)
    return policy._sample_action_chunk_with_inference_time_rtc(
        key, prev, obs, denoising_steps=10, replan_start_step=25, handoff_delay_steps=10)

result = {'batch': a.batch, 'resolution':448, 'cameras':4, 'gpu':jax.devices()[0].device_kind,
          'repeats':a.repeats, 'jax':jax.__version__, 'latency':{}}
for mode in (('rtc',) if a.profile else ('plain','rtc')):
    for _ in range(3): jax.block_until_ready(run(mode))
    print('WARM',mode,flush=True)
    if a.profile:
        cuda = ctypes.CDLL('libcuda.so.1')
        assert cuda.cuProfilerStart() == 0
        for _ in range(3): jax.block_until_ready(run(mode))
        assert cuda.cuProfilerStop() == 0
    else:
        values=[]
        for _ in range(a.repeats):
            start=time.perf_counter(); output=run(mode); jax.block_until_ready(output)
            values.append((time.perf_counter()-start)*1000)
        assert np.isfinite(np.asarray(output)).all()
        ms=float(np.mean(values))
        result['latency'][mode]={'mean_ms':ms,'median_ms':float(np.median(values)),
            'p95_ms':float(np.percentile(values,95)), 'samples_per_second':a.batch*1000/ms}
        result.setdefault('memory',{})[mode]=jax.devices()[0].memory_stats()
        print('MEASURE',mode,json.dumps(result['latency'][mode]),flush=True)
print('RESULT_JSON '+json.dumps(result),flush=True)
