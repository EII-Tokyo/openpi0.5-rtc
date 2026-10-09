"""Compare JIT wrappers in isolated processes; no robot interfaces."""
import argparse
import json
import runpy
import subprocess
import sys
import numpy as np
import flax
from flax import nnx
from openpi.shared import nnx_utils

p = argparse.ArgumentParser()
p.add_argument('--wrapper', choices=['module', 'nnx', 'nnx_explicit'], required=True)
a = p.parse_args()
if a.wrapper == 'nnx':
    nnx_utils.module_jit = nnx.jit
elif a.wrapper == 'nnx_explicit':
    def explicit_jit(method, *args, **kwargs):
        compiled = nnx.jit(method.__func__, *args, **kwargs)
        def wrapped(*inputs, **kwinputs):
            return compiled(method.__self__, *inputs, **kwinputs)
        return wrapped
    nnx_utils.module_jit = explicit_jit
print('WRAPPER', a.wrapper, 'FLAX', flax.__version__, flush=True)
sys.argv = ['benchmark_batch_profile.py', '--batch', '1', '--repeats', '30']
g = runpy.run_path('/app/benchmarks/benchmark_batch_profile.py', run_name='__main__')
np.save('/app/benchmarks/jit_compare_20260916/' + a.wrapper + '_output.npy',
        np.asarray(g['output']))
print('NVIDIA_SMI', subprocess.check_output(['nvidia-smi',
    '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True), flush=True)
