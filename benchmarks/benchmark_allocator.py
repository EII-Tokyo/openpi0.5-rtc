"""Use the same resident-input inference benchmark under different allocators."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import numpy as np

label = sys.argv[1]
sys.argv = ['benchmark_batch_profile.py', '--batch', '1', '--repeats', '60']
g = runpy.run_path('/app/benchmarks/benchmark_batch_profile.py', run_name='__main__')
result = g['result']
result['label'] = label
result['env'] = {k: os.environ.get(k) for k in (
    'XLA_PYTHON_CLIENT_ALLOCATOR', 'XLA_PYTHON_CLIENT_MEM_FRACTION',
    'XLA_PYTHON_CLIENT_PREALLOCATE')}
result['nvidia_smi'] = subprocess.check_output([
    'nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True)
out = Path('/app/benchmarks/allocator_20260917')
np.save(out / (label + '_rtc.npy'), np.asarray(g['output']))
(out / (label + '.json')).write_text(json.dumps(result, indent=2))
print('ALLOCATOR_RESULT', json.dumps(result), flush=True)
