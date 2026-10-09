import json
import os
from pathlib import Path
import subprocess

root = Path('/home/eii/openpi0.5-rtc')
out = root / 'benchmarks/resolution_scaling_20260916'
out.mkdir(parents=True, exist_ok=True)
cfg = 'eii_data_system_with_rinse_cam4_fullft_h200_from_april_original_30repo'
paths = list((root / 'checkpoints' / cfg).glob('*/3000/params'))
assert len(paths) == 1
checkpoint = '/app/' + str(paths[0].parent.relative_to(root))
for size in (56, 112, 224, 448):
    label = f'rinse4_3000_{size}_cam4'
    cmd = ['docker', 'compose', '-f', 'docker-compose.yml', 'run', '--rm', '--no-deps', '-T',
           '-e', 'TF_CPP_MIN_LOG_LEVEL=0', '-e', 'TF_CPP_VMODULE=bfc_allocator=1',
           '--name', 'openpi-resolution-benchmark', 'openpi_server', '/.venv/bin/python',
           '/app/benchmarks/benchmark_inference_matrix.py', '--config', cfg, '--checkpoint', checkpoint,
           '--label', label, '--resolution', str(size), '--cameras', '4', '--resize-pos', '--split-timing', '--repeats', '100']
    print('START', label, flush=True)
    path = out / (label + '.log')
    with path.open('w') as f:
        completed = subprocess.run(cmd, cwd=root, env=dict(os.environ, PWD=str(root)), stdout=f, stderr=subprocess.STDOUT)
    lines = [x.removeprefix('RESULT_JSON ') for x in path.read_text().splitlines() if x.startswith('RESULT_JSON ')]
    row = json.loads(lines[-1]) if lines else {'label': label, 'error': path.read_text()[-2500:]}
    row['exit_code'] = completed.returncode
    with (out / 'results.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    print('DONE', label, json.dumps(row.get('latency', row)), flush=True)
