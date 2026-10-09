import json
import os
from pathlib import Path
import subprocess
root=Path('/home/eii/openpi0.5-rtc')
out=root/'benchmarks/batch_profile_20260916'
out.mkdir(parents=True,exist_ok=True)
base=['docker','compose','-f','docker-compose.yml','run','--rm','--no-deps','-T',
      '--name','openpi-batch-benchmark']
script=['/.venv/bin/python','/app/benchmarks/benchmark_batch_profile.py']
for b in (1,2,4,8):
    name=f'batch{b}'
    print('START',name,flush=True)
    log=out/(name+'.log')
    with log.open('w') as f:
        proc=subprocess.run(base+['openpi_server']+script+['--batch',str(b)],cwd=root,
            env=dict(os.environ,PWD=str(root)),stdout=f,stderr=subprocess.STDOUT)
    rows=[x[len('RESULT_JSON '):] for x in log.read_text().splitlines() if x.startswith('RESULT_JSON ')]
    row=json.loads(rows[-1]) if rows else {'batch':b,'error':log.read_text()[-2000:]}
    row['exit_code']=proc.returncode
    with (out/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print('DONE',name,json.dumps(row.get('latency',row)),flush=True)
    if proc.returncode:break
print('START nsys',flush=True)
cmd=base+['-v','/opt/nvidia/nsight-systems:/opt/nvidia/nsight-systems:ro','openpi_server',
    '/opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/nsys','profile',
    '--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi',
    '--capture-range-end=stop','--kill=none','--force-overwrite=true',
    '--output=/app/benchmarks/batch_profile_20260916/rtc_batch1']+script+['--profile']
with (out/'nsys.log').open('w') as f:
    proc=subprocess.run(cmd,cwd=root,env=dict(os.environ,PWD=str(root)),stdout=f,stderr=subprocess.STDOUT)
print('DONE nsys',proc.returncode,flush=True)
